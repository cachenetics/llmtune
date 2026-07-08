// SPDX-License-Identifier: GPL-2.0-only
//! Node-scoped command handlers (`llmtune node <op>`, `doctor`).

use crate::cli::{ApiKeyCmd, AuthAction, ExposeAction, IdentityAction, ServerAction};
use crate::config::Config;
use crate::fmt::{fit, fmt_ts, padw};
use crate::{bench, doctor, identity, llama, mem, model, nodeops, settings, swap, transport};
use anyhow::Result;

/// Resolve the `--node` selector: a named fleet node, else the local node.
pub(crate) fn resolve_target<'a>(
    cfg: &'a Config,
    sel: Option<&str>,
) -> Result<&'a crate::config::Node> {
    match sel {
        None => Ok(cfg.local_node()),
        Some(name) => cfg.node(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown node '{name}' - configured: {}",
                cfg.nodes
                    .iter()
                    .map(|n| n.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }),
    }
}

/// Guard for node-local settings/hardware surfaces that have no remote wire:
/// `--node <remote>` is rejected with a pointer at running it ON the node.
pub(crate) fn reject_remote(cfg: &Config, sel: Option<&str>, what: &str) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    if sel.is_some() && node.transport != crate::config::Transport::Local {
        anyhow::bail!(
            "`{what}` is node-local (settings/hardware on the box it runs on) - \
             run it ON `{}` (e.g. over ssh), not via --node",
            node.name
        );
    }
    Ok(())
}

/// `node fit` - the largest context a model can run at each KV-cache quant, from
/// weights + the live usable UMA budget. Answers "specify a KV quant, get the ctx".
pub(crate) fn cmd_node_fit(
    cfg: &Config,
    model: Option<&str>,
    kv: Option<&str>,
    json: bool,
) -> Result<()> {
    let node = cfg.local_node();
    // Explicit model, else whatever's currently served.
    let m = match model {
        Some(q) => nodeops::resolve_model(node, q)?,
        None => {
            let served = llama::props(&node.llama_url)
                .and_then(|p| llama::name_from_props(&p))
                .ok_or_else(|| {
                    anyhow::anyhow!("no model given and none served - pass a model name")
                })?;
            nodeops::resolve_model(node, &served)?
        }
    };
    let dims = model::read_dims(&m.path)
        .ok_or_else(|| anyhow::anyhow!("`{}` header lacks the dims to size a KV cache", m.name))?;
    let budget =
        mem::read_uma().ok_or_else(|| anyhow::anyhow!("no UMA budget - run on the BC-250"))?;
    // Memory left for KV once weights are resident.
    let kv_budget = budget.usable() as i64 - m.size_bytes as i64;
    let quants: Vec<&str> = match kv {
        Some(k) => vec![k],
        None => vec!["f16", "q8_0", "q4_0"],
    };
    let rows: Vec<(&str, u64)> = quants
        .iter()
        .map(|q| (*q, mem::max_ctx(&dims, kv_budget, q)))
        .collect();

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": m.name,
                "arch": m.arch,
                "weights_gib": mem::gib(m.size_bytes),
                "usable_gib": mem::gib(budget.usable()),
                "kv_budget_gib": mem::gib_i(kv_budget),
                "trained_ctx": dims.ctx_train,
                "fit": rows.iter().map(|(q, c)| serde_json::json!({"kv": q, "max_ctx": c})).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    println!(
        "{} [{}]   weights {:.1} GiB   ·   free {:.1} GiB   ·   KV budget {:.1} GiB",
        m.name,
        m.arch,
        mem::gib(m.size_bytes),
        mem::gib(budget.usable()),
        mem::gib_i(kv_budget.max(0)),
    );
    if let Some(t) = dims.ctx_train {
        println!("trained context: {t}");
    }
    println!("{}", "-".repeat(36));
    for (q, c) in &rows {
        if *c == 0 {
            println!("  {q:6}  won't fit (weights over budget)");
        } else {
            let capped = dims.ctx_train.is_some_and(|t| *c >= t);
            println!(
                "  {q:6}  {c:>8} ctx{}",
                if capped { "  (trained max)" } else { "" }
            );
        }
    }
    Ok(())
}

/// `node gpu` - live GPU telemetry (the reading the TUI header shows).
pub(crate) fn cmd_node_gpu(cfg: &Config, sel: Option<&str>, json: bool) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    match transport::for_node(node)?.telemetry() {
        Some(t) => {
            if json {
                // Serialize the DTO directly - the same shape the SSH transport
                // parses (one wire contract).
                println!("{}", serde_json::to_string_pretty(&t)?);
            } else {
                let power = t
                    .power_w
                    .map(|w| format!("{w:.1} W"))
                    .unwrap_or_else(|| "n/a".into());
                println!(
                    "gfx {} MHz   uclk {} MHz   temp {:.1} C   power {power}",
                    t.gfxclk_mhz, t.uclk_mhz, t.temp_c
                );
            }
        }
        None => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({"error": "telemetry unavailable (amdgpu debugfs?)"})
                );
            } else {
                println!("GPU telemetry unavailable - need amdgpu debugfs (run on the BC-250)");
            }
        }
    }
    Ok(())
}

/// `node server <start|stop|restart|status>` - control the llama-server unit.
/// Stopping it frees the GPU so manual tests (CU benches, tuning) run without
/// inference contention; start brings inference back.
pub(crate) fn cmd_node_server(
    cfg: &Config,
    sel: Option<&str>,
    action: ServerAction,
    json: bool,
) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    let unit = node.llama_unit.clone();
    if node.transport != crate::config::Transport::Local {
        // Remote node: the transport's server_ctl / status carry the op.
        return cmd_node_server_remote(node, &unit, action, json);
    }
    let is_active = || {
        std::process::Command::new("systemctl")
            .args(["is-active", "--quiet", &unit])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    match action {
        ServerAction::Status => {
            let active = is_active();
            if json {
                println!("{{\"unit\":\"{unit}\",\"active\":{active}}}");
            } else {
                println!("{unit}: {}", if active { "running" } else { "stopped" });
            }
        }
        ServerAction::Stop => {
            swap::sudo(&["systemctl", "stop", &unit])?;
            if json {
                println!("{}", serde_json::json!({"unit": unit, "result": "stopped"}));
            } else {
                println!("stopped {unit} - GPU freed (no inference until `node server start`)");
            }
        }
        ServerAction::Start => {
            swap::sudo(&["systemctl", "start", &unit])?;
            if json {
                println!("{}", serde_json::json!({"unit": unit, "result": "started"}));
            } else {
                println!("started {unit}");
            }
        }
        ServerAction::Restart => {
            swap::sudo(&["systemctl", "restart", &unit])?;
            if json {
                println!(
                    "{}",
                    serde_json::json!({"unit": unit, "result": "restarted"})
                );
            } else {
                println!("restarted {unit}");
            }
        }
    }
    Ok(())
}

/// `node server --node <remote>`: the same verbs over the node's transport.
fn cmd_node_server_remote(
    node: &crate::config::Node,
    unit: &str,
    action: ServerAction,
    json: bool,
) -> Result<()> {
    let t = transport::for_node(node)?;
    let verb = match action {
        ServerAction::Status => {
            let s = t.status();
            if json {
                println!("{}", serde_json::json!({"unit": unit, "active": s.healthy}));
            } else {
                println!(
                    "{unit} on `{}`: {}",
                    node.name,
                    if s.healthy { "running" } else { "stopped" }
                );
            }
            return Ok(());
        }
        ServerAction::Stop => ("stop", "stopped"),
        ServerAction::Start => ("start", "started"),
        ServerAction::Restart => ("restart", "restarted"),
    };
    let (verb, done) = verb;
    t.server_ctl(verb)?;
    if json {
        println!("{}", serde_json::json!({"unit": unit, "result": done}));
    } else {
        println!("{done} {unit} on `{}`", node.name);
    }
    Ok(())
}

/// `node expose` - bind llama-server to the LAN (0.0.0.0) or localhost, applied
/// immediately by re-staging the served model's drop-in.
pub(crate) fn cmd_node_expose(cfg: &Config, action: ExposeAction, json: bool) -> Result<()> {
    let node = cfg.local_node();
    let port = nodeops::parse_bind(&node.llama_url).1;
    let addr_for = |exposed: bool| {
        if exposed {
            settings::lan_ip()
                .map(|ip| format!("http://{ip}:{port}"))
                .unwrap_or_else(|| format!("http://0.0.0.0:{port}"))
        } else {
            format!("http://127.0.0.1:{port}")
        }
    };

    if let ExposeAction::Status = action {
        let exposed = settings::exposed();
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "exposed": exposed,
                    "bind_host": settings::bind_host(),
                    "address": addr_for(exposed),
                }))?
            );
        } else if exposed {
            println!("exposed to the LAN - reachable at {}", addr_for(true));
        } else {
            println!("localhost only - {}", addr_for(false));
        }
        return Ok(());
    }

    let on = matches!(action, ExposeAction::On);
    let (applied, fw_ok, backend) = nodeops::set_exposure(node, on)?;
    let port_spec = format!("{port}/tcp");
    // Backend-accurate manual fallback (the old hint said `sudo ufw allow`
    // even on firewalld/nftables hosts, where it is wrong advice).
    let hint = crate::netboot_server::expose_hint(backend, port, on);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "exposed": on,
                "applied": applied,
                "firewall_ok": fw_ok,
                "firewall_backend": backend.as_str(),
                "address": addr_for(on),
            }))?
        );
    } else if on {
        println!("[ok]  exposed to the LAN at {}", addr_for(true));
        println!("      UNAUTHENTICATED - anyone on the network can reach it.");
        if !fw_ok {
            println!(
                "      (couldn't open the firewall ({}) - run `{hint}`)",
                backend.as_str()
            );
        }
        if !applied {
            println!("      (no model was served - applies on next `node load`)");
        }
    } else {
        println!("[ok]  localhost only - {}", addr_for(false));
        if !fw_ok {
            println!(
                "      (couldn't close the firewall port {port_spec} ({}) - run `{hint}`)",
                backend.as_str()
            );
        }
        if !applied {
            println!("      (applies on next `node load`)");
        }
    }
    Ok(())
}

/// `endpoint auth on|off|status` - the intuitive auth toggle. Keyless is the
/// default; `on` enables it (generating + applying a key), `off` returns to
/// keyless. Applies immediately by reloading the served model.
pub(crate) fn cmd_node_auth(cfg: &Config, action: AuthAction, json: bool) -> Result<()> {
    let node = cfg.local_node();

    // Status: no state change.
    if let AuthAction::Status = action {
        let key = settings::api_key();
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "auth": key.is_some(), "key": key,
                }))?
            );
        } else if let Some(k) = key {
            println!("auth ON - clients must send `Authorization: Bearer {k}`");
        } else {
            println!("auth OFF - keyless, any client on the network can connect");
        }
        return Ok(());
    }

    let key = match action {
        AuthAction::On => Some(settings::generate_key()),
        AuthAction::Off => None,
        AuthAction::Status => unreachable!(),
    };
    settings::set_api_key(key.as_deref())?;
    let applied = nodeops::restage_served(node);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true, "auth": key.is_some(), "key": key, "applied": applied,
            }))?
        );
    } else if let Some(k) = &key {
        println!("[ok]  auth ON - clients must send `Authorization: Bearer {k}`");
        println!("      point your harness at the key above (see `llmtune endpoint`)");
        if !applied {
            println!("      (no model served - applies on next `node load`)");
        }
    } else {
        println!("[ok]  auth OFF - keyless, load a model and your harness connects");
        if !applied {
            println!("      (applies on next `node load`)");
        }
    }
    Ok(())
}

/// `endpoint identity harness|model|status` - who defines the served model's
/// identity. Some GGUF chat templates embed a branding instruction that
/// overrides the harness's system prompt; `harness` serves a stripped override
/// template so the harness's system prompt fully defines identity, `model`
/// restores the embedded (branded) one. Both apply immediately by reloading
/// the served model. Local node only (the profile + template live here).
pub(crate) fn cmd_node_identity(cfg: &Config, action: IdentityAction, json: bool) -> Result<()> {
    let node = cfg.local_node();

    // Status: no state change - classify the LIVE template.
    if let IdentityAction::Status = action {
        let branded = llama::chat_template(&node.llama_url)
            .map(|t| identity::template_is_branded(&t))
            .unwrap_or(false);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "identity_branded": branded,
                }))?
            );
        } else if branded {
            println!("identity: model-branded (the model's chat template overrides your harness)");
            println!("          fix: `llmtune endpoint identity harness`");
        } else {
            println!("identity: harness-controlled - your system prompt defines identity");
        }
        return Ok(());
    }

    let to_harness = matches!(action, IdentityAction::Harness);
    let (applied, template) = if to_harness {
        let (applied, path) = identity::set_identity_harness(node)?;
        (applied, Some(path.display().to_string()))
    } else {
        (identity::set_identity_model(node)?, None)
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true, "identity_branded": !to_harness,
                "applied": applied, "template": template,
            }))?
        );
    } else if to_harness {
        println!(
            "[ok]  identity harness-controlled - stripped template installed at {}",
            template.as_deref().unwrap_or("?")
        );
        if applied {
            println!("      (server reloaded - the override is live now)");
        } else {
            println!("      (no model served - applies on next `node load`)");
        }
    } else {
        println!("[ok]  identity model-branded - the embedded chat template is back");
        if applied {
            println!("      (server reloaded - the model's own identity is live now)");
        } else {
            println!("      (applies on next `node load`)");
        }
    }
    Ok(())
}

/// `node api-key` - manage the server's required API key.
pub(crate) fn cmd_node_api_key(cfg: &Config, cmd: ApiKeyCmd, json: bool) -> Result<()> {
    let node = cfg.local_node();
    // Show: no state change, no reload.
    if let ApiKeyCmd::Show = cmd {
        let key = settings::api_key();
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "set": key.is_some(), "key": key,
                }))?
            );
        } else {
            match key {
                Some(k) => println!("api key: {k}"),
                None => println!("no api key set (server is unauthenticated)"),
            }
        }
        return Ok(());
    }

    // Set / generate / clear: persist, then re-stage the served model to apply.
    let key = match &cmd {
        ApiKeyCmd::Set { key } => Some(key.clone()),
        ApiKeyCmd::Generate => Some(settings::generate_key()),
        ApiKeyCmd::Clear => None,
        ApiKeyCmd::Show => unreachable!(),
    };
    settings::set_api_key(key.as_deref())?;
    let applied = nodeops::restage_served(node);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true, "set": key.is_some(), "key": key, "applied": applied,
            }))?
        );
    } else if let Some(k) = &key {
        println!("[ok]  api key set - clients must send `Authorization: Bearer {k}`");
        if !applied {
            println!("      (no model served - applies on next `node load`)");
        }
    } else {
        println!("[ok]  api key cleared - server is unauthenticated");
        if !applied {
            println!("      (applies on next `node load`)");
        }
    }
    Ok(())
}

/// `node boot-restore` - re-apply persisted settings after a reboot, or install
/// the systemd hook that does it automatically.
pub(crate) fn cmd_node_boot_restore(cfg: &Config, install: bool, json: bool) -> Result<()> {
    let node = cfg.local_node();
    if install {
        let unit = install_restore_hook()?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({"ok": true, "installed": unit}))?
            );
        } else {
            println!("[ok]  installed + enabled {unit} - settings re-applied on boot");
            println!("      (a snapper rollback can revert the unit; re-run `--install` after)");
        }
        return Ok(());
    }
    let (loaded, fw_ok) = nodeops::boot_restore(node)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true, "loaded": loaded, "firewall_ok": fw_ok,
                "exposed": settings::exposed(), "api_key": settings::api_key().is_some(),
            }))?
        );
    } else {
        println!(
            "[ok]  boot-restore: model {}, exposure {}",
            if loaded {
                "reloaded"
            } else {
                "none remembered"
            },
            if settings::exposed() {
                if fw_ok {
                    "re-applied"
                } else {
                    "set (firewall open failed)"
                }
            } else {
                "localhost"
            }
        );
    }
    Ok(())
}

/// Write + enable the boot-time systemd oneshot. Runs as the invoking user so it
/// resolves the same settings + (durable) binary. Returns the unit name.
pub(crate) fn install_restore_hook() -> Result<String> {
    let bin = std::env::current_exe()?.display().to_string();
    let user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
    let unit_name = "llmtune-restore.service";
    let unit = format!(
        "[Unit]\n\
         Description=llmtune boot-restore - re-apply exposure/api-key + served model\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         Type=oneshot\n\
         User={user}\n\
         ExecStart={bin} node boot-restore\n\
         \n\
         [Install]\n\
         WantedBy=multi-user.target\n"
    );
    // /etc on a normal install; /run when /etc's unit dir is read-only (the
    // NixOS netboot image) - with a --runtime enable, since a persistent
    // enable would try to symlink into the read-only /etc.
    let dir = swap::unit_install_dir();
    let path_s = format!("{dir}/{unit_name}");
    swap::sudo_tee(std::path::Path::new(&path_s), &unit)?;
    swap::sudo(&["systemctl", "daemon-reload"])?;
    if dir == swap::RUN_UNIT_DIR {
        swap::sudo(&["systemctl", "enable", "--runtime", unit_name])?;
    } else {
        swap::sudo(&["systemctl", "enable", unit_name])?;
    }
    Ok(unit_name.to_string())
}

pub(crate) fn cmd_node_unload(cfg: &Config, sel: Option<&str>, json: bool) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    match node.transport {
        crate::config::Transport::Local => swap::clear_dropin(&node.llama_unit)?,
        _ => transport::for_node(node)?.unload()?,
    }
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "node": node.name,
                "unit": node.llama_unit,
            }))?
        );
        return Ok(());
    }
    println!(
        "[ok]  removed llmtune drop-in on `{}` - `{}` reverted to its base config",
        node.name, node.llama_unit
    );
    Ok(())
}

/// `node build-version` - the installed version (commit slug) of a managed
/// build on THIS node. The wire the cluster build-parity check reads over SSH;
/// `version: null` = not installed here.
pub(crate) fn cmd_node_build_version(name: &str, json: bool) -> Result<()> {
    let version = crate::build::current_version(name);
    if json {
        let dto = transport::BuildVersion {
            name: name.to_string(),
            version,
        };
        println!("{}", serde_json::to_string_pretty(&dto)?);
        return Ok(());
    }
    match version {
        Some(v) => println!("{name} {v}"),
        None => println!("{name} not installed"),
    }
    Ok(())
}

pub(crate) fn cmd_doctor_on(cfg: &Config, sel: Option<&str>, json: bool) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    let checks = transport::for_node(node)?.doctor()?;
    if json {
        // Serialize the DTOs directly - the same shape the SSH transport parses.
        println!("{}", serde_json::to_string_pretty(&checks)?);
    } else {
        println!("llmtune doctor - node `{}`\n", node.name);
        for c in &checks {
            println!("  {} {:<16} {}", c.status, c.label, c.detail);
        }
        println!("\nverdict: {}", doctor::worst(&checks));
    }
    Ok(())
}

/// The `node list --json` payload: the `ModelInfo` DTO serialized directly, so
/// the wire shape is defined ONCE (transport.rs) and can't drift from what the
/// SSH transport parses. Pure, so the round-trip is unit-testable.
#[cfg(test)]
pub(crate) fn node_list_json(rows: &[nodeops::ModelRow]) -> Result<String> {
    let arr: Vec<transport::ModelInfo> = rows.iter().map(transport::ModelInfo::from).collect();
    Ok(serde_json::to_string_pretty(&arr)?)
}

pub(crate) fn cmd_node_list(cfg: &Config, sel: Option<&str>, json: bool) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    // The transport's ModelInfo IS the `node list --json` wire shape, local
    // and remote alike (see node_list_json / the round-trip test below).
    let rows = transport::for_node(node)?.list()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    if rows.is_empty() {
        if node.transport == crate::config::Transport::Local {
            // Make sure the directory exists so it's a concrete place to drop files.
            model::ensure_dir(&node.models_dir);
        }
        println!("No models found yet.\n");
        println!("{}", model::where_to_put_models(&node.models_dir));
        return Ok(());
    }
    println!(
        "models on `{}` in {}  (* = served now)\n",
        node.name, node.models_dir
    );
    for r in &rows {
        let mark = if r.served { "*" } else { " " };
        let prof = if r.used_default {
            format!("{} (no profile)", r.profile)
        } else {
            r.profile.clone()
        };
        println!(
            "{mark} {} {:>8} {:>6.1}G {:>9}  {}",
            padw(&r.name, 40),
            r.params.clone().unwrap_or_else(|| "?".into()),
            r.size_gib,
            r.quant.clone().unwrap_or_else(|| "-".into()),
            prof,
        );
    }
    Ok(())
}

pub(crate) fn cmd_node_served(cfg: &Config, sel: Option<&str>, json: bool) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    let (served, benchmarking) = match node.transport {
        crate::config::Transport::Local => (nodeops::served(node), nodeops::bench_marker_present()),
        _ => {
            let s = transport::for_node(node)?.status();
            (s.served, s.benchmarking)
        }
    };
    if json {
        // `benchmarking` is additive; `served` keeps its exact shape.
        println!(
            "{}",
            serde_json::json!({ "served": served, "benchmarking": benchmarking })
        );
    } else {
        match served {
            Some(s) => println!("{s}"),
            None if benchmarking => println!("(benchmarking - server paused)"),
            None => println!("(no model served / server down)"),
        }
    }
    Ok(())
}

pub(crate) fn cmd_node_status(cfg: &Config, sel: Option<&str>, json: bool) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    let s = transport::for_node(node)?.status();
    if json {
        println!("{}", serde_json::to_string(&s)?);
    } else {
        println!("{}", status_line(&s));
    }
    Ok(())
}

pub(crate) fn cmd_node_load(cfg: &Config, sel: Option<&str>, name: &str, json: bool) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    let r = transport::for_node(node)?.load(name)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        // A failed/reverted swap must exit non-zero in BOTH modes (agents
        // check the code, not the prose) - the payload above is the detail.
        if !r.ok {
            std::process::exit(1);
        }
        return Ok(());
    }
    let tag = if r.ok { "[ok]  " } else { "[fail]" };
    println!("{tag} {}", r.detail);
    if r.ok && r.used_default {
        println!("       (no arch profile - used conservative default flags)");
    }
    println!("       elapsed {:.1}s", r.elapsed_secs);
    if !r.ok {
        std::process::exit(1);
    }
    Ok(())
}

pub(crate) fn status_line(s: &transport::NodeStatus) -> String {
    if !s.reachable {
        return format!("{:<14} UNREACHABLE", s.name);
    }
    // A server the bench intentionally stopped is "bench", not "down" - the
    // pause is expected and self-clearing, not a fault.
    let health = if s.healthy {
        "up"
    } else if s.benchmarking {
        "bench"
    } else {
        "down"
    };
    let served = s
        .served
        .as_deref()
        .unwrap_or(if !s.healthy && s.benchmarking {
            "(paused for benchmark)"
        } else {
            "(none)"
        });
    let last = match (&s.last_model, s.last_gen_tok_s) {
        (Some(_), Some(t)) => format!("last {t:.1} t/s"),
        _ => "no bench".to_string(),
    };
    format!(
        "{:<14} {:<5} {} {} models  {}",
        s.name,
        health,
        fit(served, 28),
        s.models,
        last
    )
}

pub(crate) fn cmd_node_bench(
    cfg: &Config,
    sel: Option<&str>,
    target: Option<&str>,
    spec: bench::BenchSpec,
    json: bool,
) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    let t = transport::for_node(node)?;
    // Target guard: when a specific model is requested, load it and confirm it
    // actually took before measuring. A `node load` can fail and REVERT to the
    // previously-served model; without this check the bench would then measure
    // (and mislabel) that fallback. `load` verifies served==target internally
    // and reports ok=false / reverted on a mismatch, so we just require ok here.
    if let Some(q) = target {
        let outcome = t.load(q)?;
        if !outcome.ok || outcome.reverted {
            anyhow::bail!(
                "`{q}` did not load - refusing to benchmark the fallback ({}).",
                outcome.detail
            );
        }
        if !json {
            println!("loaded `{}` - benchmarking…", outcome.to);
        }
    }
    let rec = t.bench(&spec)?;
    let p = &rec.perf;
    if json {
        println!("{}", serde_json::to_string_pretty(&rec)?);
        return Ok(());
    }
    println!(
        "benchmarked `{}` [{}] on `{}`:",
        rec.model, rec.arch, rec.node
    );
    println!(
        "  prompt   {:>8.1} tok/s  ({} tokens)",
        p.prompt_tok_s, p.n_prompt
    );
    println!(
        "  gen      {:>8.1} tok/s  ({} tokens)",
        p.gen_tok_s, p.n_gen
    );
    println!("  ttft     {:>8.1} ms", p.ttft_ms);
    println!("  latency  {:>8.1} ms total", p.total_ms);
    let t = &p.telemetry;
    if t.samples > 0 {
        println!(
            "  gpu      gfx {}-{} MHz (avg {}), uclk {} MHz, peak {:.1} C",
            t.gfxclk_min, t.gfxclk_max, t.gfxclk_avg, t.uclk_mhz, t.temp_peak_c
        );
    }
    println!("  (logged to history)");
    Ok(())
}

pub(crate) fn cmd_node_history(
    cfg: &Config,
    sel: Option<&str>,
    limit: usize,
    json: bool,
) -> Result<()> {
    let node = resolve_target(cfg, sel)?;
    let recs = transport::for_node(node)?.history(limit)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&recs)?);
        return Ok(());
    }
    if recs.is_empty() {
        println!("no benchmark history for node `{}` yet", node.name);
        return Ok(());
    }
    println!(
        "{:<17} {:<26} {:>9} {:>9} {:>8} {:>7} {:>8}  pool",
        "when", "model", "prefill", "gen t/s", "ttft", "ctx", "quant"
    );
    for r in &recs {
        // A pooled (cluster) run is tagged so it can't masquerade as
        // single-node throughput: `big+2w` = cluster `big`, 2 pooled workers.
        let pool = match &r.cluster {
            Some(c) => format!("{c}+{}w", r.cluster_members.len()),
            None => "-".to_string(),
        };
        println!(
            "{:<17} {} {:>9.1} {:>9.1} {:>7.0}ms {:>7} {:>8}  {}",
            fmt_ts(r.ts),
            fit(&r.model, 26),
            r.perf.prompt_tok_s,
            r.perf.gen_tok_s,
            r.perf.ttft_ms,
            r.ctx,
            r.quant.clone().unwrap_or_else(|| "-".into()),
            pool,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_line_reports_bench_pause_distinctly() {
        let mut s = transport::NodeStatus::unreachable("n");
        s.reachable = true;
        // up: unchanged
        s.healthy = true;
        s.served = Some("m.gguf".into());
        let line = status_line(&s);
        assert!(line.contains(" up ") && line.contains("m.gguf"), "{line}");
        // down + marker: the intentional pause, never "down"/"(none)"
        s.healthy = false;
        s.served = None;
        s.benchmarking = true;
        let line = status_line(&s);
        assert!(line.contains("bench"), "{line}");
        assert!(line.contains("(paused for benchmark)"), "{line}");
        assert!(!line.contains("down") && !line.contains("(none)"), "{line}");
        // down without the marker: a real down, as before
        s.benchmarking = false;
        let line = status_line(&s);
        assert!(line.contains("down") && line.contains("(none)"), "{line}");
    }

    #[test]
    fn node_list_json_roundtrips_through_the_transport_dto() {
        // What `node list --json` prints must deserialize into the EXACT
        // Vec<ModelInfo> the SSH transport parses -- one wire contract.
        let row = nodeops::ModelRow {
            model: model::Model {
                path: std::path::PathBuf::from("/models/Qwen3.5-9B-IQ2.gguf"),
                name: "Qwen3.5-9B-IQ2.gguf".into(),
                arch: "qwen35".into(),
                params: Some("9B".into()),
                quant: Some("IQ2".into()),
                size_bytes: 6_979_321_856,
                ctx_max: Some(98304),
                has_mtp: true,
            },
            served: true,
            profile_id: "qwen35".into(),
            used_default: false,
            bin: "/var/lib/llmtune/builds/vulkan/current/llama-server".into(),
            flags: "-c 32768 -ngl 99".into(),
            overridden: true,
            mem: None,
        };
        let json = node_list_json(&[row]).unwrap();
        let parsed: Vec<transport::ModelInfo> =
            serde_json::from_str(&json).expect("node list --json must parse as Vec<ModelInfo>");
        assert_eq!(parsed.len(), 1);
        let m = &parsed[0];
        assert_eq!(m.name, "Qwen3.5-9B-IQ2.gguf");
        assert_eq!(m.arch, "qwen35");
        assert_eq!(m.params.as_deref(), Some("9B"));
        assert_eq!(m.quant.as_deref(), Some("IQ2"));
        assert_eq!(m.ctx_max, Some(98304));
        assert_eq!(m.profile, "qwen35");
        assert!(m.served && m.overridden && m.has_mtp && !m.used_default);
        assert_eq!(m.flags, "-c 32768 -ngl 99");
        assert!((m.size_gib - 6.5).abs() < 0.06, "rounded to one decimal");
    }
}
