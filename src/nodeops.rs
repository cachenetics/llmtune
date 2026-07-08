// SPDX-License-Identifier: GPL-2.0-only
//! Node-local operations. In M0-M3 these run directly against the local node;
//! M4 lifts them behind the `NodeTransport` trait so the same ops run over SSH.

use crate::config::Node;
use crate::profile::Profile;
use crate::{bench, history, llama, lock, model, profile, swap};
use anyhow::{anyhow, bail, Result};
use std::path::Path;

/// A model plus the live facts the `list` view needs.
pub struct ModelRow {
    pub model: model::Model,
    pub served: bool,
    pub profile_id: String,
    pub used_default: bool,
    /// The llama-server binary this model's profile would launch.
    pub bin: String,
    /// The effective flags (a per-model override if set, else profile + guard).
    pub flags: String,
    /// True if `flags` comes from a per-model override (not the profile).
    pub overridden: bool,
    /// Memory-fit estimate at the serving ctx/KV-quant vs the node's UMA budget.
    /// Computed node-side so it crosses the SSH transport intact.
    pub mem: Option<crate::mem::MemEstimate>,
}

/// Discover models on the node, resolve each one's profile, and mark the served
/// one (via the live `/props` endpoint).
pub fn list(node: &Node) -> Result<Vec<ModelRow>> {
    let profiles: Vec<Profile> = profile::load()?;
    let overrides = profile::load_overrides();
    let served = llama::served_name(&node.llama_url);
    let models = model::discover(Path::new(&node.models_dir))?;
    // One sysfs read for the whole list; cloned into each row's estimate.
    let budget = crate::mem::read_uma();
    Ok(models
        .into_iter()
        .map(|m| {
            let (p, used_default) = profile::resolve(&profiles, &m.arch);
            let served = served.as_deref() == Some(m.name.as_str());
            // A per-model override wins over the profile's (guarded) flags.
            let (flags, overridden) = match overrides.get(&m.name) {
                Some(f) => (f.clone(), true),
                None => (swap::adjust_flags(p, &m), false),
            };
            // Estimate memory at the ctx/KV-quant the model will actually serve at.
            let (ctx_flag, kv_flag) = crate::mem::ctx_kv_from_flags(&flags);
            let ctx = ctx_flag
                .or(m.ctx_max)
                .unwrap_or(32768)
                .min(m.ctx_max.unwrap_or(u64::MAX));
            let kv = kv_flag.unwrap_or_else(|| "f16".to_string());
            let dims = model::read_dims(&m.path);
            let mem = Some(crate::mem::estimate(
                m.size_bytes,
                dims,
                ctx,
                &kv,
                budget.clone(),
            ));
            ModelRow {
                profile_id: p.id.clone(),
                used_default,
                served,
                bin: p.launch().0,
                flags,
                overridden,
                mem,
                model: m,
            }
        })
        .collect())
}

/// Basename of the currently-served model, if any.
pub fn served(node: &Node) -> Option<String> {
    llama::served_name(&node.llama_url)
}

/// Resolve a substring query to a single model on the node. Errors on no match
/// or an ambiguous match.
pub fn resolve_model(node: &Node, query: &str) -> Result<model::Model> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        bail!("provide a model name or substring to match (got a blank query)");
    }
    let models = model::discover(Path::new(&node.models_dir))?;
    let matches: Vec<model::Model> = models
        .into_iter()
        .filter(|m| m.name.to_lowercase().contains(&q))
        .collect();
    match matches.len() {
        0 => bail!("no model matches `{query}` in {}", node.models_dir),
        1 => Ok(matches.into_iter().next().unwrap()),
        _ => {
            let names: Vec<String> = matches.iter().take(8).map(|m| m.name.clone()).collect();
            bail!("ambiguous `{query}` - matches: {}", names.join(", "))
        }
    }
}

/// Parse `host:port` for the llama bind out of a `http://host:port` URL.
pub(crate) fn parse_bind(url: &str) -> (String, u16) {
    let s = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    let s = s.split('/').next().unwrap_or(s);
    if let Some((h, p)) = s.rsplit_once(':') {
        if let Ok(port) = p.parse::<u16>() {
            return (h.to_string(), port);
        }
    }
    ("127.0.0.1".to_string(), 8080)
}

/// Hot-swap the node's served model to the one matching `query`, with
/// health-checked auto-revert. Returns the outcome and whether the model fell
/// back to default (no-profile) flags.
pub fn load(node: &Node, query: &str) -> Result<(swap::SwapOutcome, bool)> {
    let _guard = lock::LockGuard::gpu()
        .map_err(|_| anyhow::anyhow!("the GPU is busy (a swap or benchmark is already running on this node) - refusing to swap"))?;

    let m = resolve_model(node, query)?;
    if !model::is_chat_model(&m.name) {
        bail!(
            "`{}` looks like an embedding/rerank model - refusing to serve it",
            m.name
        );
    }
    let profiles: Vec<Profile> = profile::load()?;
    let (prof, used_default) = profile::resolve(&profiles, &m.arch);
    // A per-model override wins over the profile's memory-guarded flags.
    let mut flags = profile::load_overrides()
        .get(&m.name)
        .cloned()
        .unwrap_or_else(|| swap::adjust_flags(prof, &m));
    // Require an API key on the server if one is configured (so an exposed
    // server isn't wide open). llmtune's own probes send it via llama::auth.
    // Pass it via a 0600 --api-key-file (NOT --api-key <k>, which would sit in
    // the drop-in's ExecStart -> /proc/<pid>/cmdline + `systemctl cat`).
    if let Some(k) = crate::settings::api_key() {
        let key_path = swap::write_api_key_file(&node.llama_unit, &k)?;
        flags = format!("{flags} --api-key-file {key_path}");
    }

    // Bind host from the exposure setting (localhost by default; 0.0.0.0 exposes
    // to the LAN). Port stays whatever the node's URL declares. Health/served
    // probes keep using llama_url (127.0.0.1), which 0.0.0.0 still covers.
    let (_, port) = parse_bind(&node.llama_url);
    let opts = swap::SwapOpts {
        host: crate::settings::bind_host(),
        port,
        // Carry the base unit's own env (the netboot image's declarative
        // gfx1013 Vulkan stack) through the drop-in's Environment= reset.
        base_env: swap::unit_base_env(&node.llama_unit),
        ..Default::default()
    };
    let mut act = swap::SystemdActuator::new();
    let health = swap::HttpHealth {
        url: node.llama_url.clone(),
    };
    let outcome = swap::swap(&node.llama_unit, &m, prof, &flags, &opts, &mut act, &health)?;
    // Remember what we deliberately served, so boot-restore can bring it back
    // after a reboot reverts the drop-in.
    if outcome.ok {
        crate::settings::set_served_model(&m.name);
        // A single-node load replaces the cluster head's `--rpc` drop-in, so
        // the pool is no longer serving: clear the active-cluster marker or a
        // later bench would be mis-tagged as pooled. (A reverted swap restores
        // the cluster drop-in, so the marker is kept in that case.)
        if crate::cluster::active().is_some() {
            crate::cluster::clear_active();
        }
    }
    Ok((outcome, used_default))
}

/// Look up the arch/quant/profile of the currently-served model by matching its
/// name against discovered files. Returns ("?", None, "?") if not found.
fn served_meta(node: &Node, model_name: &str) -> (String, Option<String>, String) {
    let models = model::discover(Path::new(&node.models_dir)).unwrap_or_default();
    let Some(m) = models.into_iter().find(|m| m.name == model_name) else {
        return ("?".to_string(), None, "?".to_string());
    };
    let profiles = profile::load().unwrap_or_default();
    let profile_id = if profiles.is_empty() {
        "?".to_string()
    } else {
        profile::resolve(&profiles, &m.arch).0.id.clone()
    };
    (m.arch, m.quant, profile_id)
}

/// Path of the paused-for-benchmark marker. `bench` stops llama-server on
/// purpose for the run; while this marker exists, a down server is NOT a
/// fault - status surfaces report "benchmarking (paused)" instead of "down".
/// Lives under /run (tmpfs) so a hard crash mid-bench can never leave it
/// past a reboot; the normal lifetime is owned by [`BenchPauseMarker`].
/// Honors `LLMTUNE_STATE_DIR` like the lock dir, so isolated runs isolate
/// their marker too.
pub(crate) fn bench_marker_path() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LLMTUNE_STATE_DIR") {
        if !d.is_empty() {
            return std::path::PathBuf::from(d).join("benchmarking");
        }
    }
    std::path::PathBuf::from("/run/llmtune/benchmarking")
}

/// True while a benchmark has intentionally paused llama-server on this node.
pub(crate) fn bench_marker_present() -> bool {
    bench_marker_path().exists()
}

/// RAII guard for the paused-for-benchmark marker: written on construction
/// (best-effort - a marker failure must never fail the bench), removed on
/// `Drop`, so it cannot outlive the bench on success, error, or panic - the
/// same always-runs guarantee as the server restart in [`bench_streaming`].
struct BenchPauseMarker {
    path: std::path::PathBuf,
}

impl BenchPauseMarker {
    fn set(model: &str) -> Self {
        Self::set_at(bench_marker_path(), model)
    }

    fn set_at(path: std::path::PathBuf, model: &str) -> Self {
        let content = format!("model={model}\nstart_unix={}\n", history::now_unix());
        // Direct write when the dir is ours (root on a netboot image, or a
        // state-dir override); otherwise the same privileged path the
        // systemctl calls use (euid==0 runs direct, else sudo - swap::sudo).
        let direct = path.parent().is_some_and(|d| {
            std::fs::create_dir_all(d).is_ok() && std::fs::write(&path, &content).is_ok()
        });
        if !direct {
            if let Some(d) = path.parent() {
                let _ = swap::sudo(&["mkdir", "-p", &d.to_string_lossy()]);
            }
            let _ = swap::sudo_tee(&path, &content);
        }
        BenchPauseMarker { path }
    }
}

impl Drop for BenchPauseMarker {
    fn drop(&mut self) {
        if std::fs::remove_file(&self.path).is_ok() || !self.path.exists() {
            return;
        }
        let _ = swap::sudo(&["rm", "-f", &self.path.to_string_lossy()]);
    }
}

/// Benchmark the node's served model and append the result to its history.
pub fn bench(node: &Node, spec: &bench::BenchSpec) -> Result<history::Record> {
    let (tx, _rx) = std::sync::mpsc::channel();
    bench_streaming(node, spec, tx)
}

/// Like [`bench`], but reports start/finish progress on `tx` (for a live TUI).
///
/// Benchmarks the *served* model with llama.cpp's `llama-bench` - the ground
/// truth, unlike the old server-`/completion` path whose timings were corrupted
/// by speculative decode. Because every BC-250 model fills the 16 GiB UMA,
/// `llama-bench` can't share the GPU with the live server, so this STOPS the
/// server for the duration and restarts it afterward (always - even on error).
pub fn bench_streaming(
    node: &Node,
    spec: &bench::BenchSpec,
    tx: std::sync::mpsc::Sender<bench::BenchProgress>,
) -> Result<history::Record> {
    let _guard = lock::LockGuard::gpu()
        .map_err(|_| anyhow::anyhow!("the GPU is busy (a swap or benchmark is already running on this node) - wait before benchmarking"))?;

    // Identify the served model (server must be up to know what's loaded).
    let props = llama::props(&node.llama_url).ok_or_else(|| {
        anyhow!(
            "llama-server not responding at {} - nothing to benchmark (load a model first)",
            node.llama_url
        )
    })?;
    let model_name = llama::name_from_props(&props).ok_or_else(|| {
        anyhow!(
            "could not determine the served model from {}",
            node.llama_url
        )
    })?;
    let ctx = llama::ctx_from_props(&props).unwrap_or(0);
    let (arch, quant, profile_id) = served_meta(node, &model_name);

    // Resolve the model file + the profile that serves this arch.
    let m = model::discover(Path::new(&node.models_dir))?
        .into_iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| {
            anyhow!(
                "served model `{model_name}` not found in {}",
                node.models_dir
            )
        })?;
    let profiles = profile::load()?;
    let (prof, _) = profile::resolve(&profiles, &m.arch);

    // Resolve llama-bench as a sibling of the profile's llama-server binary. A
    // bare (non-path) server bin means the managed build isn't installed.
    let (server_bin, ld) = prof.launch();
    if !server_bin.contains('/') {
        bail!(
            "profile `{}` can't resolve its build (`{}`) - install it (`llmtune build install {}`) before benchmarking",
            prof.id,
            prof.build.as_deref().unwrap_or("?"),
            prof.build.as_deref().unwrap_or("?")
        );
    }
    let bench_bin = Path::new(&server_bin).with_file_name("llama-bench");

    // Tag the run with the llama.cpp build serving this arch (if a managed build).
    let build = prof
        .build
        .clone()
        .filter(|b| !b.is_empty())
        .and_then(|bname| crate::build::current_version(&bname));

    // Mirror the serving flags (a per-model override wins over the profile's).
    let serving_flags = profile::load_overrides()
        .get(&model_name)
        .cloned()
        .unwrap_or_else(|| swap::adjust_flags(prof, &m));
    let mut extra = bench::bench_flags_from(&serving_flags);

    // Pooled honesty (SPEC 5.7): while a cluster is active the bench must pool
    // over the SAME workers (`llama-bench --rpc`), or a head-only measurement
    // would be recorded as cluster throughput (and a pool-only model would fail
    // to load at all). Resolved BEFORE the record is tagged below.
    let active = crate::cluster::active();
    if let Some(ac) = &active {
        if ac.rpc_endpoints.is_empty() {
            bail!(
                "cluster `{}` is active but its marker records no RPC endpoints \
                 (written by an older llmtune) - re-run `llmtune cluster up {}` \
                 or `llmtune cluster down` before benchmarking",
                ac.name,
                ac.name
            );
        }
        extra.push("--rpc".to_string());
        extra.push(ac.rpc_endpoints.join(","));
    }

    let cfg = bench::LlamaBench {
        bin: bench_bin,
        ld_path: ld,
        env: prof.env.clone(),
        model: m.path.clone(),
        extra,
    };

    // Stop the server to give llama-bench the whole GPU/UMA, run, then ALWAYS
    // bring the server back (reset-failed clears any latched crash-loop guard so
    // the restart can't be refused). The server reloads the same model it served.
    // Mark the pause BEFORE stopping the server, so any status probe during
    // the bench window reads "benchmarking (paused)" instead of a false
    // "down" (`node status`, fleet/netboot TUI). The guard's Drop removes the
    // marker on every exit path - success, error, or panic - mirroring the
    // always-restart guarantee below; it is also dropped explicitly once the
    // server is confirmed back up.
    let bench_marker = BenchPauseMarker::set(&model_name);
    let unit = &node.llama_unit;
    swap::sudo(&["systemctl", "stop", unit])?;
    let perf_res = bench::run_llama_bench(&cfg, spec, &tx);
    let _ = swap::sudo(&["systemctl", "reset-failed", unit]);
    let restart = swap::sudo(&["systemctl", "start", unit]);
    // Best-effort: wait for the server to serve again so we don't leave it down.
    for _ in 0..24 {
        if llama::health_ok(&node.llama_url) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    drop(bench_marker); // server is back (or as back as it gets) - clear the pause
    let perf = perf_res?; // surface a bench failure only after the server is back up
    restart?;

    // Tag a pooled run with the active cluster (SPEC 5.7), so history and the
    // leaderboards can distinguish it from single-node throughput. `active` was
    // resolved above, where the matching `--rpc` endpoints were injected - the
    // tag and the measurement can't disagree.
    let rec = history::Record {
        ts: history::now_unix(),
        node: node.name.clone(),
        model: model_name,
        arch,
        quant,
        ctx,
        profile: profile_id,
        perf,
        notes: String::new(),
        build,
        cluster: active.as_ref().map(|a| a.name.clone()),
        cluster_members: active.map(|a| a.members).unwrap_or_default(),
    };
    history::Store::for_node(&node.name).append(rec.clone())?;
    Ok(rec)
}

/// This node's benchmark history, newest first, truncated to `limit`.
pub fn history(node: &Node, limit: usize) -> Vec<history::Record> {
    let mut v = history::Store::for_node(&node.name).load();
    v.truncate(limit);
    v
}

/// Remove llmtune's model-select drop-in - the unit reverts to its base config
/// on the next restart. (Mirror of the CLI `node unload`, for the transport.)
pub fn unload(node: &Node) -> Result<()> {
    swap::clear_dropin(&node.llama_unit)?;
    // Dropping the drop-in drops `--rpc` with it: the head no longer serves the
    // pool, so the active-cluster marker must not survive to mis-tag benches.
    crate::cluster::clear_active();
    Ok(())
}

/// Control the node's llama-server unit (stop/start/restart) - the CLI
/// `node server` operation, for the transport. Stopping frees the GPU.
pub fn server_ctl(node: &Node, action: &str) -> Result<()> {
    match action {
        "stop" | "start" | "restart" => swap::sudo(&["systemctl", action, &node.llama_unit]),
        other => bail!("unknown server action `{other}` (stop|start|restart)"),
    }
}

/// Set + apply network exposure: persist the bind preference (0.0.0.0 vs
/// 127.0.0.1), open/close the firewall port, and re-stage the served model's
/// drop-in so it takes effect now. Returns `(applied, firewall_ok, backend)`.
/// Shared by the CLI `node expose` and the TUI toggle. Binding alone isn't
/// enough on a box with a default-drop firewall, hence the firewall step
/// (best-effort). The step handles ufw AND firewalld AND nftables via
/// `netboot_server::expose_port` - the old ufw-only path silently left the
/// port closed on firewalld/nftables hosts (and, worse for `off`, could leave
/// it OPEN while the operator believed it closed).
pub fn set_exposure(node: &Node, on: bool) -> Result<(bool, bool, crate::netboot_server::Backend)> {
    crate::settings::set_exposed(on)?;
    let port = parse_bind(&node.llama_url).1;
    let (fw_ok, backend) = crate::netboot_server::expose_port(port, on);
    Ok((restage_served(node), fw_ok, backend))
}

/// Boot-restore: re-apply the persisted settings after a reboot reverted the
/// (root-subvol) systemd drop-in + firewall rule. Loads the remembered model
/// (which bakes the bind host + api key from settings back into the drop-in) and
/// re-opens the firewall if exposed. Returns `(loaded, firewall_ok)`.
pub fn boot_restore(node: &Node) -> Result<(bool, bool)> {
    // A reboot killed any pooled workers and reverted the head's `--rpc`
    // drop-in; this path reloads SINGLE-node, so a surviving active-cluster
    // marker would mis-tag every post-reboot bench as pooled. Clear it first.
    crate::cluster::clear_active();
    let loaded = match crate::settings::served_model() {
        Some(m) => {
            // Force a re-stage even if the base unit auto-started the same model.
            let _ = swap::sudo(&["systemctl", "stop", &node.llama_unit]);
            load(node, &m).is_ok()
        }
        None => false,
    };
    let fw_ok = if crate::settings::exposed() {
        // Multi-backend (ufw/firewalld/nftables), like set_exposure.
        crate::netboot_server::expose_port(parse_bind(&node.llama_url).1, true).0
    } else {
        true
    };
    Ok((loaded, fw_ok))
}

/// Re-stage the served model's drop-in so a settings change (bind host / API key)
/// takes effect now. A same-model reload short-circuits ("already served"), so
/// stop the unit first, then load to regenerate the drop-in. Returns whether a
/// model was actually re-staged.
///
/// The model name comes from the persisted `served_model`, falling back to a
/// live probe. It must NOT rely on the live probe alone: changing the very
/// setting we're applying (the API key) breaks llmtune's own authenticated
/// probe against the STILL-running old server, so `served()` returns None and
/// the relaunch is wrongly skipped - the "cleared the key but the server kept
/// enforcing it" trap. The persisted model is the authoritative "what should be
/// running" (it's also what boot-restore uses).
pub fn restage_served(node: &Node) -> bool {
    match crate::settings::served_model().or_else(|| served(node)) {
        Some(name) => {
            let _ = swap::sudo(&["systemctl", "stop", &node.llama_unit]);
            load(node, &name).is_ok()
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_node() -> Node {
        Node {
            name: "x".into(),
            host: None,
            transport: crate::config::Transport::Local,
            ssh_user: None,
            ssh_key: None,
            models_dir: "/nonexistent".into(),
            llama_unit: "u".into(),
            llama_url: "http://127.0.0.1:1".into(),
            power_cmd: None,
        }
    }

    #[test]
    fn resolve_model_rejects_blank_query() {
        assert!(resolve_model(&local_node(), "   ").is_err());
        assert!(resolve_model(&local_node(), "").is_err());
    }

    #[test]
    fn bench_pause_marker_present_while_benching_gone_after() {
        let dir = std::env::temp_dir().join(format!("llmtune-marker-a-{}", std::process::id()));
        let path = dir.join("benchmarking");
        {
            let _m = BenchPauseMarker::set_at(path.clone(), "m.gguf");
            assert!(path.exists(), "marker must exist while the guard lives");
            let body = std::fs::read_to_string(&path).unwrap();
            assert!(body.contains("model=m.gguf"), "{body}");
            assert!(body.contains("start_unix="), "{body}");
        }
        assert!(!path.exists(), "drop must clear the marker");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bench_pause_marker_cleared_on_error_and_panic_paths() {
        let dir = std::env::temp_dir().join(format!("llmtune-marker-b-{}", std::process::id()));
        let path = dir.join("benchmarking");
        // Error path: the guard drops on the early return, like a failed
        // systemctl/bench mid-flow.
        fn failing_bench(path: std::path::PathBuf) -> Result<()> {
            let _m = BenchPauseMarker::set_at(path, "m.gguf");
            bail!("bench blew up")
        }
        assert!(failing_bench(path.clone()).is_err());
        assert!(!path.exists(), "marker must not outlive an error return");
        // Panic path: the guard drops during unwind.
        let p2 = path.clone();
        let r = std::panic::catch_unwind(move || {
            let _m = BenchPauseMarker::set_at(p2, "m.gguf");
            panic!("bench panicked");
        });
        assert!(r.is_err());
        assert!(!path.exists(), "marker must not outlive a panic");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_bind_basic() {
        assert_eq!(
            parse_bind("http://127.0.0.1:8080"),
            ("127.0.0.1".into(), 8080)
        );
        assert_eq!(
            parse_bind("http://192.0.2.10:9001/"),
            ("192.0.2.10".into(), 9001)
        );
        assert_eq!(parse_bind("garbage"), ("127.0.0.1".into(), 8080));
    }
}
