// SPDX-License-Identifier: GPL-2.0-only
//! Profile command handlers (`llmtune profile <op>`, `node profile <op>`).

use crate::config::Config;
use crate::{nodeops, profile, swap};
use anyhow::Result;

/// `profile set-model` - set or clear a per-model flag override (the TUI flag
/// editor's operation), so agents can do what the `e` key does.
pub(crate) fn cmd_profile_set_model(
    cfg: &Config,
    model: &str,
    flags: Option<String>,
    reset: bool,
    json: bool,
) -> Result<()> {
    let node = cfg.local_node();
    let m = nodeops::resolve_model(node, model)?;
    if reset {
        profile::clear_override(&m.name)?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"ok": true, "model": m.name, "cleared": true})
                )?
            );
        } else {
            println!("[ok]  cleared per-model override for `{}`", m.name);
        }
        return Ok(());
    }
    let f =
        flags.ok_or_else(|| anyhow::anyhow!("pass --flags \"…\" to set, or --reset to clear"))?;
    profile::set_override(&m.name, &f)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"ok": true, "model": m.name, "flags": f})
            )?
        );
    } else {
        println!("[ok]  set per-model override for `{}`", m.name);
    }
    Ok(())
}

pub(crate) fn cmd_profile_list(json: bool) -> Result<()> {
    let profiles = profile::load()?;
    if json {
        let arr: Vec<_> = profiles
            .iter()
            .map(|p| {
                let (bin, ld) = p.launch();
                serde_json::json!({
                    "id": p.id,
                    "arch_match": p.arch_match,
                    "build": p.build,
                    "bin": bin,
                    "ld": ld,
                    "flags": p.flags,
                    "installed": std::path::Path::new(&bin).is_absolute(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }
    let src = if profile::user_file_exists() {
        "user overrides"
    } else {
        "bundled seed"
    };
    println!("profiles ({src}):\n");
    for p in &profiles {
        let m = if p.arch_match.is_empty() {
            "(fallback)".to_string()
        } else {
            p.arch_match.join(", ")
        };
        println!("{:<12} matches: {m}", p.id);
        if let Some(b) = p.build.as_deref().filter(|b| !b.is_empty()) {
            let (bin, _) = p.launch();
            let state = if std::path::Path::new(&bin).is_absolute() {
                bin
            } else {
                format!("{} (not installed - `llmtune build install {b}`)", p.bin)
            };
            println!("   build: {b} -> {state}");
        } else {
            println!("   bin:   {}", p.bin);
            if let Some(ld) = &p.ld_path {
                println!("   ld:    {ld}");
            }
        }
        println!("   flags: {}\n", p.flags);
    }
    Ok(())
}

pub(crate) fn cmd_profile_show(cfg: &Config, model: &str, json: bool) -> Result<()> {
    let node = cfg.local_node();
    let m = nodeops::resolve_model(node, model)?;
    let profiles = profile::load()?;
    let (p, used_default) = profile::resolve(&profiles, &m.arch, m.quant.as_deref());
    let effective = swap::adjust_flags(p, &m);
    if json {
        let (bin, ld) = p.launch();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": m.name,
                "arch": m.arch,
                "profile": p.id,
                "used_default": used_default,
                "build": p.build,
                "bin": bin,
                "ld": ld,
                "flags": effective,
                "adjusted": effective != p.flags,
            }))?
        );
        return Ok(());
    }
    println!(
        "model   {} [{}]{}",
        m.name,
        m.arch,
        if used_default {
            "  (no arch profile)"
        } else {
            ""
        }
    );
    println!("profile {}", p.id);
    let (bin, ld) = p.launch();
    match p.build.as_deref().filter(|b| !b.is_empty()) {
        Some(b) => println!("build   {b} -> {bin}"),
        None => println!("bin     {bin}"),
    }
    if let Some(ld) = ld {
        println!("ld      {ld}");
    }
    if effective != p.flags {
        println!("flags   {effective}");
        println!("        (memory-guard adjusted from profile default)");
    } else {
        println!("flags   {effective}");
    }
    Ok(())
}

pub(crate) fn cmd_profile_set(
    id: &str,
    flags: Option<String>,
    bin: Option<String>,
    ld: Option<String>,
    json: bool,
) -> Result<()> {
    if flags.is_none() && bin.is_none() && ld.is_none() {
        anyhow::bail!("nothing to set - pass --flags, --bin and/or --ld");
    }
    let mut profiles = profile::load()?;
    let p = profiles
        .iter_mut()
        .find(|p| p.id == id)
        .ok_or_else(|| anyhow::anyhow!("no profile `{id}` - see `llmtune profile list`"))?;
    if let Some(f) = flags {
        p.flags = f;
    }
    if let Some(b) = bin {
        // An absolute/relative path is a literal binary, not a managed build -
        // clear the build reference so launch() uses the path directly.
        if b.contains('/') {
            p.build = None;
        }
        p.bin = b;
    }
    if let Some(l) = ld {
        p.ld_path = Some(l);
    }
    profile::save_user(&profiles)?;
    let where_ = profile::user_path()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "id": id,
                "path": where_,
            }))?
        );
        return Ok(());
    }
    println!("[ok]  profile `{id}` updated -> {where_}");
    Ok(())
}
