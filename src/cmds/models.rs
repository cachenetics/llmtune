// SPDX-License-Identifier: GPL-2.0-only
//! `llmtune models` - manage the control-host model library (the NFS-exported
//! `[netboot] models_dir`). The render layer over `library`; see library.rs
//! for the atomicity + served-guard mechanics.

use anyhow::Result;
use std::path::Path;

use crate::config::Config;
use crate::fmt::padw;
use crate::{library, model};

/// One `models list` row, serialized directly for `--json`. Same facts
/// `node list` shows for a model (name/params/quant/arch/size), minus the
/// node-side state (served/profile/mem) that has no meaning library-side.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct LibraryRow {
    pub name: String,
    pub arch: String,
    #[serde(default)]
    pub params: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    pub size_gib: f64,
    #[serde(default)]
    pub ctx_max: Option<u64>,
}

impl From<&model::Model> for LibraryRow {
    fn from(m: &model::Model) -> Self {
        LibraryRow {
            name: m.name.clone(),
            arch: m.arch.clone(),
            params: m.params.clone(),
            quant: m.quant.clone(),
            size_gib: (m.size_gib() * 10.0).round() / 10.0,
            ctx_max: m.ctx_max,
        }
    }
}

/// The `models list --json` payload. Pure over the discovered models so the
/// shape is unit-testable.
pub(crate) fn models_list_json(dir: &Path, models: &[model::Model]) -> Result<String> {
    let rows: Vec<LibraryRow> = models.iter().map(LibraryRow::from).collect();
    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "dir": dir.to_string_lossy(),
        "models": rows,
    }))?)
}

pub(crate) fn cmd_models_list(cfg: &Config, json: bool) -> Result<()> {
    let dir = library::dir(cfg);
    let models = library::list(&dir)?;
    if json {
        println!("{}", models_list_json(&dir, &models)?);
        return Ok(());
    }
    if models.is_empty() {
        model::ensure_dir(&dir.to_string_lossy());
        println!("Model library is empty.\n");
        println!("{}", model::where_to_put_models(&dir.to_string_lossy()));
        println!("Or: llmtune models add <path-or-url>");
        return Ok(());
    }
    println!(
        "model library {}  (NFS-exported read-only to nodes)\n",
        dir.display()
    );
    for m in &models {
        println!(
            "  {} {:>8} {:>7.1}G {:>9}  {}",
            padw(&m.name, 44),
            m.params.clone().unwrap_or_else(|| "?".into()),
            m.size_gib(),
            m.quant.clone().unwrap_or_else(|| "-".into()),
            m.arch,
        );
    }
    let total: u64 = models.iter().map(|m| m.size_bytes).sum();
    println!(
        "\n{} models, {:.1} GiB",
        models.len(),
        total as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

pub(crate) fn cmd_models_add(cfg: &Config, source: &str, json: bool) -> Result<()> {
    let dir = library::dir(cfg);
    let name = library::dest_name(source)?;
    let out = if library::is_url(source) {
        if !json {
            println!("downloading {source} -> {}", dir.display());
        }
        library::add_url(&dir, source, &name)?
    } else {
        library::add_local(&dir, Path::new(source), &name)?
    };
    if json {
        println!(
            "{}",
            serde_json::json!({
                "added": out.name,
                "path": out.path.to_string_lossy(),
                "size_gib": (out.size_bytes as f64 / (1u64 << 30) as f64 * 10.0).round() / 10.0,
            })
        );
    } else {
        println!(
            "[ok] added {} ({:.1} GiB) to the library",
            out.name,
            out.size_bytes as f64 / (1u64 << 30) as f64
        );
        println!("     nodes see it on next access over the NFS automount - no remount needed");
    }
    Ok(())
}

pub(crate) fn cmd_models_rm(
    cfg: &Config,
    query: &str,
    yes: bool,
    force: bool,
    json: bool,
) -> Result<()> {
    let dir = library::dir(cfg);
    let name = library::resolve_rm(&dir, query)?;
    // Confirm before unlinking a multi-GB file that a SUBSTRING matched -
    // destructive actions follow the agentic contract: interactive y/N on a
    // tty, `--yes`/`--force` as the non-interactive escape hatch, refusal
    // (exit 2) otherwise. `--json` counts as non-interactive: a prompt would
    // corrupt the stream.
    if !yes && !force {
        let size = std::fs::metadata(dir.join(&name))
            .map(|m| format!(" ({:.1} GiB)", m.len() as f64 / (1u64 << 30) as f64))
            .unwrap_or_default();
        let refuse_msg =
            format!("refusing to remove `{name}` without confirmation - pass --yes (or --force)");
        if json {
            return Err(crate::agentic::refusal(refuse_msg));
        }
        if !crate::agentic::confirm_tty(
            &format!("remove `{name}`{size} from the library? [y/N] "),
            &refuse_msg,
        )? {
            println!("[..] aborted - `{name}` untouched");
            return Ok(());
        }
    }
    // The served-guard: ask every fleet node what it serves before unlinking a
    // file they mmap over NFS. --force skips the queries entirely (offline
    // library surgery must not require a reachable fleet).
    let guard = if force {
        library::RmGuard {
            serving: Vec::new(),
            unknown: Vec::new(),
        }
    } else {
        library::rm_guard(&name, &library::served_across_fleet(cfg))
    };
    if let Some(msg) = library::rm_block_message(&name, &guard, force) {
        // The served-guard is a REFUSAL (state untouched, --force overrides):
        // exit 2, and under --json the standard {"error", "refused": true}
        // object on stderr (see src/agentic.rs).
        return Err(crate::agentic::refusal(msg));
    }
    let path = dir.join(&name);
    std::fs::remove_file(&path).map_err(|e| anyhow::anyhow!("removing {}: {e}", path.display()))?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "removed": name,
                "blocked": false,
                "forced": force,
                "unreachable_nodes": guard.unknown,
            })
        );
    } else {
        println!("[ok] removed {name} from the library");
        if !guard.unknown.is_empty() {
            println!(
                "     note: could not verify {} (unreachable) - if one of them was serving \
                 this model its server just lost the file",
                guard.unknown.join(", ")
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fake_model(name: &str, bytes: u64) -> model::Model {
        model::Model {
            path: PathBuf::from(format!("/lib/{name}")),
            name: name.to_string(),
            arch: "gemma3".to_string(),
            params: Some("27B".to_string()),
            quant: Some("Q4_K_M".to_string()),
            size_bytes: bytes,
            ctx_max: Some(131072),
            has_mtp: false,
        }
    }

    #[test]
    fn list_json_shape_round_trips() {
        let models = vec![fake_model("A-27B-Q4_K_M.gguf", 16 << 30)];
        let json = models_list_json(Path::new("/srv/models"), &models).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["dir"], "/srv/models");
        let rows: Vec<LibraryRow> = serde_json::from_value(v["models"].clone()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "A-27B-Q4_K_M.gguf");
        assert_eq!(rows[0].arch, "gemma3");
        assert_eq!(rows[0].params.as_deref(), Some("27B"));
        assert_eq!(rows[0].quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(rows[0].size_gib, 16.0);
        assert_eq!(rows[0].ctx_max, Some(131072));
    }

    #[test]
    fn list_json_empty_library_is_an_empty_array() {
        let json = models_list_json(Path::new("/srv/models"), &[]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["models"].as_array().unwrap().is_empty());
    }
}
