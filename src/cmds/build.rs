// SPDX-License-Identifier: GPL-2.0-only
//! Build-manager command handlers (`llmtune build <op>`).

use crate::config::Config;
use crate::{build, nodeops};
use anyhow::Result;

pub(crate) fn resolve_build_spec(
    name: &str,
    ref_override: Option<String>,
) -> Result<build::BuildSpec> {
    let specs = build::load()?;
    let mut s = build::spec(&specs, name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no build `{name}` - see `llmtune build list`"))?;
    if let Some(r) = ref_override {
        // `--ref` bypasses builds.toml's validate(); reject the same footguns
        // (empty resolves to HEAD silently; a leading '-' is a git option).
        if r.is_empty() {
            anyhow::bail!("--ref may not be empty");
        }
        if r.starts_with('-') {
            anyhow::bail!("--ref may not start with '-'");
        }
        s.git_ref = r;
    }
    Ok(s)
}

pub(crate) fn cmd_build_install(
    name: &str,
    ref_override: Option<String>,
    retain: usize,
    is_update: bool,
    json: bool,
) -> Result<()> {
    let spec = resolve_build_spec(name, ref_override)?;
    let mut builder = build::RealBuilder::new();
    let out = build::install(&spec, &mut builder, retain)?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "name": out.name,
                "commit": out.commit,
                "slug": out.slug,
                "dir": out.dir.display().to_string(),
                "action": format!("{:?}", out.action).to_lowercase(),
                "previous": out.previous,
            })
        );
        return Ok(());
    }
    match out.action {
        build::Action::AlreadyCurrent => {
            let what = if is_update {
                "already up to date"
            } else {
                "already current"
            };
            println!(
                "[ok]  `{}` {} at {} ({})",
                out.name, what, out.slug, out.commit
            );
        }
        build::Action::Switched => {
            println!(
                "[ok]  `{}` activated {} ({}) without rebuild{}",
                out.name,
                out.slug,
                out.commit,
                out.previous
                    .as_deref()
                    .map(|p| format!(" (was {p})"))
                    .unwrap_or_default()
            );
        }
        build::Action::Built => {
            println!(
                "[ok]  `{}` built {} ({}) and made current{}",
                out.name,
                out.slug,
                out.commit,
                out.previous
                    .as_deref()
                    .map(|p| format!(" (was {p}, kept for rollback)"))
                    .unwrap_or_default()
            );
            println!("       {}", out.dir.display());
        }
    }
    Ok(())
}

pub(crate) fn cmd_build_rollback(cfg: &Config, name: &str, json: bool) -> Result<()> {
    let specs = build::load().unwrap_or_default();
    let out = build::rollback(name, &specs)?;
    // Bench-informed: if history has runs on both the old and new build versions,
    // show the throughput delta so a regression-driven rollback is justified.
    let delta = bench_delta(cfg, out.from.as_deref(), &out.to);
    if json {
        println!(
            "{}",
            serde_json::json!({ "name": out.name, "from": out.from, "to": out.to, "perf": delta })
        );
        return Ok(());
    }
    println!(
        "[ok]  `{}` rolled back {}-> {}",
        out.name,
        out.from
            .as_deref()
            .map(|f| format!("{f} "))
            .unwrap_or_default(),
        out.to
    );
    if let Some((from_ts, to_ts)) = delta {
        println!(
            "      best gen t/s: {from_ts:.1} (rolled-from) vs {to_ts:.1} (now) - {}",
            if to_ts >= from_ts {
                format!(
                    "+{:.1}% on the version you rolled back to",
                    (to_ts / from_ts - 1.0) * 100.0
                )
            } else {
                format!(
                    "{:.1}% slower; the newer build was faster",
                    (to_ts / from_ts - 1.0) * 100.0
                )
            }
        );
    }
    Ok(())
}

/// Best gen tok/s recorded under each build version across the local node's
/// history, returned as (from_best, to_best) only when BOTH have runs.
pub(crate) fn bench_delta(cfg: &Config, from: Option<&str>, to: &str) -> Option<(f64, f64)> {
    let from = from?;
    let recs = nodeops::history(cfg.local_node(), 1000);
    let best = |slug: &str| {
        recs.iter()
            .filter(|r| r.build.as_deref() == Some(slug))
            .map(|r| r.perf.gen_tok_s)
            .fold(f64::MIN, f64::max)
    };
    let (f, t) = (best(from), best(to));
    (f > f64::MIN && t > f64::MIN && f > 0.0).then_some((f, t))
}

pub(crate) fn cmd_build_list(json: bool) -> Result<()> {
    let specs = build::load()?;
    let statuses = build::list(&specs);
    if json {
        let arr: Vec<_> = statuses
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "configured": s.spec.is_some(),
                    "ref": s.spec.as_ref().map(|sp| sp.git_ref.clone()),
                    "current": s.current,
                    "versions": s.versions.iter().map(|v| serde_json::json!({
                        "slug": v.slug, "current": v.current,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }
    if statuses.is_empty() {
        println!("no builds configured (see builds.toml)");
        return Ok(());
    }
    for s in &statuses {
        let tracked = s
            .spec
            .as_ref()
            .map(|sp| format!("  tracks {} @ {}", sp.git_url, sp.git_ref))
            .unwrap_or_else(|| "  (installed, not in builds.toml)".to_string());
        println!("{}{}", s.name, tracked);
        if s.versions.is_empty() {
            println!("    (not installed - `llmtune build install {}`)", s.name);
        }
        for v in &s.versions {
            let mark = if v.current { "* " } else { "  " };
            println!("  {mark}{}", v.slug);
        }
    }
    Ok(())
}
