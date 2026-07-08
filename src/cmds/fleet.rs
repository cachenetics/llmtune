// SPDX-License-Identifier: GPL-2.0-only
//! Fleet-wide command handlers (`llmtune fleet <op>`).

use crate::cmds::node::status_line;
use crate::config::Config;
use crate::fmt::fit;
use crate::{bench, transport};
use anyhow::Result;

pub(crate) fn cmd_fleet_status(cfg: &Config, json: bool) -> Result<()> {
    let mut all = Vec::new();
    for n in &cfg.nodes {
        match transport::for_node(n) {
            Ok(t) => all.push(t.status()),
            Err(_) => all.push(transport::NodeStatus {
                name: n.name.clone(),
                reachable: false,
                healthy: false,
                benchmarking: false,
                served: None,
                models: 0,
                last_model: None,
                last_gen_tok_s: None,
            }),
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&all)?);
        return Ok(());
    }
    for s in &all {
        println!("{}", status_line(s));
    }
    Ok(())
}

pub(crate) fn cmd_fleet_bench_all(cfg: &Config, spec: bench::BenchSpec, json: bool) -> Result<()> {
    let mut board: Vec<(String, String, f64)> = Vec::new();
    let mut results: Vec<serde_json::Value> = Vec::new();
    for n in &cfg.nodes {
        let t = match transport::for_node(n) {
            Ok(t) => t,
            Err(e) => {
                if json {
                    results.push(serde_json::json!({ "node": n.name, "error": e.to_string() }));
                } else {
                    println!("{:<14} skipped: {e}", n.name);
                }
                continue;
            }
        };
        match t.bench(&spec) {
            Ok(rec) => {
                if json {
                    results.push(serde_json::json!({
                        "node": n.name,
                        "model": rec.model,
                        "gen_tok_s": rec.perf.gen_tok_s,
                        "prompt_tok_s": rec.perf.prompt_tok_s,
                        "ttft_ms": rec.perf.ttft_ms,
                    }));
                } else {
                    println!(
                        "{:<14} {} {:>7.1} gen t/s  ttft {:.0} ms",
                        n.name,
                        fit(&rec.model, 26),
                        rec.perf.gen_tok_s,
                        rec.perf.ttft_ms
                    );
                }
                board.push((n.name.clone(), rec.model, rec.perf.gen_tok_s));
            }
            Err(e) => {
                if json {
                    results.push(serde_json::json!({ "node": n.name, "error": e.to_string() }));
                } else {
                    println!("{:<14} bench failed: {e}", n.name);
                }
            }
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }
    board.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    if !board.is_empty() {
        println!("\nleaderboard (gen tok/s):");
        for (i, (node, model, t)) in board.iter().enumerate() {
            println!("  {}. {:<14} {:<26} {:>7.1}", i + 1, node, model, t);
        }
    }
    Ok(())
}

pub(crate) fn cmd_fleet_swap_all(cfg: &Config, name: &str, json: bool) -> Result<()> {
    let mut results: Vec<serde_json::Value> = Vec::new();
    for n in &cfg.nodes {
        let t = match transport::for_node(n) {
            Ok(t) => t,
            Err(e) => {
                if json {
                    results.push(serde_json::json!({
                        "node": n.name, "ok": false, "error": e.to_string(),
                    }));
                } else {
                    println!("{:<14} skipped: {e}", n.name);
                }
                continue;
            }
        };
        match t.load(name) {
            Ok(r) => {
                if json {
                    results.push(serde_json::json!({
                        "node": n.name, "ok": r.ok, "detail": r.detail,
                    }));
                } else {
                    let tag = if r.ok { "[ok]  " } else { "[fail]" };
                    println!("{:<14} {tag} {}", n.name, r.detail);
                }
            }
            Err(e) => {
                if json {
                    results.push(serde_json::json!({
                        "node": n.name, "ok": false, "error": e.to_string(),
                    }));
                } else {
                    println!("{:<14} [fail] {e}", n.name);
                }
            }
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    }
    Ok(())
}

pub(crate) fn cmd_fleet_leaderboard(cfg: &Config, json: bool) -> Result<()> {
    // Best gen tok/s seen per (node, model, cluster) across history. The
    // cluster tag is part of the KEY so a pooled run competes as its own row,
    // never as (and never against) single-node throughput - honest in
    // history/leaderboard (SPEC 5.7).
    type BoardKey = (String, String, Option<String>);
    let mut best: std::collections::BTreeMap<BoardKey, f64> = std::collections::BTreeMap::new();
    for n in &cfg.nodes {
        let Ok(t) = transport::for_node(n) else {
            continue;
        };
        let recs = t.history(200).unwrap_or_default();
        for r in recs {
            // Skip fantasy records (old /completion path) so one bad row can't
            // win a model's leaderboard slot - see PerfBench::is_plausible.
            if !r.perf.is_plausible() {
                continue;
            }
            let k = (n.name.clone(), r.model.clone(), r.cluster.clone());
            let e = best.entry(k).or_insert(0.0);
            *e = e.max(r.perf.gen_tok_s);
        }
    }
    let mut rows: Vec<(BoardKey, f64)> = best.into_iter().collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if json {
        let arr: Vec<_> = rows
            .iter()
            .map(|((node, model, cluster), t)| {
                serde_json::json!({
                    "node": node,
                    "model": model,
                    "cluster": cluster,
                    "gen_tok_s": t,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("no benchmark history across the fleet yet");
        return Ok(());
    }
    println!("{:<14} {:<28} {:>9}", "node", "model", "gen tok/s");
    for ((node, model, cluster), t) in rows {
        let pooled = cluster
            .map(|c| format!("  [pooled: {c}]"))
            .unwrap_or_default();
        println!("{:<14} {} {:>9.1}{pooled}", node, fit(&model, 28), t);
    }
    Ok(())
}
