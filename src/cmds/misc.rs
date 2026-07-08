// SPDX-License-Identifier: GPL-2.0-only
//! Top-level one-off command handlers: `endpoint`, `mem`, `compare`.

use crate::config::Config;
use crate::{compare, endpoint, mem, model, nodeops};
use anyhow::Result;

pub(crate) fn cmd_compare(cfg: &Config, filter: Option<String>, json: bool) -> Result<()> {
    let node = cfg.local_node();
    let recs = nodeops::history(node, 1000);
    let mut fams = compare::group(&recs);
    if let Some(f) = &filter {
        let fl = f.to_lowercase();
        fams.retain(|fam| fam.key.to_lowercase().contains(&fl));
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&fams)?);
        return Ok(());
    }
    if fams.is_empty() {
        println!("no matching benchmark history - run `llmtune node bench` first");
        return Ok(());
    }
    for fam in &fams {
        println!("{}", fam.key);
        println!(
            "  {:<10} {:>7} {:>10} {:>10} {:>5}  build",
            "quant", "ctx", "gen t/s", "prefill", "runs"
        );
        for v in &fam.variants {
            println!(
                "  {:<10} {:>7} {:>10.1} {:>10.1} {:>5}  {}",
                v.quant,
                v.ctx,
                v.best_gen_tok_s,
                v.best_prompt_tok_s,
                v.runs,
                v.build.as_deref().unwrap_or("-"),
            );
        }
        println!();
    }
    Ok(())
}

pub(crate) fn cmd_endpoint(cfg: &Config, json: bool) -> Result<()> {
    cmd_endpoint_on(cfg, None, json)
}

/// `node endpoint [--node <name>]` - the endpoint view, probed node-side over
/// the target's transport (the wire the TUI and fleet views read).
pub(crate) fn cmd_endpoint_on(cfg: &Config, sel: Option<&str>, json: bool) -> Result<()> {
    let node = crate::cmds::node::resolve_target(cfg, sel)?;
    let ep = crate::transport::for_node(node)?.endpoint()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&ep)?);
        return Ok(());
    }
    let state = if ep.healthy {
        "up"
    } else {
        "down (no model loaded yet)"
    };
    println!("endpoint   {}  [{state}]", ep.openai_base);
    println!(
        "model      {}",
        ep.model.as_deref().unwrap_or("(none served)")
    );
    let exposure = if ep.exposed {
        "exposed to LAN (0.0.0.0)".to_string()
    } else if ep.lan_base.is_some() {
        "localhost-only  (run `llmtune endpoint expose on` to reach the LAN)".to_string()
    } else {
        "localhost-only".to_string()
    };
    println!("exposure   {exposure}");
    let auth = match &ep.api_key {
        Some(k) => format!("ON  (key required: {k})"),
        None => "OFF (keyless - turn on with `llmtune endpoint auth on`)".to_string(),
    };
    println!("auth       {auth}");
    let identity = if ep.identity_branded {
        "model-branded - the chat template overrides your harness \
         (fix: `llmtune endpoint identity harness`)"
    } else {
        "harness-controlled - your system prompt defines identity"
    };
    println!("identity   {identity}\n");
    println!("{}", endpoint::snippets(&ep));
    Ok(())
}

pub(crate) fn cmd_mem(
    cfg: &Config,
    model: &str,
    ctx: Option<u64>,
    kv: &str,
    json: bool,
) -> Result<()> {
    let node = cfg.local_node();
    let m = nodeops::resolve_model(node, model)?;
    let dims = model::read_dims(&m.path);
    let ctx = ctx.unwrap_or(32768);
    let budget = mem::read_uma();
    let est = mem::estimate(m.size_bytes, dims, ctx, kv, budget);

    if json {
        let mut v = serde_json::to_value(&est)?;
        v["model"] = serde_json::json!(m.name);
        v["arch"] = serde_json::json!(m.arch);
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    println!("model   {} [{}]", m.name, m.arch);
    println!("weights      {:>6.1} GiB", mem::gib(est.weights_bytes));
    match &est.dims {
        Some(d) => println!(
            "KV @{:<6} {}  {:>6.1} GiB   ({} layers, {} kv-heads x {})",
            ctx,
            kv,
            mem::gib(est.kv_bytes),
            d.n_layers,
            d.n_head_kv,
            d.head_dim
        ),
        None => {
            println!("KV @{ctx:<6} {kv}     n/a       (header lacks dims - weights-only estimate)")
        }
    }
    println!("working set  {:>6.1} GiB", mem::gib(est.working_bytes));
    println!("{}", "-".repeat(40));
    match (&est.budget, est.headroom_bytes) {
        (Some(b), Some(h)) => {
            // Physical UMA (real 16 GiB), not vram+gtt (16.5, which double-counts
            // the shared DRAM), and what's actually allocatable right now.
            println!(
                "UMA         {:>6.1} GiB physical   free {:.1} GiB now",
                mem::gib(b.physical_total()),
                mem::gib(b.usable()),
            );
            let sign = if h >= 0 { "+" } else { "" };
            let note = if h < 0 {
                "   (over budget - lower ctx or KV quant)"
            } else {
                ""
            };
            println!("headroom    {sign}{:>5.1} GiB{note}", mem::gib_i(h));
            if let Some(mc) = est.max_ctx {
                let capped = est
                    .dims
                    .as_ref()
                    .and_then(|d| d.ctx_train)
                    .is_some_and(|t| mc >= t);
                let tag = if capped { "   (trained max)" } else { "" };
                println!("fits ctx    {mc:>6} @ {kv}{tag}");
            }
        }
        _ => println!("UMA budget   unknown   (not an amdgpu box - run on the BC-250)"),
    }
    if let Some(d) = &est.dims {
        if let Some(t) = d.ctx_train {
            println!("levers: --ctx <n> (trained {t})  --kv <type> (now {kv})");
        } else {
            println!("levers: --ctx <n>  --kv <type> (now {kv})");
        }
    }
    Ok(())
}
