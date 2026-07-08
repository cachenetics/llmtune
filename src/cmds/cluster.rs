// SPDX-License-Identifier: GPL-2.0-only
//! Cluster command handlers (`llmtune cluster <op>`).

use crate::cluster::WorkerCtl;
use crate::config::{self, Config};
use crate::{cluster, model, nodeops, profile, swap, transport};
use anyhow::Result;
use serde::Serialize;

pub(crate) fn resolve_workers(cfg: &Config, names: &[String]) -> Result<Vec<config::Node>> {
    names
        .iter()
        .map(|w| {
            cfg.node(w)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("worker `{w}` not in fleet config"))
        })
        .collect()
}

/// Lenient worker resolution for teardown paths: unknown names are returned
/// separately instead of failing the whole operation. `cluster down` must stop
/// every worker it CAN reach - bailing on one renamed/removed [[node]] before
/// touching anything strands live unauthenticated rpc-servers on all the
/// others (`cluster status` is already lenient the same way).
pub(crate) fn resolve_workers_lenient(
    cfg: &Config,
    names: &[String],
) -> (Vec<config::Node>, Vec<String>) {
    let mut known = Vec::new();
    let mut unknown = Vec::new();
    for w in names {
        match cfg.node(w) {
            Some(n) => known.push(n.clone()),
            None => unknown.push(w.clone()),
        }
    }
    (known, unknown)
}

/// Resolve the address the workers' rpc-server binds. llama.cpp's rpc-server
/// is UNAUTHENTICATED - upstream documents it as an arbitrary-memory surface
/// unsafe on untrusted networks - so the default must be loopback, never
/// all-interfaces. Unconfigured + remote workers is a refusal rather than a
/// silent widening: a loopback bind on a remote worker can never form a
/// cluster (the head's TCP connect would just time out), and choosing
/// `0.0.0.0` on the user's behalf would expose every worker to its whole
/// network. The operator opts in explicitly via `rpc_bind` in the [[cluster]]
/// block; `cluster up` still warns on any non-loopback bind.
pub(crate) fn resolve_rpc_bind(
    configured: Option<&str>,
    has_remote_workers: bool,
) -> Result<String> {
    match configured {
        Some(b) => Ok(b.to_string()),
        None if !has_remote_workers => Ok("127.0.0.1".to_string()),
        None => Err(crate::agentic::refusal(
            "this cluster has remote workers but no `rpc_bind` configured. \
             llama.cpp's rpc-server is UNAUTHENTICATED (an arbitrary-memory \
             surface), so llmtune will not bind it to all interfaces by \
             default. Set `rpc_bind = \"<the worker's fleet-facing IP>\"` (or \
             `\"0.0.0.0\"` on a fully trusted network) in the [[cluster]] \
             block to opt in",
        )),
    }
}

/// True for the loopback binds that need no exposure warning.
pub(crate) fn is_loopback_bind(bind: &str) -> bool {
    matches!(bind, "127.0.0.1" | "localhost" | "::1")
}

pub(crate) fn cmd_cluster_list(cfg: &Config, json: bool) -> Result<()> {
    if json {
        let arr: Vec<_> = cfg
            .clusters
            .iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "head": c.head,
                    "workers": c.workers,
                    "rpc_port": c.rpc_port,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }
    if cfg.clusters.is_empty() {
        println!("no clusters configured (add a [[cluster]] block to fleet.toml)");
        return Ok(());
    }
    for c in &cfg.clusters {
        println!(
            "{:<12} head={}  workers=[{}]  rpc_port={}",
            c.name,
            c.head,
            c.workers.join(","),
            c.rpc_port
        );
    }
    Ok(())
}

/// One cluster's live status (also the `cluster status --json` row shape).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ClusterStatusRow {
    pub name: String,
    pub head: String,
    pub head_reachable: bool,
    pub head_healthy: bool,
    /// The model the head is currently serving, if any.
    pub serving: Option<String>,
    /// True if the active-cluster marker says THIS cluster is up on this head.
    pub active: bool,
    pub rpc_port: u16,
    pub workers: Vec<WorkerStatusRow>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WorkerStatusRow {
    pub name: String,
    /// False if the worker name isn't in the fleet config.
    pub configured: bool,
    /// Is the worker's rpc-server accepting on the cluster's RPC port?
    pub rpc_accepting: bool,
}

/// `cluster status` - each configured cluster's head/workers, whether the head
/// is currently serving, and basic worker reachability (RPC-port TCP check).
pub(crate) fn cmd_cluster_status(cfg: &Config, name: Option<&str>, json: bool) -> Result<()> {
    let clusters: Vec<_> = cfg
        .clusters
        .iter()
        .filter(|c| name.is_none_or(|n| c.name == n))
        .collect();
    if let Some(n) = name {
        if clusters.is_empty() {
            anyhow::bail!("unknown cluster `{n}`");
        }
    }
    if clusters.is_empty() {
        if json {
            println!("[]");
        } else {
            println!("no clusters configured (add a [[cluster]] block to fleet.toml)");
        }
        return Ok(());
    }
    let active = cluster::active();
    let wctl = cluster::RealWorkerCtl::new();
    let mut rows: Vec<ClusterStatusRow> = Vec::new();
    for c in clusters {
        // Head state through its transport (works local or over SSH).
        let head_status = cfg
            .node(&c.head)
            .and_then(|n| transport::for_node(n).ok())
            .map(|t| t.status());
        let (head_reachable, head_healthy, serving) = match &head_status {
            Some(s) => (s.reachable, s.healthy, s.served.clone()),
            None => (false, false, None),
        };
        let workers = c
            .workers
            .iter()
            .map(|w| match cfg.node(w) {
                Some(n) => WorkerStatusRow {
                    name: w.clone(),
                    configured: true,
                    rpc_accepting: wctl.accepting(n, c.rpc_port),
                },
                None => WorkerStatusRow {
                    name: w.clone(),
                    configured: false,
                    rpc_accepting: false,
                },
            })
            .collect();
        rows.push(ClusterStatusRow {
            name: c.name.clone(),
            head: c.head.clone(),
            head_reachable,
            head_healthy,
            serving,
            active: active.as_ref().is_some_and(|a| a.name == c.name),
            rpc_port: c.rpc_port,
            workers,
        });
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    for r in &rows {
        let head_state = if !r.head_reachable {
            "UNREACHABLE"
        } else if r.head_healthy {
            "up"
        } else {
            "down"
        };
        let serving = match (&r.serving, r.head_healthy) {
            (Some(m), true) => format!("serving {m}"),
            _ => "not serving".to_string(),
        };
        let tag = if r.active { "  [ACTIVE]" } else { "" };
        println!(
            "{:<12} head={} ({head_state})  {serving}  rpc :{}{tag}",
            r.name, r.head, r.rpc_port
        );
        for w in &r.workers {
            let state = if !w.configured {
                "NOT IN FLEET CONFIG"
            } else if w.rpc_accepting {
                "rpc accepting"
            } else {
                "rpc not answering"
            };
            println!("  worker {:<12} {state}", w.name);
        }
    }
    Ok(())
}

/// Model gate for POOLING (cluster-only; single-node `node load` deliberately
/// does not use this). Only dense-transformer chat models can shard over
/// llama.cpp RPC: a recurrent/hybrid arch crashes the pool at load - the
/// worker rejects the recurrent-state graph (`[create_node] invalid data ptr`)
/// and the head aborts. Pure, so it's unit-testable without a rack.
pub(crate) fn guard_cluster_model(m: &model::Model) -> Result<()> {
    if !model::is_chat_model(&m.name) {
        anyhow::bail!("`{}` is not a chat model", m.name);
    }
    if model::is_recurrent_arch(&m.arch) {
        anyhow::bail!(
            "`{}` is a recurrent/hybrid arch ({}) - it cannot be split over llama.cpp RPC \
             (workers reject the recurrent-state graph); pool a dense transformer instead, \
             or serve it single-node with `llmtune node load`",
            m.name,
            m.arch
        );
    }
    Ok(())
}

pub(crate) fn cmd_cluster_up(
    cfg: &Config,
    cluster_name: &str,
    model: &str,
    allow_version_skew: bool,
    json: bool,
) -> Result<()> {
    let cl = cfg
        .cluster(cluster_name)
        .ok_or_else(|| anyhow::anyhow!("unknown cluster `{cluster_name}`"))?;
    let head = cfg
        .node(&cl.head)
        .ok_or_else(|| anyhow::anyhow!("cluster head `{}` not in fleet config", cl.head))?;
    if head.transport != config::Transport::Local {
        anyhow::bail!(
            "the cluster head must be the local node in this version - run llmtune on `{}`",
            cl.head
        );
    }
    // A cluster up stages+restarts the head's llama unit - a GPU-touching swap.
    // Take the same lock swap/bench use so it can't race them on the one GPU.
    let _gpu = crate::lock::LockGuard::gpu().map_err(|_| {
        anyhow::anyhow!("the GPU is busy (a swap or benchmark is running) - try again")
    })?;
    let m = nodeops::resolve_model(head, model)?;
    guard_cluster_model(&m)?;
    let profiles = profile::load()?;
    let (prof, _used_default) = profile::resolve(&profiles, &m.arch);
    let workers = resolve_workers(cfg, &cl.workers)?;
    // Cross-node build parity, BEFORE any worker is touched: mixed llama.cpp
    // versions abort at the RPC handshake with a cryptic "malformed response"
    // on the head. A CONFIRMED mismatch is refused (a homogeneous cluster never
    // confirms one, so this can't block legitimate use; --allow-version-skew
    // overrides). An identity we can't determine only warns - the check stays
    // best-effort (rpc_bin overrides / older remote llmtune are legitimate).
    match cluster::build_parity(prof.build.as_deref(), &workers) {
        cluster::Parity::Match => {}
        cluster::Parity::Unverified { detail } => {
            eprintln!(
                "WARNING: cross-node llama.cpp build parity UNVERIFIED: {detail}. \
                 If head and workers run different llama.cpp versions the head \
                 aborts at the RPC handshake with a \"malformed response\" error."
            );
        }
        cluster::Parity::Mismatch { detail } => {
            if allow_version_skew {
                eprintln!(
                    "WARNING: proceeding despite llama.cpp build-version skew \
                     (--allow-version-skew): {detail}"
                );
            } else {
                // A guard refusal (exit 2): nothing was touched and
                // --allow-version-skew overrides.
                return Err(crate::agentic::refusal(format!(
                    "llama.cpp build version differs across cluster nodes: {detail}. \
                     A mixed-version cluster aborts at the RPC handshake \
                     (\"malformed response\") - update the stale node(s) with \
                     `llmtune build install {}`, or pass --allow-version-skew to try anyway",
                    prof.build.as_deref().unwrap_or("<name>")
                )));
            }
        }
    }
    let rpc_bin = cl
        .rpc_bin
        .clone()
        .unwrap_or_else(|| "rpc-server".to_string());
    // llama.cpp RPC has no authentication: default the workers' bind to
    // loopback and require an explicit `rpc_bind` opt-in for anything
    // reachable off-host (see `resolve_rpc_bind`). Warn whenever the
    // configured bind is non-loopback so the operator knows the pool must sit
    // on a trusted network.
    let has_remote_workers = workers
        .iter()
        .any(|w| w.transport != config::Transport::Local);
    let rpc_bind = resolve_rpc_bind(cl.rpc_bind.as_deref(), has_remote_workers)?;
    if !is_loopback_bind(&rpc_bind) {
        eprintln!(
            "WARNING: cluster RPC workers bind {rpc_bind} and llama.cpp RPC is \
             UNAUTHENTICATED (an arbitrary-memory surface). Only run this on a \
             trusted network; set `rpc_bind` on the cluster to a specific \
             fleet-facing interface to narrow exposure."
        );
    }
    let (host, port) = nodeops::parse_bind(&head.llama_url);
    let opts = swap::SwapOpts {
        host,
        port,
        // Carry the head's base-unit env (e.g. a declarative Vulkan stack)
        // through the drop-in's Environment= reset.
        base_env: swap::unit_base_env(&head.llama_unit),
        ..Default::default()
    };
    let mut wctl = cluster::RealWorkerCtl::new();
    let mut head_act = swap::SystemdActuator::new();
    let head_health = swap::HttpHealth {
        url: head.llama_url.clone(),
    };
    let out = match cluster::up(
        cluster_name,
        &head.llama_unit,
        &m,
        prof,
        &workers,
        cl.rpc_port,
        &rpc_bin,
        &rpc_bind,
        &mut wctl,
        &mut head_act,
        &head_health,
        &opts,
    ) {
        Ok(o) => o,
        Err(e) => {
            // A hard error mid-up (workers already torn down inside `up`) must
            // not leave a stale marker claiming the cluster is pooled.
            cluster::clear_active();
            return Err(e);
        }
    };
    // Maintain the active-cluster marker so a bench on the head is tagged as a
    // pooled run (history/leaderboards, SPEC 5.7) AND pools over the same
    // workers (`llama-bench --rpc` reads the recorded endpoints).
    if out.ok {
        let endpoints: Vec<String> = workers
            .iter()
            .map(|w| cluster::worker_endpoint(w, cl.rpc_port))
            .collect();
        let _ = cluster::record_active(cluster_name, &out.workers_up, &endpoints);
    } else {
        cluster::clear_active();
    }
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": out.ok,
                "cluster": out.name,
                "model": out.model,
                "reverted": out.reverted,
                "workers_up": out.workers_up,
                "detail": out.detail,
            }))?
        );
        if !out.ok {
            std::process::exit(1);
        }
        return Ok(());
    }
    let tag = if out.ok { "[ok]  " } else { "[fail]" };
    println!("{tag} {}", out.detail);
    if !out.ok {
        std::process::exit(1);
    }
    Ok(())
}

pub(crate) fn cmd_cluster_down(cfg: &Config, cluster_name: &str, json: bool) -> Result<()> {
    let cl = cfg
        .cluster(cluster_name)
        .ok_or_else(|| anyhow::anyhow!("unknown cluster `{cluster_name}`"))?;
    let head = cfg
        .node(&cl.head)
        .ok_or_else(|| anyhow::anyhow!("cluster head `{}` not in fleet config", cl.head))?;
    // Mirror the up-path guard: `down` reverts the head's drop-in via LOCAL
    // sudo/systemctl, so running it for a remote-head cluster would strip THIS
    // machine's drop-in for that unit name.
    if head.transport != config::Transport::Local {
        anyhow::bail!(
            "the cluster head must be the local node in this version - run llmtune on `{}`",
            cl.head
        );
    }
    // GPU-touching (reverts the head unit) - serialize against swap/bench.
    let _gpu = crate::lock::LockGuard::gpu().map_err(|_| {
        anyhow::anyhow!("the GPU is busy (a swap or benchmark is running) - try again")
    })?;
    // Lenient resolution (unlike `up`): a worker renamed/removed from the
    // fleet config must not wedge the teardown - the OTHER workers' live,
    // unauthenticated rpc-servers still get stopped, and the unknown ones are
    // reported so the operator can stop them by hand.
    let (workers, unknown) = resolve_workers_lenient(cfg, &cl.workers);
    for u in &unknown {
        eprintln!(
            "WARNING: worker `{u}` is not in the fleet config - cannot stop its \
             rpc-server from here. If the board is still up, stop it by hand: \
             `systemctl stop llmtune-rpc-{}` on the worker.",
            cl.rpc_port
        );
    }
    let mut wctl = cluster::RealWorkerCtl::new();
    let res = cluster::down(&head.llama_unit, &workers, cl.rpc_port, &mut wctl);
    // Clear the marker (before surfacing a teardown error) - but only if THIS
    // cluster is the active one, so `down <B>` can't erase cluster A's marker.
    if cluster::active().is_none_or(|a| a.name == cluster_name) {
        cluster::clear_active();
    }
    let failed = res?;
    let stopped = workers.len() - failed.len();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": failed.is_empty() && unknown.is_empty(),
                "cluster": cluster_name,
                "workers_stopped": stopped,
                "workers_failed": failed.iter().map(|(n, e)| serde_json::json!({
                    "name": n, "error": e,
                })).collect::<Vec<_>>(),
                "workers_unknown": unknown,
            }))?
        );
        if !failed.is_empty() || !unknown.is_empty() {
            std::process::exit(1);
        }
        return Ok(());
    }
    println!(
        "cluster `{cluster_name}` down: {stopped} worker(s) stopped, head `{}` reverted",
        cl.head
    );
    for (n, e) in &failed {
        println!("[fail] worker `{n}`: rpc-server stop FAILED ({e}) - it may still be running");
    }
    if !failed.is_empty() || !unknown.is_empty() {
        anyhow::bail!(
            "{} worker(s) could not be stopped ({} unknown) - unauthenticated \
             rpc-servers may still be listening on :{}",
            failed.len(),
            unknown.len(),
            cl.rpc_port
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_model(name: &str, arch: &str) -> model::Model {
        model::Model {
            path: std::path::PathBuf::from(format!("/models/{name}")),
            name: name.into(),
            arch: arch.into(),
            params: None,
            quant: None,
            size_bytes: 1,
            ctx_max: None,
            has_mtp: false,
        }
    }

    #[test]
    fn guard_refuses_recurrent_arch_with_actionable_message() {
        // The RPC-verification crash: LFM2.5 pooled -> worker rejects the
        // recurrent-state graph. `cluster up` must refuse it up front.
        let err = guard_cluster_model(&a_model("LFM2.5-14B-IQ4.gguf", "lfm2"))
            .expect_err("recurrent arch must be refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("recurrent/hybrid arch (lfm2)"), "{msg}");
        assert!(msg.contains("dense transformer"), "{msg}");
        // Other recurrent families are covered by the same gate.
        assert!(guard_cluster_model(&a_model("m.gguf", "mamba2")).is_err());
        assert!(guard_cluster_model(&a_model("m.gguf", "rwkv7")).is_err());
    }

    #[test]
    fn guard_allows_dense_transformers() {
        for arch in ["qwen35", "qwen35moe", "gemma3", "llama"] {
            assert!(
                guard_cluster_model(&a_model("Big-70B-Q4.gguf", arch)).is_ok(),
                "{arch} must pool"
            );
        }
    }

    #[test]
    fn guard_still_refuses_non_chat_models() {
        let err = guard_cluster_model(&a_model("nomic-embed-text.gguf", "bert")).unwrap_err();
        assert!(format!("{err:#}").contains("not a chat model"));
    }

    #[test]
    fn lenient_resolution_partitions_unknown_workers() {
        // `cluster down` must not bail on a renamed/removed [[node]]: the
        // known workers still get their rpc-servers stopped, the unknown are
        // reported. (`resolve_workers` stays strict for `up`.)
        let cfg =
            Config::load_str("[[node]]\nname=\"bc250-2\"\nhost=\"192.0.2.2\"\ntransport=\"ssh\"\n")
                .unwrap();
        let names = vec!["bc250-2".to_string(), "renamed-away".to_string()];
        let (known, unknown) = resolve_workers_lenient(&cfg, &names);
        assert_eq!(known.len(), 1);
        assert_eq!(known[0].name, "bc250-2");
        assert_eq!(unknown, vec!["renamed-away"]);
        // strict resolution still refuses (up-path behavior unchanged)
        assert!(resolve_workers(&cfg, &names).is_err());
    }

    #[test]
    fn rpc_bind_defaults_to_loopback_for_local_workers() {
        // No config + no remote workers -> the safe loopback default, never
        // 0.0.0.0 (the pre-hardening default exposed an unauthenticated
        // arbitrary-memory server on every interface).
        assert_eq!(resolve_rpc_bind(None, false).unwrap(), "127.0.0.1");
    }

    #[test]
    fn rpc_bind_unset_with_remote_workers_is_a_refusal() {
        // Remote workers need an off-loopback bind to form a cluster, but
        // choosing one silently would be the exposure #35 removed - the
        // operator must opt in via `rpc_bind`.
        let err = resolve_rpc_bind(None, true).expect_err("must refuse");
        let msg = format!("{err:#}");
        assert!(msg.contains("rpc_bind"), "{msg}");
        assert!(msg.contains("UNAUTHENTICATED"), "{msg}");
        assert!(
            crate::agentic::is_refusal(&err),
            "guard refusal, not an error"
        );
    }

    #[test]
    fn rpc_bind_explicit_value_is_honored() {
        // An explicit bind is the opt-in - used verbatim, remote or not.
        assert_eq!(
            resolve_rpc_bind(Some("192.0.2.7"), true).unwrap(),
            "192.0.2.7"
        );
        assert_eq!(resolve_rpc_bind(Some("0.0.0.0"), false).unwrap(), "0.0.0.0");
    }

    #[test]
    fn loopback_binds_classified() {
        assert!(is_loopback_bind("127.0.0.1"));
        assert!(is_loopback_bind("localhost"));
        assert!(is_loopback_bind("::1"));
        assert!(!is_loopback_bind("0.0.0.0"));
        assert!(!is_loopback_bind("192.0.2.7"));
    }
}
