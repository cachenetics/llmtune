// SPDX-License-Identifier: GPL-2.0-only
//! Cluster mode (M5): pool several BC-250s to serve one model too big for a
//! single box, via llama.cpp RPC. Each worker runs `rpc-server`; the head runs
//! `llama-server --rpc <workers> -ngl 99` so the model's layers shard across the
//! pooled memory of every box.
//!
//! The head launch is the single-node swap (reusing `swap::*`) with `--rpc`
//! injected into the per-arch profile flags. Worker control is abstracted behind
//! [`WorkerCtl`] so the orchestration - including whole-cluster auto-revert - is
//! unit-tested with mocks (no rack required). This version requires the head to
//! be the local node (run llmtune on the head); workers may be local or SSH.

use crate::config::{Node, Transport};
use crate::model::Model;
use crate::profile::Profile;
use crate::swap;
use anyhow::{bail, Result};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::Command;
use std::time::Duration;

/// Result of bringing a cluster up. (Several fields are structured result data
/// for callers/JSON; the CLI currently prints only `ok`/`detail`.)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ClusterOutcome {
    pub name: String,
    pub ok: bool,
    pub reverted: bool,
    pub model: String,
    pub workers_up: Vec<String>,
    pub detail: String,
}

/// `host:port` endpoint a worker's rpc-server listens on.
pub fn worker_endpoint(node: &Node, port: u16) -> String {
    let host = node.host.clone().unwrap_or_else(|| node.name.clone());
    format!("{host}:{port}")
}

/// Append `--rpc h1:p,h2:p` to a profile's flags (no-op for an empty worker set).
pub fn inject_rpc(flags: &str, endpoints: &[String]) -> String {
    if endpoints.is_empty() {
        flags.to_string()
    } else {
        format!("{flags} --rpc {}", endpoints.join(","))
    }
}

/// Starts/stops `rpc-server` on a worker and checks its RPC port.
pub trait WorkerCtl {
    fn start(&mut self, worker: &Node, port: u16, rpc_bin: &str, bind: &str) -> Result<()>;
    fn stop(&mut self, worker: &Node, port: u16) -> Result<()>;
    fn accepting(&self, worker: &Node, port: u16) -> bool;
}

/// Stop every worker's rpc-server, returning the workers whose stop FAILED
/// (name, error). A stranded rpc-server is an unauthenticated arbitrary-memory
/// surface, so callers must surface failures, never swallow them.
fn teardown<W: WorkerCtl>(wctl: &mut W, workers: &[Node], port: u16) -> Vec<(String, String)> {
    workers
        .iter()
        .filter_map(|w| {
            wctl.stop(w, port)
                .err()
                .map(|e| (w.name.clone(), format!("{e:#}")))
        })
        .collect()
}

/// Best-effort teardown on an `up` failure path: warn about (but do not fail
/// on) workers whose stop failed - the primary error is what the caller
/// reports, but a stranded rpc-server must still be called out.
fn teardown_warn<W: WorkerCtl>(wctl: &mut W, workers: &[Node], port: u16) {
    for (name, err) in teardown(wctl, workers, port) {
        eprintln!(
            "WARNING: could not stop rpc-server on worker `{name}` ({err}) - \
             an UNAUTHENTICATED rpc-server may still be running there \
             (stop it: `systemctl stop llmtune-rpc-{port}` on the worker)"
        );
    }
}

fn poll_accept<W: WorkerCtl>(wctl: &W, worker: &Node, port: u16, opts: &swap::SwapOpts) -> bool {
    for i in 0..opts.health_attempts {
        if wctl.accepting(worker, port) {
            return true;
        }
        if i + 1 < opts.health_attempts {
            std::thread::sleep(opts.health_interval);
        }
    }
    false
}

/// Bring a cluster up: start every worker's rpc-server, then launch the head
/// serving `head_model` with `--rpc <workers>`. Any failure tears the whole
/// cluster down cleanly (the cluster never half-forms).
#[allow(clippy::too_many_arguments)]
pub fn up<W: WorkerCtl, A: swap::Actuator, H: swap::Health>(
    cluster_name: &str,
    head_unit: &str,
    head_model: &Model,
    head_profile: &Profile,
    workers: &[Node],
    rpc_port: u16,
    rpc_bin: &str,
    rpc_bind: &str,
    wctl: &mut W,
    head_act: &mut A,
    head_health: &H,
    opts: &swap::SwapOpts,
) -> Result<ClusterOutcome> {
    let failure = |detail: String| ClusterOutcome {
        name: cluster_name.to_string(),
        ok: false,
        reverted: true,
        model: head_model.name.clone(),
        workers_up: vec![],
        detail,
    };

    // 1. Start workers (stop the ones already up if any fails).
    for (i, w) in workers.iter().enumerate() {
        if let Err(e) = wctl.start(w, rpc_port, rpc_bin, rpc_bind) {
            teardown_warn(wctl, &workers[..i], rpc_port);
            return Ok(failure(format!(
                "worker `{}` failed to start rpc-server: {e}",
                w.name
            )));
        }
    }

    // 2. Wait for each worker's RPC port to accept.
    for w in workers {
        if !poll_accept(wctl, w, rpc_port, opts) {
            teardown_warn(wctl, workers, rpc_port);
            return Ok(failure(format!(
                "worker `{}` ({}) never accepted on its RPC port",
                w.name,
                worker_endpoint(w, rpc_port)
            )));
        }
    }

    // 3. Launch the head with --rpc injected.
    let endpoints: Vec<String> = workers
        .iter()
        .map(|w| worker_endpoint(w, rpc_port))
        .collect();
    let flags = inject_rpc(&swap::adjust_flags(head_profile, head_model), &endpoints);
    let (head_bin, head_ld) = head_profile.launch();
    let dropin = swap::render_dropin(
        head_profile,
        &head_bin,
        head_ld.as_deref(),
        &flags,
        &head_model.path,
        &opts.host,
        opts.port,
        &opts.base_env,
    );
    // A stage/restart ERROR (not just an unhealthy head) must still stop the
    // workers already running, or `?` leaks live rpc-servers across the rack.
    if let Err(e) = head_act
        .stage(head_unit, &dropin)
        .and_then(|()| head_act.reload_restart(head_unit))
    {
        teardown_warn(wctl, workers, rpc_port);
        return Err(e);
    }

    if swap::poll_health(head_health, opts) {
        if opts.prewarm {
            head_health.warm();
        }
        head_act.commit()?;
        return Ok(ClusterOutcome {
            name: cluster_name.to_string(),
            ok: true,
            reverted: false,
            model: head_model.name.clone(),
            workers_up: workers.iter().map(|w| w.name.clone()).collect(),
            detail: format!(
                "cluster `{cluster_name}` up: head serving `{}` across {} worker(s)",
                head_model.name,
                workers.len()
            ),
        });
    }

    // Head failed to come up: revert head + stop all workers. Workers are torn
    // down even if the head rollback itself errors (no leaked rpc-servers).
    let rolled = head_act
        .rollback(head_unit)
        .and_then(|()| head_act.reload_restart(head_unit));
    teardown_warn(wctl, workers, rpc_port);
    rolled?;
    Ok(failure(format!(
        "head failed to come up serving `{}` - reverted head and stopped {} worker(s)",
        head_model.name,
        workers.len()
    )))
}

/// Tear a cluster down: stop every worker's rpc-server and return the head to
/// its base (non-clustered) config. Returns the workers whose stop FAILED
/// (name, error) - the caller must report them: a `down` that prints success
/// while an unauthenticated rpc-server is still listening is worse than one
/// that fails loudly. The head is reverted even when worker stops fail.
pub fn down<W: WorkerCtl>(
    head_unit: &str,
    workers: &[Node],
    rpc_port: u16,
    wctl: &mut W,
) -> Result<Vec<(String, String)>> {
    let failed = teardown(wctl, workers, rpc_port);
    swap::clear_dropin(head_unit)?;
    Ok(failed)
}

// ---------------------------------------------------------------------------
// Cross-node build parity. Mixed llama.cpp versions between the head and a
// worker abort at the RPC handshake with a cryptic "malformed response" on the
// head - long after the workers are up. `cluster up` compares the profile's
// managed-build version (commit slug) across every node FIRST, so a version
// skew is a clear refusal instead of a mid-launch crash. Best-effort: an
// identity we can't determine (older remote llmtune, literal-bin profile)
// downgrades to a loud warning, never a block on a legitimate cluster.
// ---------------------------------------------------------------------------

/// Outcome of the cross-node build-parity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parity {
    /// Every node reports the same installed build version.
    Match,
    /// At least one node reports a DIFFERENT version than the head - this WILL
    /// abort at the RPC handshake. `detail` names the skewed nodes.
    Mismatch { detail: String },
    /// One or more identities could not be determined; `detail` says why.
    Unverified { detail: String },
}

/// Pure comparison (unit-testable without nodes): `head` is the head's
/// installed version of build `build`; `workers` is `(node_name, version)` as
/// queried over each worker's transport (None = undeterminable there).
pub fn check_parity(
    build: &str,
    head: Option<&str>,
    workers: &[(String, Option<String>)],
) -> Parity {
    let Some(head_slug) = head else {
        return Parity::Unverified {
            detail: format!("build `{build}` is not installed on the head"),
        };
    };
    let mismatched: Vec<String> = workers
        .iter()
        .filter(|(_, v)| v.as_deref().is_some_and(|v| v != head_slug))
        .map(|(n, v)| format!("{n}={}", v.as_deref().unwrap_or("?")))
        .collect();
    if !mismatched.is_empty() {
        return Parity::Mismatch {
            detail: format!(
                "head has build `{build}` {head_slug}, but {}",
                mismatched.join(", ")
            ),
        };
    }
    let unknown: Vec<&str> = workers
        .iter()
        .filter(|(_, v)| v.is_none())
        .map(|(n, _)| n.as_str())
        .collect();
    if !unknown.is_empty() {
        return Parity::Unverified {
            detail: format!(
                "could not determine build `{build}` version on: {} \
                 (build not installed there, or an older llmtune without `node build-version`)",
                unknown.join(", ")
            ),
        };
    }
    Parity::Match
}

/// Gather + compare build identity across the (local) head and all workers.
/// `build` = the head profile's managed-build name (None for a literal-bin
/// profile, whose version we cannot resolve).
pub fn build_parity(build: Option<&str>, workers: &[Node]) -> Parity {
    let Some(build) = build.filter(|b| !b.is_empty()) else {
        return Parity::Unverified {
            detail: "the profile launches a literal bin path (no managed build), \
                     so build versions cannot be compared across nodes"
                .to_string(),
        };
    };
    let head = crate::build::current_version(build);
    let per_worker: Vec<(String, Option<String>)> = workers
        .iter()
        .map(|w| {
            let v = crate::transport::for_node(w)
                .ok()
                .and_then(|t| t.build_version(build).ok())
                .flatten();
            (w.name.clone(), v)
        })
        .collect();
    check_parity(build, head.as_deref(), &per_worker)
}

// ---------------------------------------------------------------------------
// Active-cluster marker: who is pooled right now. Written on a successful
// `cluster up`, cleared on `cluster down` (or a failed up), read by the bench
// path so a pooled run is tagged in history (SPEC 5.7) - a 70B served across
// 3 boxes must not masquerade as single-node throughput on the leaderboard.
// ---------------------------------------------------------------------------

/// The currently-active cluster on this head (name + pooled worker names +
/// the workers' rpc-server endpoints, so a bench on the head can pool over
/// the SAME workers with `llama-bench --rpc`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActiveCluster {
    pub name: String,
    pub members: Vec<String>,
    /// `host:port` rpc-server endpoints (default-empty for pre-existing markers).
    #[serde(default)]
    pub rpc_endpoints: Vec<String>,
}

fn active_path() -> std::path::PathBuf {
    crate::paths::state_dir().join("cluster_active.json")
}

fn record_active_at(path: &std::path::Path, ac: &ActiveCluster) -> Result<()> {
    crate::paths::write_durable(path, serde_json::to_string_pretty(ac)?.as_bytes())
}

fn active_at(path: &std::path::Path) -> Option<ActiveCluster> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Persist the active-cluster marker (call after a successful `up`).
pub fn record_active(name: &str, members: &[String], rpc_endpoints: &[String]) -> Result<()> {
    record_active_at(
        &active_path(),
        &ActiveCluster {
            name: name.to_string(),
            members: members.to_vec(),
            rpc_endpoints: rpc_endpoints.to_vec(),
        },
    )
}

/// Remove the marker (call on `down`, or when an `up` fails/reverts).
pub fn clear_active() {
    let _ = std::fs::remove_file(active_path());
}

/// The active cluster, if one is up on this head. Best-effort: the marker
/// reflects llmtune's own up/down; a cluster torn down out-of-band leaves a
/// stale marker until the next `cluster down`.
pub fn active() -> Option<ActiveCluster> {
    active_at(&active_path())
}

// ---------------------------------------------------------------------------
// Real worker control (systemd-run + TCP check). Untested on a non-BC-250 host.
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct RealWorkerCtl;

impl RealWorkerCtl {
    pub fn new() -> Self {
        RealWorkerCtl
    }
}

fn ssh_target(node: &Node) -> String {
    let host = node.host.clone().unwrap_or_else(|| node.name.clone());
    match &node.ssh_user {
        Some(u) => format!("{u}@{host}"),
        None => host,
    }
}

/// Run a privileged command on a node (locally via sudo, or over ssh+sudo).
fn run_on(node: &Node, argv: &[&str]) -> Result<()> {
    let status = if matches!(node.transport, Transport::Local) {
        Command::new("sudo").args(argv).status()?
    } else {
        let mut cmd = Command::new("ssh");
        cmd.arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg("ConnectTimeout=8")
            .arg("-o")
            .arg(format!(
                "StrictHostKeyChecking={}",
                crate::transport::hostkey_policy()
            ))
            .arg("-o")
            .arg("ServerAliveInterval=5")
            .arg("-o")
            .arg("ServerAliveCountMax=3");
        if let Some(k) = &node.ssh_key {
            cmd.arg("-i").arg(k);
        }
        // ssh joins the remaining args with spaces into ONE remote command that
        // the login shell re-splits, so each token must be shell-quoted or a
        // value with spaces/metacharacters (e.g. rpc_bin from fleet.toml) would
        // split or inject. The primary transport path already does this. The
        // `--` end-of-options guard keeps a leading-`-` target from being read
        // as an ssh option (see `transport::ssh_argv`).
        cmd.arg("--").arg(ssh_target(node)).arg("sudo");
        for a in argv {
            cmd.arg(crate::transport::sh_quote(a));
        }
        cmd.status()?
    };
    if !status.success() {
        bail!("command {:?} on `{}` failed ({status})", argv, node.name);
    }
    Ok(())
}

/// systemd-run argv to start a worker's rpc-server. When `rpc_bin` is a path,
/// the transient unit gets `LD_LIBRARY_PATH=<bin's dir>`: llmtune's installed
/// builds are dynamically linked against libggml/libllama living NEXT TO the
/// binary (their RUNPATH points at the original build tree, useless once
/// installed), so without this every build-dir rpc_bin fails at ld.so and the
/// cluster auto-reverts before it can ever form. A bare `rpc-server` (PATH
/// lookup) is left untouched.
fn start_argv(unit: &str, rpc_bin: &str, bind: &str, port: u16) -> Vec<String> {
    let mut argv = vec![
        "systemd-run".to_string(),
        "--unit".into(),
        unit.to_string(),
        "--collect".into(),
    ];
    if let Some(dir) = std::path::Path::new(rpc_bin)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        argv.push(format!("--setenv=LD_LIBRARY_PATH={}", dir.display()));
    }
    argv.extend([
        rpc_bin.to_string(),
        "-H".into(),
        bind.to_string(),
        "-p".into(),
        port.to_string(),
    ]);
    argv
}

impl WorkerCtl for RealWorkerCtl {
    fn start(&mut self, worker: &Node, port: u16, rpc_bin: &str, bind: &str) -> Result<()> {
        // Transient managed unit so `stop` is a clean systemctl stop.
        let unit = format!("llmtune-rpc-{port}");
        let argv = start_argv(&unit, rpc_bin, bind, port);
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        run_on(worker, &argv)
    }

    fn stop(&mut self, worker: &Node, port: u16) -> Result<()> {
        let unit = format!("llmtune-rpc-{port}");
        run_on(worker, &["systemctl", "stop", &unit])
    }

    fn accepting(&self, worker: &Node, port: u16) -> bool {
        let ep = worker_endpoint(worker, port);
        ep.to_socket_addrs()
            .ok()
            .and_then(|mut it| it.next())
            .map(|addr| TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile;
    use std::cell::{Cell, RefCell};
    use std::path::PathBuf;

    fn node(name: &str) -> Node {
        Node {
            name: name.into(),
            host: Some(format!("{name}.lan")),
            transport: Transport::Ssh,
            ssh_user: Some("user".into()),
            ssh_key: None,
            models_dir: "/home/user/models".into(),
            llama_unit: "llama-server.service".into(),
            llama_url: "http://127.0.0.1:8080".into(),
            power_cmd: None,
        }
    }

    fn head_model() -> Model {
        Model {
            path: PathBuf::from("/models/big-70B.gguf"),
            name: "big-70B.gguf".into(),
            arch: "llama".into(),
            params: Some("70B".into()),
            quant: Some("Q4_K_M".into()),
            size_bytes: 40_000_000_000,
            ctx_max: None,
            has_mtp: false,
        }
    }

    fn a_profile() -> Profile {
        profile::load()
            .unwrap()
            .into_iter()
            .find(|p| p.id == "_default")
            .unwrap()
    }

    fn fast_opts() -> swap::SwapOpts {
        swap::SwapOpts {
            health_attempts: 3,
            health_interval: Duration::ZERO,
            prewarm: false,
            ..Default::default()
        }
    }

    #[test]
    fn inject_rpc_appends_endpoints() {
        let f = inject_rpc("-c 32768 -ngl 99", &["a:1".into(), "b:2".into()]);
        assert_eq!(f, "-c 32768 -ngl 99 --rpc a:1,b:2");
        assert_eq!(inject_rpc("-ngl 99", &[]), "-ngl 99");
    }

    #[test]
    fn start_argv_bare_bin_skips_ld_path() {
        let argv = start_argv("llmtune-rpc-50052", "rpc-server", "0.0.0.0", 50052);
        assert!(
            !argv.iter().any(|a| a.starts_with("--setenv")),
            "bare PATH-looked-up bin must not get an LD_LIBRARY_PATH: {argv:?}"
        );
        assert_eq!(
            argv,
            vec![
                "systemd-run",
                "--unit",
                "llmtune-rpc-50052",
                "--collect",
                "rpc-server",
                "-H",
                "0.0.0.0",
                "-p",
                "50052",
            ]
        );
    }

    #[test]
    fn start_argv_build_dir_bin_sets_ld_library_path() {
        // Installed builds are dynamically linked against libs in the bin's own
        // dir (stale RUNPATH) - the unit must carry LD_LIBRARY_PATH or the
        // worker dies at ld.so and the cluster can never come up.
        let argv = start_argv(
            "llmtune-rpc-50052",
            "/var/lib/llmtune/builds/vk/local/rpc-server",
            "127.0.0.1",
            50052,
        );
        let setenv = "--setenv=LD_LIBRARY_PATH=/var/lib/llmtune/builds/vk/local";
        let si = argv.iter().position(|a| a == setenv).expect("setenv arg");
        let bi = argv
            .iter()
            .position(|a| a == "/var/lib/llmtune/builds/vk/local/rpc-server")
            .expect("bin arg");
        assert!(si < bi, "--setenv must precede the command: {argv:?}");
    }

    #[test]
    fn worker_endpoint_uses_host() {
        assert_eq!(
            worker_endpoint(&node("bc250-2"), 50052),
            "bc250-2.lan:50052"
        );
    }

    // ---- mocks ----

    struct MockWorker {
        start_fails_on: Option<String>,
        stop_fails_on: Option<String>,
        accepts: bool,
        started: RefCell<Vec<String>>,
        stopped: RefCell<Vec<String>>,
    }
    impl WorkerCtl for MockWorker {
        fn start(&mut self, worker: &Node, _p: u16, _b: &str, _bind: &str) -> Result<()> {
            if self.start_fails_on.as_deref() == Some(worker.name.as_str()) {
                bail!("mock start failure");
            }
            self.started.borrow_mut().push(worker.name.clone());
            Ok(())
        }
        fn stop(&mut self, worker: &Node, _p: u16) -> Result<()> {
            if self.stop_fails_on.as_deref() == Some(worker.name.as_str()) {
                bail!("mock stop failure");
            }
            self.stopped.borrow_mut().push(worker.name.clone());
            Ok(())
        }
        fn accepting(&self, _w: &Node, _p: u16) -> bool {
            self.accepts
        }
    }

    struct MockAct {
        committed: Cell<bool>,
        rolled_back: Cell<bool>,
    }
    impl swap::Actuator for MockAct {
        fn stage(&mut self, _u: &str, _d: &str) -> Result<()> {
            Ok(())
        }
        fn rollback(&mut self, _u: &str) -> Result<()> {
            self.rolled_back.set(true);
            Ok(())
        }
        fn commit(&mut self) -> Result<()> {
            self.committed.set(true);
            Ok(())
        }
        fn reload_restart(&mut self, _u: &str) -> Result<()> {
            Ok(())
        }
    }

    struct MockHealth {
        healthy: bool,
    }
    impl swap::Health for MockHealth {
        fn healthy(&self) -> bool {
            self.healthy
        }
        fn served(&self) -> Option<String> {
            None
        }
        fn warm(&self) {}
    }

    fn run_up(
        workers: &[Node],
        start_fails_on: Option<String>,
        accepts: bool,
        head_healthy: bool,
    ) -> (ClusterOutcome, bool, bool, Vec<String>) {
        let mut w = MockWorker {
            start_fails_on,
            stop_fails_on: None,
            accepts,
            started: RefCell::new(vec![]),
            stopped: RefCell::new(vec![]),
        };
        let mut act = MockAct {
            committed: Cell::new(false),
            rolled_back: Cell::new(false),
        };
        let health = MockHealth {
            healthy: head_healthy,
        };
        let out = up(
            "big",
            "llama-server.service",
            &head_model(),
            &a_profile(),
            workers,
            50052,
            "rpc-server",
            "127.0.0.1",
            &mut w,
            &mut act,
            &health,
            &fast_opts(),
        )
        .unwrap();
        let stopped = w.stopped.borrow().clone();
        (out, act.committed.get(), act.rolled_back.get(), stopped)
    }

    #[test]
    fn active_marker_roundtrips_and_clears() {
        // Path-injected so this doesn't race the build tests' LLMTUNE_STATE_DIR.
        let dir = std::env::temp_dir().join(format!("llmtune-cluster-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = dir.join("cluster_active.json");
        assert!(active_at(&p).is_none(), "no marker -> no active cluster");
        record_active_at(
            &p,
            &ActiveCluster {
                name: "big".into(),
                members: vec!["bc250-2".into(), "bc250-3".into()],
                rpc_endpoints: vec!["bc250-2.lan:50052".into(), "bc250-3.lan:50052".into()],
            },
        )
        .unwrap();
        let ac = active_at(&p).expect("marker must read back");
        assert_eq!(ac.name, "big");
        assert_eq!(ac.members, vec!["bc250-2", "bc250-3"]);
        assert_eq!(
            ac.rpc_endpoints,
            vec!["bc250-2.lan:50052", "bc250-3.lan:50052"]
        );
        std::fs::remove_file(&p).unwrap();
        assert!(active_at(&p).is_none(), "cleared marker -> none");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn active_marker_parses_pre_endpoint_shape() {
        // A marker written before rpc_endpoints existed must still parse
        // (serde default), so an upgrade doesn't wedge the bench path.
        let ac: ActiveCluster =
            serde_json::from_str(r#"{"name":"big","members":["bc250-2"]}"#).unwrap();
        assert_eq!(ac.name, "big");
        assert!(ac.rpc_endpoints.is_empty());
    }

    #[test]
    fn parity_match_when_all_slugs_equal() {
        let ws = vec![
            ("bc250-2".to_string(), Some("a1b2c3d4e5f6".to_string())),
            ("bc250-3".to_string(), Some("a1b2c3d4e5f6".to_string())),
        ];
        assert_eq!(
            check_parity("vulkan", Some("a1b2c3d4e5f6"), &ws),
            Parity::Match
        );
        // No workers is trivially a match (single-node "cluster").
        assert_eq!(
            check_parity("vulkan", Some("a1b2c3d4e5f6"), &[]),
            Parity::Match
        );
    }

    #[test]
    fn parity_mismatch_names_the_skewed_node() {
        // The exact footgun: one worker on an older llama.cpp -> the head aborts
        // at the RPC handshake. The check must call it out BEFORE launch.
        let ws = vec![
            ("bc250-2".to_string(), Some("a1b2c3d4e5f6".to_string())),
            ("bc250-3".to_string(), Some("0ldc0mm17abc".to_string())),
        ];
        match check_parity("vulkan", Some("a1b2c3d4e5f6"), &ws) {
            Parity::Mismatch { detail } => {
                assert!(detail.contains("bc250-3=0ldc0mm17abc"), "{detail}");
                assert!(detail.contains("a1b2c3d4e5f6"), "{detail}");
                assert!(
                    !detail.contains("bc250-2="),
                    "matching node must not be listed: {detail}"
                );
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn parity_unverified_when_a_node_cannot_report() {
        // A worker whose identity can't be determined must WARN (unverified),
        // not silently pass and not hard-block a possibly-homogeneous cluster.
        let ws = vec![
            ("bc250-2".to_string(), Some("a1b2c3d4e5f6".to_string())),
            ("bc250-3".to_string(), None),
        ];
        match check_parity("vulkan", Some("a1b2c3d4e5f6"), &ws) {
            Parity::Unverified { detail } => assert!(detail.contains("bc250-3"), "{detail}"),
            other => panic!("expected Unverified, got {other:?}"),
        }
        // Head itself not installed -> also unverified.
        assert!(matches!(
            check_parity("vulkan", None, &ws),
            Parity::Unverified { .. }
        ));
        // A confirmed mismatch outranks an unknown (refusal beats a warning).
        let ws = vec![
            ("bc250-2".to_string(), None),
            ("bc250-3".to_string(), Some("0ldc0mm17abc".to_string())),
        ];
        assert!(matches!(
            check_parity("vulkan", Some("a1b2c3d4e5f6"), &ws),
            Parity::Mismatch { .. }
        ));
    }

    #[test]
    fn parity_literal_bin_profile_is_unverified() {
        assert!(matches!(build_parity(None, &[]), Parity::Unverified { .. }));
        assert!(matches!(
            build_parity(Some(""), &[]),
            Parity::Unverified { .. }
        ));
    }

    #[test]
    fn cluster_up_success() {
        let workers = vec![node("bc250-2"), node("bc250-3")];
        let (out, committed, rolled_back, _stopped) = run_up(&workers, None, true, true);
        assert!(out.ok);
        assert!(!out.reverted);
        assert!(committed);
        assert!(!rolled_back);
        assert_eq!(out.workers_up.len(), 2);
    }

    #[test]
    fn cluster_up_worker_start_fail_tears_down() {
        let workers = vec![node("bc250-2"), node("bc250-3")];
        // second worker fails to start -> first should be stopped, head never launched
        let (out, committed, _rb, stopped) = run_up(&workers, Some("bc250-3".into()), true, true);
        assert!(!out.ok);
        assert!(out.reverted);
        assert!(!committed);
        assert_eq!(stopped, vec!["bc250-2"]); // the one already up
    }

    #[test]
    fn cluster_up_head_fail_reverts_all() {
        let workers = vec![node("bc250-2"), node("bc250-3")];
        let (out, committed, rolled_back, stopped) = run_up(&workers, None, true, false);
        assert!(!out.ok);
        assert!(out.reverted);
        assert!(!committed);
        assert!(rolled_back); // head drop-in rolled back
        assert_eq!(stopped.len(), 2); // both workers stopped
    }

    #[test]
    fn cluster_up_stage_error_still_stops_workers() {
        // A hard ERROR from the head actuator (not just an unhealthy head) must
        // not leak running rpc-servers on the workers.
        struct FailingStage;
        impl swap::Actuator for FailingStage {
            fn stage(&mut self, _u: &str, _d: &str) -> Result<()> {
                bail!("disk full")
            }
            fn rollback(&mut self, _u: &str) -> Result<()> {
                Ok(())
            }
            fn commit(&mut self) -> Result<()> {
                Ok(())
            }
            fn reload_restart(&mut self, _u: &str) -> Result<()> {
                Ok(())
            }
        }
        let workers = vec![node("bc250-2"), node("bc250-3")];
        let mut w = MockWorker {
            start_fails_on: None,
            stop_fails_on: None,
            accepts: true,
            started: RefCell::new(vec![]),
            stopped: RefCell::new(vec![]),
        };
        let health = MockHealth { healthy: true };
        let res = up(
            "big",
            "llama-server.service",
            &head_model(),
            &a_profile(),
            &workers,
            50052,
            "rpc-server",
            "127.0.0.1",
            &mut w,
            &mut FailingStage,
            &health,
            &fast_opts(),
        );
        assert!(res.is_err(), "stage error must propagate");
        assert_eq!(
            *w.stopped.borrow(),
            vec!["bc250-2", "bc250-3"],
            "both workers must be stopped on the error path"
        );
    }

    #[test]
    fn teardown_reports_stop_failures_and_stops_the_rest() {
        // The #41 core: one worker whose stop fails must be REPORTED, and the
        // other workers must still be stopped (no bail-on-first).
        let workers = vec![node("bc250-2"), node("bc250-3"), node("bc250-4")];
        let mut w = MockWorker {
            start_fails_on: None,
            stop_fails_on: Some("bc250-3".into()),
            accepts: true,
            started: RefCell::new(vec![]),
            stopped: RefCell::new(vec![]),
        };
        let failed = teardown(&mut w, &workers, 50052);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "bc250-3");
        assert!(failed[0].1.contains("mock stop failure"), "{failed:?}");
        assert_eq!(
            *w.stopped.borrow(),
            vec!["bc250-2", "bc250-4"],
            "the failing worker must not block the others"
        );
    }

    #[test]
    fn cluster_up_worker_not_accepting_tears_down() {
        let workers = vec![node("bc250-2")];
        let (out, _c, _rb, stopped) = run_up(&workers, None, false, true);
        assert!(!out.ok);
        assert!(out.detail.contains("never accepted"));
        assert_eq!(stopped.len(), 1);
    }
}
