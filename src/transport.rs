// SPDX-License-Identifier: GPL-2.0-only
//! Node transport abstraction (M4). Node operations are defined once as the
//! [`NodeTransport`] trait; the controller is identical whether a node is local
//! or reached over SSH. The SSH transport simply runs `llmtune --json node
//! <op>` on the remote and parses the JSON the same CLI emits locally.
//!
//! The Agent (daemon) transport is designed in but lands later (SPEC M8).

use crate::bench::BenchSpec;
use crate::config::{Node, Transport};
use crate::endpoint::{self, Endpoint};
use crate::history::Record;
use crate::telemetry::{self, Telemetry};
use crate::{doctor, llama, nodeops, profile};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Serializable per-model view (the `node list --json` row shape).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub arch: String,
    #[serde(default)]
    pub params: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    pub size_gib: f64,
    #[serde(default)]
    pub ctx_max: Option<u64>,
    pub profile: String,
    pub used_default: bool,
    pub served: bool,
    #[serde(default)]
    pub bin: String,
    #[serde(default)]
    pub flags: String,
    #[serde(default)]
    pub overridden: bool,
    /// Memory-fit estimate (weights + KV vs UMA budget), computed node-side.
    #[serde(default)]
    pub mem: Option<crate::mem::MemEstimate>,
    /// True if the GGUF carries MTP layers (served with `--spec-type draft-mtp`);
    /// false models have the MTP flags stripped at load. See swap::adjust_flags.
    #[serde(default)]
    pub has_mtp: bool,
}

impl From<&nodeops::ModelRow> for ModelInfo {
    fn from(r: &nodeops::ModelRow) -> Self {
        ModelInfo {
            name: r.model.name.clone(),
            arch: r.model.arch.clone(),
            params: r.model.params.clone(),
            quant: r.model.quant.clone(),
            size_gib: (r.model.size_gib() * 10.0).round() / 10.0,
            ctx_max: r.model.ctx_max,
            profile: r.profile_id.clone(),
            used_default: r.used_default,
            served: r.served,
            bin: r.bin.clone(),
            flags: r.flags.clone(),
            overridden: r.overridden,
            mem: r.mem.clone(),
            has_mtp: r.model.has_mtp,
        }
    }
}

/// Flattened swap outcome (so the local `--json` and the SSH parse agree).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwapReport {
    pub from: Option<String>,
    pub to: String,
    pub ok: bool,
    pub reverted: bool,
    pub elapsed_secs: f64,
    pub used_default: bool,
    pub detail: String,
}

/// The `node build-version --json` wire shape: which version (commit slug) of a
/// managed build is installed on a node. `version: null` = not installed there.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BuildVersion {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
}

/// A one-line node status for the fleet view.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeStatus {
    pub name: String,
    pub reachable: bool,
    pub healthy: bool,
    /// True while a benchmark has INTENTIONALLY paused llama-server on this
    /// node (the bench-pause marker is present - see `nodeops::bench_marker_path`).
    /// A down server then reads "benchmarking (paused)", not a fault. Additive
    /// `--json` field: old remotes simply omit it (defaults false); existing
    /// fields are unchanged.
    #[serde(default)]
    pub benchmarking: bool,
    #[serde(default)]
    pub served: Option<String>,
    pub models: usize,
    #[serde(default)]
    pub last_model: Option<String>,
    #[serde(default)]
    pub last_gen_tok_s: Option<f64>,
}

impl NodeStatus {
    pub(crate) fn unreachable(name: &str) -> NodeStatus {
        NodeStatus {
            name: name.to_string(),
            reachable: false,
            healthy: false,
            benchmarking: false,
            served: None,
            models: 0,
            last_model: None,
            last_gen_tok_s: None,
        }
    }
}

/// The node-op surface. Every method maps to one `llmtune node <op>`. This is
/// the WHOLE node contract: anything node-scoped (telemetry, doctor, endpoint,
/// per-model overrides) goes through here, so a drilled-into remote node shows
/// and edits the REMOTE machine's state, never the controller's.
pub trait NodeTransport {
    fn status(&self) -> NodeStatus;
    fn list(&self) -> Result<Vec<ModelInfo>>;
    fn load(&self, query: &str) -> Result<SwapReport>;
    fn unload(&self) -> Result<()>;
    fn server_ctl(&self, action: &str) -> Result<()>;
    fn bench(&self, spec: &BenchSpec) -> Result<Record>;
    fn history(&self, limit: usize) -> Result<Vec<Record>>;
    /// Live GPU telemetry from THIS node (None if unreadable/unreachable).
    fn telemetry(&self) -> Option<Telemetry>;
    /// Preflight checks run ON this node.
    fn doctor(&self) -> Result<Vec<doctor::Check>>;
    /// This node's OpenAI-compatible endpoint (probed node-side).
    fn endpoint(&self) -> Result<Endpoint>;
    /// Set (`Some`) or clear (`None`) a per-MODEL flag override ON this node -
    /// the TUI flag editor's persistence, stored in the node's own config.
    fn set_model_flags(&self, model: &str, flags: Option<&str>) -> Result<()>;
    /// The installed version (commit slug) of managed build `name` ON this node
    /// (None = not installed there). What the cluster build-parity check queries:
    /// mixed llama.cpp versions across head + workers abort at the RPC handshake.
    fn build_version(&self, name: &str) -> Result<Option<String>>;
}

/// Build the controller's transport for a node from its config.
pub fn for_node(node: &Node) -> Result<Box<dyn NodeTransport>> {
    match node.transport {
        Transport::Local => Ok(Box::new(LocalTransport { node: node.clone() })),
        Transport::Ssh => Ok(Box::new(SshTransport { node: node.clone() })),
        Transport::Agent => bail!(
            "agent transport for `{}` lands in a later milestone",
            node.name
        ),
    }
}

// ---------------------------------------------------------------------------
// Local
// ---------------------------------------------------------------------------

pub struct LocalTransport {
    node: Node,
}

impl NodeTransport for LocalTransport {
    fn status(&self) -> NodeStatus {
        let props = llama::props(&self.node.llama_url);
        let served = props.as_ref().and_then(llama::name_from_props);
        let healthy = llama::health_ok(&self.node.llama_url);
        let models = nodeops::list(&self.node).map(|v| v.len()).unwrap_or(0);
        // "last" = the most recent *plausible* run, so a fantasy record (old
        // /completion path) can't headline the node as e.g. "last 475.6 t/s".
        let hist = nodeops::history(&self.node, 50);
        let (last_model, last_gen) = hist
            .iter()
            .find(|r| r.perf.is_plausible())
            .map(|r| (Some(r.model.clone()), Some(r.perf.gen_tok_s)))
            .unwrap_or((None, None));
        NodeStatus {
            name: self.node.name.clone(),
            reachable: true,
            healthy,
            benchmarking: nodeops::bench_marker_present(),
            served,
            models,
            last_model,
            last_gen_tok_s: last_gen,
        }
    }

    fn list(&self) -> Result<Vec<ModelInfo>> {
        Ok(nodeops::list(&self.node)?
            .iter()
            .map(ModelInfo::from)
            .collect())
    }

    fn load(&self, query: &str) -> Result<SwapReport> {
        let (o, used_default) = nodeops::load(&self.node, query)?;
        Ok(SwapReport {
            from: o.from,
            to: o.to,
            ok: o.ok,
            reverted: o.reverted,
            elapsed_secs: (o.elapsed_secs * 10.0).round() / 10.0,
            used_default,
            detail: o.detail,
        })
    }

    fn unload(&self) -> Result<()> {
        nodeops::unload(&self.node)
    }

    fn server_ctl(&self, action: &str) -> Result<()> {
        nodeops::server_ctl(&self.node, action)
    }

    fn bench(&self, spec: &BenchSpec) -> Result<Record> {
        nodeops::bench(&self.node, spec)
    }

    fn history(&self, limit: usize) -> Result<Vec<Record>> {
        Ok(nodeops::history(&self.node, limit))
    }

    fn telemetry(&self) -> Option<Telemetry> {
        telemetry::read()
    }

    fn doctor(&self) -> Result<Vec<doctor::Check>> {
        Ok(doctor::run(&self.node))
    }

    fn endpoint(&self) -> Result<Endpoint> {
        Ok(endpoint::probe(&self.node))
    }

    fn set_model_flags(&self, model: &str, flags: Option<&str>) -> Result<()> {
        match flags {
            Some(f) => profile::set_override(model, f),
            None => profile::clear_override(model),
        }
    }

    fn build_version(&self, name: &str) -> Result<Option<String>> {
        Ok(crate::build::current_version(name))
    }
}

// ---------------------------------------------------------------------------
// SSH
// ---------------------------------------------------------------------------

pub struct SshTransport {
    node: Node,
}

/// POSIX single-quote an argument for the REMOTE login shell. ssh joins the
/// remote-command argv with spaces and hands the string to the remote shell,
/// which re-splits it - so an arg containing whitespace (a spaced `--flags`
/// value, a model query with a space) must be quoted or it arrives at the
/// remote clap as several arguments. Embedded single quotes become `'\''`.
/// Plain `[A-Za-z0-9_@%+=:,./-]` tokens pass through unquoted for readability.
pub(crate) fn sh_quote(s: &str) -> String {
    let safe = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if safe {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// The `StrictHostKeyChecking` value every llmtune ssh invocation uses.
///
/// Default `accept-new`: a KNOWN host is verified against the user's
/// `known_hosts` (a changed key fails closed), and a first-seen host's key is
/// recorded (trust-on-first-use). For a hostile network, set
/// `LLMTUNE_SSH_STRICT=yes` to require every host key to already be pinned in
/// `known_hosts` (no TOFU; first contact to an unpinned host fails). See the
/// README's "SSH host keys" section for how to pre-pin keys.
pub(crate) fn hostkey_policy() -> &'static str {
    hostkey_policy_from(std::env::var("LLMTUNE_SSH_STRICT").ok().as_deref())
}

/// Pure core of [`hostkey_policy`] (env value injected) so it unit-tests.
/// Only two postures exist on purpose: there is no way to turn checking OFF.
pub(crate) fn hostkey_policy_from(v: Option<&str>) -> &'static str {
    match v.map(str::trim) {
        Some("yes") | Some("1") | Some("true") => "yes",
        _ => "accept-new",
    }
}

/// Build the ssh argv to run `llmtune --json node <remote_args>` on a node.
/// Remote args are shell-quoted (see [`sh_quote`]) so the remote login shell
/// reconstructs the exact argv. Pure (no IO) so it can be unit-tested.
pub fn ssh_argv(node: &Node, remote_args: &[&str]) -> Vec<String> {
    let mut v = vec![
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        "-o".to_string(),
        "ConnectTimeout=8".to_string(),
        // Fail closed on a changed key; the posture for first-seen hosts is
        // configurable (LLMTUNE_SSH_STRICT) rather than inheriting whatever the
        // invoking user's ssh_config default happens to be.
        "-o".to_string(),
        format!("StrictHostKeyChecking={}", hostkey_policy()),
        // Bound a connected-but-hung session (ConnectTimeout only covers connect).
        "-o".to_string(),
        "ServerAliveInterval=5".to_string(),
        "-o".to_string(),
        "ServerAliveCountMax=3".to_string(),
    ];
    if let Some(k) = &node.ssh_key {
        v.push("-i".to_string());
        v.push(k.clone());
    }
    let host = node.host.clone().unwrap_or_else(|| "localhost".to_string());
    let target = match &node.ssh_user {
        Some(u) => format!("{u}@{host}"),
        None => host,
    };
    // End-of-options guard: config load rejects leading-`-` host/user, but the
    // argv itself must not depend on that - a target starting with `-` would
    // otherwise be parsed as an ssh option (`-oProxyCommand=...` is command
    // execution). Defense in depth on every ssh invocation.
    v.push("--".to_string());
    v.push(target);
    v.push("llmtune".to_string());
    v.push("--json".to_string());
    v.push("node".to_string());
    v.extend(remote_args.iter().map(|s| sh_quote(s)));
    v
}

/// Per-stream cap on remote command output. Every remote reply llmtune parses
/// (status/list/bench JSON) is tiny; a hostile or garbled peer streaming
/// gigabytes must not be buffered into the controller's memory.
const SSH_OUTPUT_MAX: usize = 8 * 1024 * 1024;

/// Wall-clock deadline for one remote command. Generous because a remote
/// `node bench` legitimately runs for minutes; the keepalive options already
/// kill a DEAD connection - this bounds a LIVE peer that trickles forever.
const SSH_DEADLINE: Duration = Duration::from_secs(600);

/// `Command::output()` with bounds: each stream is capped at `max` bytes
/// (excess is drained and discarded so the child never blocks on a full pipe)
/// and the child is killed once `deadline` elapses.
#[derive(Debug)]
pub(crate) struct CappedOutput {
    pub status: std::process::ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub truncated: bool,
}

pub(crate) fn output_capped(
    cmd: &mut Command,
    max: usize,
    deadline: Duration,
) -> Result<CappedOutput> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().context("spawning command")?;
    fn drain<R: Read + Send + 'static>(
        mut r: R,
        max: usize,
    ) -> std::thread::JoinHandle<(Vec<u8>, bool)> {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut truncated = false;
            let mut chunk = [0u8; 64 * 1024];
            loop {
                match r.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let take = n.min(max.saturating_sub(buf.len()));
                        buf.extend_from_slice(&chunk[..take]);
                        if take < n {
                            truncated = true; // keep draining so the child never stalls
                        }
                    }
                }
            }
            (buf, truncated)
        })
    }
    let so = drain(child.stdout.take().expect("stdout piped"), max);
    let se = drain(child.stderr.take().expect("stderr piped"), max);
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if start.elapsed() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "command exceeded the {}s deadline; killed",
                deadline.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let (stdout, t_out) = so.join().unwrap_or_default();
    let (stderr, t_err) = se.join().unwrap_or_default();
    Ok(CappedOutput {
        status,
        stdout,
        stderr,
        truncated: t_out || t_err,
    })
}

impl SshTransport {
    fn run(&self, remote_args: &[&str]) -> Result<String> {
        let argv = ssh_argv(&self.node, remote_args);
        let mut cmd = Command::new("ssh");
        cmd.args(&argv);
        let out = output_capped(&mut cmd, SSH_OUTPUT_MAX, SSH_DEADLINE)
            .with_context(|| format!("running ssh for node `{}`", self.node.name))?;
        if !out.status.success() {
            bail!(
                "ssh `{}` {:?} failed: {}",
                self.node.name,
                remote_args,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        if out.truncated {
            // A capped reply can never parse as the JSON we asked for; say why.
            bail!(
                "ssh `{}` {:?} returned more than {} bytes - refusing the truncated reply",
                self.node.name,
                remote_args,
                SSH_OUTPUT_MAX
            );
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn json<T: for<'de> Deserialize<'de>>(&self, remote_args: &[&str]) -> Result<T> {
        let raw = self.run(remote_args)?;
        serde_json::from_str(&raw)
            .with_context(|| format!("parsing remote JSON from node `{}`", self.node.name))
    }
}

impl NodeTransport for SshTransport {
    fn status(&self) -> NodeStatus {
        self.json::<NodeStatus>(&["status"])
            .unwrap_or_else(|_| NodeStatus::unreachable(&self.node.name))
    }

    fn list(&self) -> Result<Vec<ModelInfo>> {
        self.json(&["list"])
    }

    fn load(&self, query: &str) -> Result<SwapReport> {
        self.json(&["load", query])
    }

    fn unload(&self) -> Result<()> {
        self.run(&["unload"]).map(|_| ())
    }

    fn server_ctl(&self, action: &str) -> Result<()> {
        self.run(&["server", action]).map(|_| ())
    }

    fn bench(&self, spec: &BenchSpec) -> Result<Record> {
        let pt = spec.prompt_tokens.to_string();
        let gt = spec.gen_tokens.to_string();
        let rp = spec.repeats.to_string();
        self.json(&[
            "bench",
            "--prompt-tokens",
            &pt,
            "--gen-tokens",
            &gt,
            "--repeats",
            &rp,
        ])
    }

    fn history(&self, limit: usize) -> Result<Vec<Record>> {
        let l = limit.to_string();
        self.json(&["history", "--limit", &l])
    }

    fn telemetry(&self) -> Option<Telemetry> {
        // `node gpu --json` emits an error object when telemetry is unavailable,
        // which fails the Telemetry parse - collapsing to None, same as local.
        self.json::<Telemetry>(&["gpu"]).ok()
    }

    fn doctor(&self) -> Result<Vec<doctor::Check>> {
        self.json(&["doctor"])
    }

    fn endpoint(&self) -> Result<Endpoint> {
        self.json(&["endpoint"])
    }

    fn set_model_flags(&self, model: &str, flags: Option<&str>) -> Result<()> {
        match flags {
            Some(f) => self
                .run(&["profile", "set-model", model, "--flags", f])
                .map(|_| ()),
            None => self
                .run(&["profile", "set-model", model, "--reset"])
                .map(|_| ()),
        }
    }

    fn build_version(&self, name: &str) -> Result<Option<String>> {
        self.json::<BuildVersion>(&["build-version", name])
            .map(|b| b.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_status_json_benchmarking_is_additive() {
        // An old remote's JSON (no `benchmarking` key) parses as false.
        let old = r#"{"name":"n","reachable":true,"healthy":false,"served":null,"models":0}"#;
        let s: NodeStatus = serde_json::from_str(old).expect("pre-marker JSON must still parse");
        assert!(!s.benchmarking);
        // The new field rides the wire without disturbing existing fields.
        let mut s = NodeStatus::unreachable("n");
        s.reachable = true;
        s.benchmarking = true;
        let j = serde_json::to_string(&s).unwrap();
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        assert_eq!(v["benchmarking"], true);
        assert_eq!(v["reachable"], true);
        assert_eq!(v["healthy"], false);
        assert!(v["served"].is_null());
        let back: NodeStatus = serde_json::from_str(&j).unwrap();
        assert!(back.benchmarking);
    }

    #[test]
    fn ssh_argv_sets_hostkey_and_keepalive_options() {
        let v = ssh_argv(&ssh_node(), &["doctor"]);
        assert!(v.iter().any(|a| a.starts_with("StrictHostKeyChecking=")));
        assert!(v.contains(&"ServerAliveInterval=5".to_string()));
        assert!(v.contains(&"ServerAliveCountMax=3".to_string()));
    }

    #[test]
    fn hostkey_policy_default_and_strict() {
        // Default (unset / unrecognized) = accept-new: known_hosts is honored,
        // a changed key fails, a first-seen key is recorded (TOFU).
        assert_eq!(hostkey_policy_from(None), "accept-new");
        assert_eq!(hostkey_policy_from(Some("")), "accept-new");
        assert_eq!(hostkey_policy_from(Some("banana")), "accept-new");
        // Strict = require a pre-pinned known_hosts entry (no TOFU).
        assert_eq!(hostkey_policy_from(Some("yes")), "yes");
        assert_eq!(hostkey_policy_from(Some("1")), "yes");
        assert_eq!(hostkey_policy_from(Some("true")), "yes");
        assert_eq!(hostkey_policy_from(Some(" yes ")), "yes");
        // There is deliberately no value that DISABLES host-key checking.
        assert_eq!(hostkey_policy_from(Some("no")), "accept-new");
        assert_eq!(hostkey_policy_from(Some("off")), "accept-new");
    }

    #[test]
    fn output_capped_bounds_a_flooding_child() {
        // A child streaming far more than the cap: the buffer stays at the cap,
        // the child still exits cleanly (excess is drained, not left to block
        // on a full pipe), and the truncation is reported.
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "head -c 1000000 /dev/zero"]);
        let out = output_capped(&mut cmd, 1024, Duration::from_secs(30)).unwrap();
        assert!(out.status.success());
        assert_eq!(out.stdout.len(), 1024, "capped at max");
        assert!(out.truncated);
    }

    #[test]
    fn output_capped_kills_a_hung_child() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "sleep 30"]);
        let start = Instant::now();
        let err = output_capped(&mut cmd, 1024, Duration::from_millis(200)).unwrap_err();
        assert!(err.to_string().contains("deadline"), "{err}");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "must not wait for the child's own exit"
        );
    }

    #[test]
    fn output_capped_small_output_untouched() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "printf hello; printf world >&2"]);
        let out = output_capped(&mut cmd, 1024, Duration::from_secs(30)).unwrap();
        assert!(out.status.success());
        assert_eq!(out.stdout, b"hello");
        assert_eq!(out.stderr, b"world");
        assert!(!out.truncated);
    }

    fn ssh_node() -> Node {
        Node {
            name: "bc250-2".into(),
            host: Some("192.0.2.11".into()),
            transport: Transport::Ssh,
            ssh_user: Some("user".into()),
            ssh_key: Some("/home/me/.ssh/id_ed25519".into()),
            models_dir: "/home/user/models".into(),
            llama_unit: "llama-server.service".into(),
            llama_url: "http://127.0.0.1:8080".into(),
            power_cmd: None,
        }
    }

    #[test]
    fn ssh_argv_builds_remote_command() {
        let v = ssh_argv(&ssh_node(), &["load", "qwen"]);
        // identity/options present
        assert!(v.contains(&"BatchMode=yes".to_string()));
        assert!(v.contains(&"-i".to_string()));
        assert!(v.contains(&"/home/me/.ssh/id_ed25519".to_string()));
        // target user@host
        assert!(v.contains(&"user@192.0.2.11".to_string()));
        // remote invocation, in order
        let tail: Vec<&str> = v.iter().rev().take(5).rev().map(|s| s.as_str()).collect();
        assert_eq!(tail, vec!["llmtune", "--json", "node", "load", "qwen"]);
    }

    #[test]
    fn ssh_argv_without_user_or_key() {
        let mut n = ssh_node();
        n.ssh_user = None;
        n.ssh_key = None;
        n.host = Some("box".into());
        let v = ssh_argv(&n, &["list"]);
        assert!(!v.contains(&"-i".to_string()));
        assert!(v.contains(&"box".to_string()));
    }

    #[test]
    fn ssh_argv_end_of_options_guard_precedes_target() {
        // A crafted host/user beginning with `-` must never be parsed as an ssh
        // option (`-oProxyCommand=...` would be command execution): `--` must
        // sit immediately before the target on EVERY invocation.
        let mut n = ssh_node();
        n.host = Some("-oProxyCommand=touch /tmp/pwn".into());
        n.ssh_user = None;
        let v = ssh_argv(&n, &["list"]);
        let dd = v
            .iter()
            .position(|a| a == "--")
            .expect("`--` guard missing");
        assert_eq!(
            v[dd + 1],
            "-oProxyCommand=touch /tmp/pwn",
            "target must follow the guard: {v:?}"
        );
        // No option flags after the guard (everything past `--` is target+command).
        assert!(
            v[..dd].iter().all(|a| a != &v[dd + 1]),
            "target must not appear before the guard: {v:?}"
        );
    }

    /// Minimal POSIX-shell word splitter (whitespace + single quotes +
    /// backslash-escape outside quotes) - models how the remote login shell
    /// re-tokenizes the single string ssh sends it.
    fn remote_shell_split(s: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut in_q = false;
        let mut has = false;
        let mut it = s.chars();
        while let Some(c) = it.next() {
            match c {
                '\\' if !in_q => {
                    if let Some(n) = it.next() {
                        cur.push(n);
                        has = true;
                    }
                }
                '\'' => {
                    in_q = !in_q;
                    has = true;
                }
                ' ' if !in_q => {
                    if has {
                        out.push(std::mem::take(&mut cur));
                        has = false;
                    }
                }
                _ => {
                    cur.push(c);
                    has = true;
                }
            }
        }
        if has {
            out.push(cur);
        }
        out
    }

    #[test]
    fn ssh_argv_quotes_spaced_flags_for_the_remote_shell() {
        // The remote flag-save path: a spaced `--flags` value must arrive at the
        // remote clap as ONE argument, not be re-split into `-c` `32768` `-ngl`…
        let v = ssh_argv(
            &ssh_node(),
            &[
                "profile",
                "set-model",
                "m.gguf",
                "--flags",
                "-c 32768 -ngl 99",
            ],
        );
        // ssh joins the remote command with spaces; the remote shell re-splits it.
        let remote_cmd = v
            .iter()
            .skip_while(|s| *s != "llmtune")
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        let argv = remote_shell_split(&remote_cmd);
        assert_eq!(
            argv,
            vec![
                "llmtune",
                "--json",
                "node",
                "profile",
                "set-model",
                "m.gguf",
                "--flags",
                "-c 32768 -ngl 99",
            ]
        );
    }

    #[test]
    fn sh_quote_roundtrips_awkward_args() {
        // simple tokens pass through unquoted
        assert_eq!(sh_quote("--flags"), "--flags");
        assert_eq!(sh_quote("m.gguf"), "m.gguf");
        // spaces / metacharacters / embedded single quotes survive the remote
        // shell's re-split intact
        for raw in [
            "-c 32768 -ngl 99",
            "it's a name.gguf",
            "a;b&&c|d",
            "$HOME `id` \"x\"",
            "",
        ] {
            let joined = format!("llmtune {}", sh_quote(raw));
            let argv = remote_shell_split(&joined);
            let want: Vec<&str> = if raw.is_empty() {
                vec!["llmtune", ""]
            } else {
                vec!["llmtune", raw]
            };
            assert_eq!(argv, want, "round-trip of {raw:?}");
        }
    }

    #[test]
    fn model_info_parses_node_list_json() {
        let json = r#"[
          {"name":"Qwen3.5-9B-IQ2.gguf","arch":"qwen35","params":"9B","quant":"IQ2",
           "size_gib":6.5,"ctx_max":98304,"profile":"qwen35","used_default":false,"served":true}
        ]"#;
        let v: Vec<ModelInfo> = serde_json::from_str(json).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].name, "Qwen3.5-9B-IQ2.gguf");
        assert!(v[0].served);
        assert_eq!(v[0].profile, "qwen35");
    }

    #[test]
    fn ssh_argv_new_node_verbs() {
        // The transport's new ops reinvoke `llmtune --json node <op>` remotely.
        let tail_of = |args: &[&str]| -> Vec<String> {
            let v = ssh_argv(&ssh_node(), args);
            v.iter().skip_while(|s| *s != "llmtune").cloned().collect()
        };
        assert_eq!(
            tail_of(&["endpoint"]),
            vec!["llmtune", "--json", "node", "endpoint"]
        );
        assert_eq!(
            tail_of(&["doctor"]),
            vec!["llmtune", "--json", "node", "doctor"]
        );
        assert_eq!(tail_of(&["gpu"]), vec!["llmtune", "--json", "node", "gpu"]);
        assert_eq!(
            tail_of(&["profile", "set-model", "m.gguf", "--reset"]),
            vec![
                "llmtune",
                "--json",
                "node",
                "profile",
                "set-model",
                "m.gguf",
                "--reset"
            ]
        );
    }

    #[test]
    fn local_transport_implements_the_full_contract() {
        // LocalTransport must satisfy the WHOLE (extended) NodeTransport trait as
        // a trait object, and its node-scoped probes must run without panicking
        // on a non-BC-250 box (degrading, not failing).
        let node = Node {
            name: "localhost".into(),
            host: None,
            transport: Transport::Local,
            ssh_user: None,
            ssh_key: None,
            models_dir: "/nonexistent".into(),
            llama_unit: "llama-server.service".into(),
            llama_url: "http://127.0.0.1:1".into(),
            power_cmd: None,
        };
        let t: Box<dyn NodeTransport> = Box::new(LocalTransport { node });
        let _ = t.telemetry(); // Some on a GPU box, None elsewhere - both fine
        let checks = t.doctor().expect("doctor must run anywhere");
        assert!(!checks.is_empty(), "doctor returns checks");
        let ep = t.endpoint().expect("endpoint probe must not fail");
        assert!(ep.openai_base.ends_with("/v1"));
        assert!(!ep.healthy, "nothing listens on port 1");
    }

    #[test]
    fn telemetry_and_doctor_wire_shapes_parse() {
        // The exact JSON the node-side CLI emits must deserialize into the DTOs
        // the SSH transport parses.
        let t: Telemetry =
            serde_json::from_str(r#"{"gfxclk_mhz":1500,"uclk_mhz":450,"temp_c":61.5}"#).unwrap();
        assert_eq!(t.gfxclk_mhz, 1500);
        // an error object must NOT parse as telemetry (collapses to None)
        assert!(serde_json::from_str::<Telemetry>(r#"{"error":"unavailable"}"#).is_err());
        let checks: Vec<doctor::Check> = serde_json::from_str(
            r#"[{"label":"gpu","status":"ok","detail":"BC-250 present"},
                {"label":"unit","status":"fail","detail":"missing"}]"#,
        )
        .unwrap();
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].status, doctor::Status::Ok);
        assert_eq!(checks[1].status, doctor::Status::Fail);
        let ep: Endpoint = serde_json::from_str(
            r#"{"base_url":"http://x:8080","openai_base":"http://x:8080/v1","healthy":true}"#,
        )
        .unwrap();
        assert!(ep.healthy);
        assert!(ep.model.is_none() && ep.api_key.is_none());
    }

    #[test]
    fn build_version_wire_shape_parses() {
        // What `node build-version --json` emits must deserialize into the DTO
        // the SSH transport parses; null = build not installed on that node.
        let b: BuildVersion =
            serde_json::from_str(r#"{"name":"vulkan","version":"a1b2c3d4e5f6"}"#).unwrap();
        assert_eq!(b.version.as_deref(), Some("a1b2c3d4e5f6"));
        let b: BuildVersion = serde_json::from_str(r#"{"name":"vulkan","version":null}"#).unwrap();
        assert!(b.version.is_none());
    }

    #[test]
    fn ssh_argv_build_version_verb() {
        let v = ssh_argv(&ssh_node(), &["build-version", "vulkan"]);
        let tail: Vec<&str> = v.iter().rev().take(5).rev().map(|s| s.as_str()).collect();
        assert_eq!(
            tail,
            vec!["llmtune", "--json", "node", "build-version", "vulkan"]
        );
    }

    #[test]
    fn node_status_parses() {
        let json = r#"{"name":"bc250-1","reachable":true,"healthy":true,
          "served":"m.gguf","models":3,"last_model":"m.gguf","last_gen_tok_s":42.5}"#;
        let s: NodeStatus = serde_json::from_str(json).unwrap();
        assert!(s.healthy);
        assert_eq!(s.models, 3);
        assert_eq!(s.last_gen_tok_s, Some(42.5));
    }
}
