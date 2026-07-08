// SPDX-License-Identifier: GPL-2.0-only
//! The boot-server control plane: `netboot up/down/status`.
//!
//! `up` stands up a REACHABLE boot server, not just a running one: it starts
//! the artifact HTTP server + the NFS export of the model library (+ optional
//! proxyDHCP dnsmasq), OPENS the host firewall for exactly those ports (scoped
//! to the LAN CIDR, additive, never a flush), and then PROVES reachability
//! from OFF-HOST. The reference session lost hours to a closed ufw silently
//! dropping :8090 from the LAN - SYN dropped pre-accept, zero HTTP log, while
//! every localhost curl succeeded. This module makes that failure mode loud:
//! a localhost-only success is reported as UNPROVEN, and a local-ok/remote-dead
//! split is diagnosed as an interfering firewall.
//!
//! Everything `up` changes is RECORDED in a state file (`<work_dir>/up-state.toml`):
//! which firewall rules were added (vs already present), whether the exports
//! drop-in was created (vs a pre-existing hand-managed export), which services
//! were actually started/enabled (vs already running). `down` reverses exactly
//! that record and nothing else - a pre-existing rule, export, or service the
//! user already had is never touched.
//!
//! The generators/parsers are pure functions (unit-tested); the orchestrators
//! do the privileged IO via `swap::sudo`.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::config::{Config, Netboot, Transport};
use crate::netboot::HTTP_UNIT;

/// The exports drop-in this module owns (the MODEL LIBRARY export; the
/// diskless-root export from `netboot init` is a separate file).
pub const MODELS_EXPORTS: &str = "/etc/exports.d/llmtune-models.exports";
/// The nftables table this module owns (created/deleted whole; never touches
/// another table).
pub const NFT_TABLE: &str = "llmtune_netboot";
/// State file recording what `up` changed, so `down` reverses only that.
pub const STATE_FILE: &str = "up-state.toml";
/// Current state-file schema.
pub const STATE_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Ports: the exact surface the boot server needs open.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proto {
    Tcp,
    Udp,
}

impl Proto {
    pub fn as_str(&self) -> &'static str {
        match self {
            Proto::Tcp => "tcp",
            Proto::Udp => "udp",
        }
    }
}

/// One port the boot server needs open, with what it carries (for narration).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortRule {
    pub port: u16,
    pub proto: Proto,
    pub what: &'static str,
}

impl PortRule {
    /// The token recorded in the state file, e.g. `8090/tcp`.
    pub fn token(&self) -> String {
        format!("{}/{}", self.port, self.proto.as_str())
    }
}

/// The ports `up` opens: HTTP artifacts, NFS (nfsd + rpcbind), and - only when
/// dnsmasq is part of the stack - proxyDHCP + TFTP.
pub fn required_ports(nb: &Netboot) -> Vec<PortRule> {
    let mut v = vec![
        PortRule {
            port: nb.http_port,
            proto: Proto::Tcp,
            what: "http artifacts (boot.ipxe/vmlinuz/initrd)",
        },
        PortRule {
            port: 2049,
            proto: Proto::Tcp,
            what: "nfsd (model library)",
        },
        PortRule {
            port: 111,
            proto: Proto::Tcp,
            what: "rpcbind",
        },
        PortRule {
            port: 111,
            proto: Proto::Udp,
            what: "rpcbind",
        },
    ];
    if nb.dnsmasq {
        v.extend([
            PortRule {
                port: 67,
                proto: Proto::Udp,
                what: "proxyDHCP",
            },
            PortRule {
                port: 69,
                proto: Proto::Udp,
                what: "tftp (ipxe.efi)",
            },
            PortRule {
                port: 4011,
                proto: Proto::Udp,
                what: "proxyDHCP (PXE alt)",
            },
        ]);
    }
    v
}

// ---------------------------------------------------------------------------
// Firewall backends: pure rule builders + presence parsers per backend.
// Additive and LAN-scoped by construction; there is no "flush" path.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Ufw,
    Firewalld,
    Nftables,
    /// No active firewall found - nothing filters, nothing to open.
    None,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Ufw => "ufw",
            Backend::Firewalld => "firewalld",
            Backend::Nftables => "nftables",
            Backend::None => "none",
        }
    }
}

/// `ufw allow` argv for one scoped rule (comment marks it as llmtune's).
pub fn ufw_allow_argv(cidr: &str, r: &PortRule) -> Vec<String> {
    vec![
        "ufw".into(),
        "allow".into(),
        "from".into(),
        cidr.into(),
        "to".into(),
        "any".into(),
        "port".into(),
        r.port.to_string(),
        "proto".into(),
        r.proto.as_str().into(),
        "comment".into(),
        "llmtune-netboot".into(),
    ]
}

/// `ufw delete allow` argv - the exact spec `ufw_allow_argv` added.
pub fn ufw_delete_argv(cidr: &str, r: &PortRule) -> Vec<String> {
    let mut v = ufw_allow_argv(cidr, r);
    v.truncate(v.len() - 2); // ufw delete does not take the comment
    v.insert(1, "delete".into());
    v
}

/// Whether `ufw status` (verbose or not) reports "Status: active".
pub fn ufw_active(status: &str) -> bool {
    status
        .lines()
        .any(|l| l.trim().eq_ignore_ascii_case("status: active"))
}

/// Whether a `ufw status` listing already carries our scoped ALLOW rule. The
/// listing renders a from-scoped rule as `8090/tcp  ALLOW  192.168.1.0/24`; a
/// rule added without a proto covers BOTH protocols and renders as the bare
/// port (`111  ALLOW  192.168.1.0/24`) - that also satisfies us.
pub fn ufw_rule_present(status: &str, cidr: &str, r: &PortRule) -> bool {
    let with_proto = r.token();
    let bare = r.port.to_string();
    status.lines().any(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        let port_match =
            f.first() == Some(&with_proto.as_str()) || f.first() == Some(&bare.as_str());
        port_match && f.contains(&"ALLOW") && f.contains(&cidr)
    })
}

/// The firewalld rich rule for one scoped opening, in firewalld's own
/// canonical form (so presence checks against `--list-rich-rules` match).
pub fn firewalld_rich_rule(cidr: &str, r: &PortRule) -> String {
    format!(
        "rule family=\"ipv4\" source address=\"{cidr}\" port port=\"{}\" protocol=\"{}\" accept",
        r.port,
        r.proto.as_str()
    )
}

/// Whether `firewall-cmd --list-rich-rules` output already carries the rule.
pub fn firewalld_rule_present(list: &str, rule: &str) -> bool {
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let want = norm(rule);
    list.lines().any(|l| norm(l) == want)
}

/// The nft rule body for one opening, inside llmtune's OWN table (created and
/// deleted whole; other tables are never modified). Note the honest limit: an
/// accept here cannot override a drop verdict in another table's hook chain -
/// the post-`up` reachability check is what catches that.
pub fn nft_rule_needle(cidr: &str, r: &PortRule) -> String {
    format!(
        "ip saddr {cidr} {} dport {} accept",
        r.proto.as_str(),
        r.port
    )
}

/// `nft add rule` argv for one opening in llmtune's table.
pub fn nft_add_rule_argv(cidr: &str, r: &PortRule) -> Vec<String> {
    vec![
        "nft".into(),
        "add".into(),
        "rule".into(),
        "inet".into(),
        NFT_TABLE.into(),
        "input".into(),
        "ip".into(),
        "saddr".into(),
        cidr.into(),
        r.proto.as_str().into(),
        "dport".into(),
        r.port.to_string(),
        "accept".into(),
    ]
}

/// Whether `nft list table inet llmtune_netboot` already carries the rule
/// (whitespace-normalized substring match on the rule body).
pub fn nft_rule_present(listing: &str, cidr: &str, r: &PortRule) -> bool {
    let needle = nft_rule_needle(cidr, r);
    listing
        .lines()
        .any(|l| l.split_whitespace().collect::<Vec<_>>().join(" ") == needle)
}

// ---------------------------------------------------------------------------
// Endpoint exposure (`node expose` / boot-restore): open/close ONE unscoped
// tcp port on whatever backend is active. This reuses the same backend
// detection as `netboot up` - the old path was ufw-only, so on a firewalld or
// nftables box (Fedora default) "expose" reported success while the port
// stayed closed, with a wrong `sudo ufw allow` hint.
// ---------------------------------------------------------------------------

/// The nftables table endpoint exposure owns. Separate from [`NFT_TABLE`] so
/// closing exposure deletes exactly this table and never touches netboot's.
pub const NFT_EXPOSE_TABLE: &str = "llmtune_expose";

/// `ufw` argv to allow/delete one unscoped tcp port.
pub fn ufw_expose_argv(port: u16, open: bool) -> Vec<String> {
    let mut v: Vec<String> = vec!["ufw".into()];
    if !open {
        v.push("delete".into());
    }
    v.extend(["allow".into(), format!("{port}/tcp")]);
    v
}

/// `firewall-cmd --permanent` port argument for one unscoped tcp port.
pub fn firewalld_expose_arg(port: u16, open: bool) -> String {
    format!("--{}-port={port}/tcp", if open { "add" } else { "remove" })
}

/// The nft rule body for the exposure rule (inside [`NFT_EXPOSE_TABLE`]).
pub fn nft_expose_needle(port: u16) -> String {
    format!("tcp dport {port} accept")
}

/// Open (or close) `port`/tcp on the active firewall backend, for `node
/// expose` and boot-restore. Returns `(ok, backend)`; `Backend::None` means
/// nothing filters, which is trivially ok. Best-effort like the old ufw path:
/// a `false` is reported to the operator with a backend-accurate hint.
pub fn expose_port(port: u16, open: bool) -> (bool, Backend) {
    let backend = detect_backend(true);
    let ok = match backend {
        Backend::None => true,
        Backend::Ufw => {
            let argv = ufw_expose_argv(port, open);
            let argv_ref: Vec<&str> = argv.iter().map(String::as_str).collect();
            crate::swap::sudo(&argv_ref).is_ok()
        }
        Backend::Firewalld => {
            let arg = firewalld_expose_arg(port, open);
            crate::swap::sudo(&["firewall-cmd", "--permanent", &arg]).is_ok()
                && crate::swap::sudo(&["firewall-cmd", "--reload"]).is_ok()
        }
        Backend::Nftables => {
            if open {
                // Own-table pattern (like netboot's): created whole, deleted
                // whole; `nft add table/chain` are idempotent re-runs.
                let chain_spec = "{ type filter hook input priority -10 ; policy accept ; }";
                let needle = nft_expose_needle(port);
                let listing =
                    priv_output(&["nft", "list", "table", "inet", NFT_EXPOSE_TABLE], true)
                        .unwrap_or_default();
                let present = listing
                    .lines()
                    .any(|l| l.split_whitespace().collect::<Vec<_>>().join(" ") == needle);
                crate::swap::sudo(&["nft", "add", "table", "inet", NFT_EXPOSE_TABLE]).is_ok()
                    && crate::swap::sudo(&[
                        "nft",
                        "add",
                        "chain",
                        "inet",
                        NFT_EXPOSE_TABLE,
                        "input",
                        chain_spec,
                    ])
                    .is_ok()
                    && (present
                        || crate::swap::sudo(&[
                            "nft",
                            "add",
                            "rule",
                            "inet",
                            NFT_EXPOSE_TABLE,
                            "input",
                            "tcp",
                            "dport",
                            &port.to_string(),
                            "accept",
                        ])
                        .is_ok())
            } else {
                // Absent table = already closed.
                priv_output(&["nft", "list", "table", "inet", NFT_EXPOSE_TABLE], true).is_none()
                    || crate::swap::sudo(&["nft", "delete", "table", "inet", NFT_EXPOSE_TABLE])
                        .is_ok()
            }
        }
    };
    (ok, backend)
}

/// The command a user should run by hand when [`expose_port`] fails - accurate
/// per backend (the old hint said `sudo ufw allow ...` on every distro).
pub fn expose_hint(backend: Backend, port: u16, open: bool) -> String {
    match (backend, open) {
        (Backend::Ufw, true) => format!("sudo ufw allow {port}/tcp"),
        (Backend::Ufw, false) => format!("sudo ufw delete allow {port}/tcp"),
        (Backend::Firewalld, true) => format!(
            "sudo firewall-cmd --permanent --add-port={port}/tcp && sudo firewall-cmd --reload"
        ),
        (Backend::Firewalld, false) => format!(
            "sudo firewall-cmd --permanent --remove-port={port}/tcp && sudo firewall-cmd --reload"
        ),
        (Backend::Nftables, true) => format!(
            "sudo nft add table inet {NFT_EXPOSE_TABLE}; sudo nft add rule inet \
             {NFT_EXPOSE_TABLE} input tcp dport {port} accept"
        ),
        (Backend::Nftables, false) => {
            format!("sudo nft delete table inet {NFT_EXPOSE_TABLE}")
        }
        (Backend::None, _) => "no active firewall was detected - nothing to change".to_string(),
    }
}

// ---------------------------------------------------------------------------
// NFS exports: one line, one llmtune-owned drop-in file.
// ---------------------------------------------------------------------------

/// The exports line for the model library: read-only, root-squashed, scoped to
/// the LAN CIDR (design of record 3.3).
pub fn exports_line(models_dir: &str, cidr: &str) -> String {
    format!("{models_dir} {cidr}(ro,sync,no_subtree_check,root_squash)")
}

/// The full contents of llmtune's exports drop-in.
pub fn exports_file(models_dir: &str, cidr: &str) -> String {
    format!(
        "# AUTO-GENERATED by llmtune (`llmtune netboot up`). Removed by\n\
         # `llmtune netboot down`. Edit fleet.toml, not this file.\n\
         {}\n",
        exports_line(models_dir, cidr)
    )
}

/// Whether an exports corpus (any /etc/exports + drop-ins text) already
/// exports `models_dir` to `cidr` - regardless of options, so a hand-managed
/// export with different flags is respected, not duplicated or clobbered.
pub fn exports_present(corpus: &str, models_dir: &str, cidr: &str) -> bool {
    let client_prefix = format!("{cidr}(");
    corpus
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .any(|l| {
            let mut f = l.split_whitespace();
            f.next() == Some(models_dir) && f.any(|c| c == cidr || c.starts_with(&client_prefix))
        })
}

// ---------------------------------------------------------------------------
// State: the exact record of what `up` changed, so `down` reverses only that.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpState {
    pub version: u32,
    /// The backend rules were added to ("none" = nothing was filtering).
    pub firewall_backend: String,
    /// Rule tokens (`8090/tcp`) `up` ADDED - rules that already existed are
    /// not listed and therefore never removed.
    pub fw_added: Vec<String>,
    /// `up` created llmtune's whole nft table (down deletes the table).
    pub nft_table_added: bool,
    /// `up` wrote the models exports drop-in (a pre-existing hand-managed
    /// export means false: down leaves exports alone).
    pub exports_file_added: bool,
    /// Services `up` actually started (they were inactive before).
    pub started: Vec<String>,
    /// Services `up` actually enabled (they were disabled before).
    pub enabled: Vec<String>,
}

impl UpState {
    /// Fold a previous `up`'s record into this one. A re-run of `up` sees its
    /// own earlier additions as "already present" and would otherwise record
    /// nothing - then `down` after `up; up` would leave them behind. The
    /// union keeps ownership of everything ANY `up` added.
    pub fn merge_prior(&mut self, prior: &UpState) {
        for t in &prior.fw_added {
            if !self.fw_added.contains(t) {
                self.fw_added.push(t.clone());
            }
        }
        for s in &prior.started {
            if !self.started.contains(s) {
                self.started.push(s.clone());
            }
        }
        for s in &prior.enabled {
            if !self.enabled.contains(s) {
                self.enabled.push(s.clone());
            }
        }
        self.nft_table_added |= prior.nft_table_added;
        self.exports_file_added |= prior.exports_file_added;
        if self.firewall_backend.is_empty() {
            self.firewall_backend = prior.firewall_backend.clone();
        }
    }

    /// True when nothing is owned - `down` may delete the state file only
    /// then; otherwise the residual record must be kept so a re-run can retry
    /// (a deleted record makes a still-live rule unreachable forever).
    pub fn owns_nothing(&self) -> bool {
        self.fw_added.is_empty()
            && !self.nft_table_added
            && !self.exports_file_added
            && self.started.is_empty()
            && self.enabled.is_empty()
    }
}

/// Parse a state-file rule token back into (port, proto).
pub fn parse_rule_token(tok: &str) -> Option<PortRule> {
    let (p, proto) = tok.split_once('/')?;
    let port = p.parse().ok()?;
    let proto = match proto {
        "tcp" => Proto::Tcp,
        "udp" => Proto::Udp,
        _ => return None,
    };
    Some(PortRule {
        port,
        proto,
        what: "",
    })
}

fn state_path(nb: &Netboot) -> PathBuf {
    Path::new(&nb.work_dir).join(STATE_FILE)
}

fn load_state(nb: &Netboot) -> Option<UpState> {
    let text = std::fs::read_to_string(state_path(nb)).ok()?;
    toml::from_str(&text).ok()
}

fn save_state(nb: &Netboot, st: &UpState) -> Result<()> {
    let text = toml::to_string_pretty(st).context("serializing up-state")?;
    crate::swap::sudo(&["mkdir", "-p", &nb.work_dir])?;
    crate::swap::sudo_tee(&state_path(nb), &text)
        .with_context(|| format!("writing {}", state_path(nb).display()))
}

// ---------------------------------------------------------------------------
// Reachability: local bind checks + an OFF-HOST probe, and the diagnosis.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortCheck {
    pub port: u16,
    pub ok: bool,
}

/// The reachability evidence: local TCP connects to the server's LAN IP, plus
/// (when a probe host is available) the same connects run FROM ANOTHER HOST.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReachReport {
    pub local: Vec<PortCheck>,
    pub remote_host: Option<String>,
    pub remote: Vec<PortCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Proven reachable from off-host.
    Ok,
    /// Only local evidence - explicitly NOT proof (localhost bypasses the
    /// LAN-facing firewall path; the exact trap from the reference session).
    LocalOnlyUnproven,
    /// Listening locally but dead from the LAN: a firewall is eating the SYNs.
    Interference(Vec<u16>),
    /// Not even listening locally - a service problem, not a firewall one.
    NotListening(Vec<u16>),
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Ok => "ok",
            Verdict::LocalOnlyUnproven => "local-only-unproven",
            Verdict::Interference(_) => "interfering-firewall",
            Verdict::NotListening(_) => "not-listening",
        }
    }
}

/// Turn the evidence into a crisp verdict + a one-look diagnosis. Pure.
pub fn diagnose(r: &ReachReport, backend: Backend) -> (Verdict, String) {
    let dead_local: Vec<u16> = r.local.iter().filter(|c| !c.ok).map(|c| c.port).collect();
    if !dead_local.is_empty() {
        let msg = format!(
            "not listening on {} even locally - a service is down (check \
             `systemctl status {HTTP_UNIT} nfs-server`), not a firewall problem",
            fmt_ports(&dead_local)
        );
        return (Verdict::NotListening(dead_local), msg);
    }
    if r.remote.is_empty() {
        let msg = "listening locally, but NO off-host proof: localhost reachability \
                   does NOT prove LAN reachability (a firewall drops LAN SYNs \
                   pre-accept with zero log while local curl works). Register an \
                   ssh node (`netboot nodes --register`) or probe from another \
                   host: curl http://<server_ip>:<port>/boot.ipxe"
            .to_string();
        return (Verdict::LocalOnlyUnproven, msg);
    }
    let blocked: Vec<u16> = r.remote.iter().filter(|c| !c.ok).map(|c| c.port).collect();
    if !blocked.is_empty() {
        let host = r.remote_host.as_deref().unwrap_or("?");
        let hint = match backend {
            Backend::Ufw => "check `ufw status` for the llmtune-netboot rules",
            Backend::Firewalld => "check `firewall-cmd --list-rich-rules`",
            Backend::Nftables => {
                "another nft table's hook chain is likely dropping - an accept in \
                 llmtune's table cannot override a drop elsewhere; inspect `nft list ruleset`"
            }
            Backend::None => {
                "no firewall backend was detected, yet the LAN cannot connect - \
                 check for an unmanaged filter (iptables? a NIC-level ACL?)"
            }
        };
        let msg = format!(
            "INTERFERING FIREWALL: listening locally but {} unreachable from {host} \
             (SYN dropped pre-accept - this never shows in the HTTP log); {hint}",
            fmt_ports(&blocked)
        );
        return (Verdict::Interference(blocked), msg);
    }
    let host = r.remote_host.as_deref().unwrap_or("?");
    (
        Verdict::Ok,
        format!(
            "reachable from {host}: {} all open off-host",
            fmt_ports(&r.remote.iter().map(|c| c.port).collect::<Vec<_>>())
        ),
    )
}

fn fmt_ports(ports: &[u16]) -> String {
    ports
        .iter()
        .map(|p| format!(":{p}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The remote probe script: bash /dev/tcp connects per port, one `<port>
/// open|closed` line each (no curl/nc dependency on the probe host).
pub fn probe_script(server_ip: &str, ports: &[u16]) -> String {
    let list = ports
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "for p in {list}; do (timeout 5 bash -c \"exec 3<>/dev/tcp/{server_ip}/$p\") \
         >/dev/null 2>&1 && echo \"$p open\" || echo \"$p closed\"; done"
    )
}

/// Parse the probe script's output into per-port results. Unknown lines are
/// skipped; an empty result means the probe itself failed (no evidence).
pub fn parse_probe_output(out: &str) -> Vec<PortCheck> {
    out.lines()
        .filter_map(|l| {
            let (p, verdict) = l.trim().split_once(' ')?;
            let port = p.parse().ok()?;
            match verdict {
                "open" => Some(PortCheck { port, ok: true }),
                "closed" => Some(PortCheck { port, ok: false }),
                _ => None,
            }
        })
        .collect()
}

/// Pick the off-host probe: the first configured ssh node whose host is not
/// this server. Returns (display name, host, ssh_user, ssh_key).
pub fn pick_probe_node(cfg: &Config, nb: &Netboot) -> Option<crate::config::Node> {
    cfg.nodes
        .iter()
        .find(|n| {
            n.transport == Transport::Ssh
                && n.host.as_deref().is_some_and(|h| {
                    !h.is_empty() && h != nb.server_ip && h != "localhost" && h != "127.0.0.1"
                })
        })
        .cloned()
}

/// Run the off-host probe over ssh. Empty result = no evidence (probe host
/// down, no bash, etc.) - never counted as "closed".
fn remote_probe(node: &crate::config::Node, server_ip: &str, ports: &[u16]) -> Vec<PortCheck> {
    let script = probe_script(server_ip, ports);
    let mut argv: Vec<String> = vec![
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "ConnectTimeout=8".into(),
        "-o".into(),
        format!(
            "StrictHostKeyChecking={}",
            crate::transport::hostkey_policy()
        ),
    ];
    if let Some(k) = &node.ssh_key {
        argv.push("-i".into());
        argv.push(k.clone());
    }
    let host = node.host.clone().unwrap_or_default();
    argv.push(match &node.ssh_user {
        Some(u) => format!("{u}@{host}"),
        None => host,
    });
    argv.push("sh".into());
    argv.push("-c".into());
    argv.push(crate::transport::sh_quote(&script));
    match std::process::Command::new("ssh").args(&argv).output() {
        Ok(o) if o.status.success() => parse_probe_output(&String::from_utf8_lossy(&o.stdout)),
        _ => Vec::new(),
    }
}

/// Local TCP connect checks against the server's LAN IP (catches a dead or
/// localhost-bound service; can NOT catch a LAN-facing firewall drop).
fn local_checks(server_ip: &str, ports: &[u16]) -> Vec<PortCheck> {
    ports
        .iter()
        .map(|&port| {
            let ok = format!("{server_ip}:{port}")
                .parse()
                .ok()
                .and_then(|addr: std::net::SocketAddr| {
                    std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(2))
                        .ok()
                })
                .is_some();
            PortCheck { port, ok }
        })
        .collect()
}

/// The ports whose reachability is REQUIRED for a working boot: the HTTP
/// artifact server and nfsd. Deliberately NOT :111 - an NFSv4-only server
/// (the netboot image mounts nfs4) has no rpcbind listening, and that is
/// fine; the firewall still opens it for v3 clients.
pub fn reach_ports(nb: &Netboot) -> Vec<u16> {
    vec![nb.http_port, 2049]
}

/// The full reachability check: local binds + the off-host probe when a probe
/// node is configured. Probes only the essential TCP ports (UDP has no
/// connect proof; see [`reach_ports`]).
pub fn reach_check(cfg: &Config, nb: &Netboot) -> ReachReport {
    let tcp_ports: Vec<u16> = reach_ports(nb);
    let local = local_checks(&nb.server_ip, &tcp_ports);
    let (remote_host, remote) = match pick_probe_node(cfg, nb) {
        Some(node) => {
            let host = node.host.clone().unwrap_or_default();
            let res = remote_probe(&node, &nb.server_ip, &tcp_ports);
            (Some(format!("{} ({host})", node.name)), res)
        }
        None => (None, Vec::new()),
    };
    ReachReport {
        local,
        remote_host,
        remote,
    }
}

// ---------------------------------------------------------------------------
// Privileged IO helpers.
// ---------------------------------------------------------------------------

/// Whether `name` resolves on PATH or in the sbin dirs (services like nfs's
/// exportfs live in sbin, which a user PATH often lacks).
fn have_cmd(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        for d in path.split(':') {
            if !d.is_empty() && Path::new(d).join(name).exists() {
                return true;
            }
        }
    }
    ["/usr/sbin", "/sbin", "/usr/local/sbin", "/usr/bin"]
        .iter()
        .any(|d| Path::new(d).join(name).exists())
}

/// Run a privileged command capturing stdout. `interactive` allows a sudo
/// password prompt (actuation paths); non-interactive (`sudo -n`) is for
/// read-only status queries so `netboot status` never hangs on a prompt.
fn priv_output(args: &[&str], interactive: bool) -> Option<String> {
    let mut cmd = if crate::swap::is_root() {
        let mut c = std::process::Command::new(args[0]);
        c.args(&args[1..]);
        c
    } else {
        let mut c = std::process::Command::new("sudo");
        if !interactive {
            c.arg("-n");
        }
        c.args(args);
        c
    };
    let out = cmd.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

fn is_active(svc: &str) -> String {
    std::process::Command::new("systemctl")
        .args(["is-active", svc])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn is_enabled(svc: &str) -> String {
    std::process::Command::new("systemctl")
        .args(["is-enabled", svc])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

/// The unit's fragment path ("" = unit unknown to systemd).
fn unit_fragment(svc: &str) -> String {
    std::process::Command::new("systemctl")
        .args(["show", "-p", "FragmentPath", "--value", svc])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The stack this config runs: HTTP always, NFS always, dnsmasq when enabled.
fn stack(nb: &Netboot) -> Vec<&'static str> {
    let mut v = vec![HTTP_UNIT, "nfs-server"];
    if nb.dnsmasq {
        v.push("dnsmasq");
    }
    v
}

/// Detect the active firewall backend. Order matters: ufw and firewalld both
/// sit on nftables underneath, so they are checked first.
fn detect_backend(interactive: bool) -> Backend {
    if have_cmd("ufw") {
        if let Some(out) = priv_output(&["ufw", "status"], interactive) {
            if ufw_active(&out) {
                return Backend::Ufw;
            }
            // ufw installed but inactive: it is not filtering; fall through.
        }
    }
    if have_cmd("firewall-cmd") {
        if let Some(out) = priv_output(&["firewall-cmd", "--state"], interactive) {
            if out.trim() == "running" {
                return Backend::Firewalld;
            }
        }
    }
    if have_cmd("nft") {
        if let Some(out) = priv_output(&["nft", "list", "tables"], interactive) {
            if !out.trim().is_empty() {
                return Backend::Nftables;
            }
        }
    }
    Backend::None
}

/// The exports corpus: /etc/exports + every /etc/exports.d/*.exports,
/// optionally excluding llmtune's own drop-in (for the "already hand-managed
/// elsewhere?" check).
fn read_exports_corpus(exclude_own: bool) -> String {
    let mut s = std::fs::read_to_string("/etc/exports").unwrap_or_default();
    if let Ok(rd) = std::fs::read_dir("/etc/exports.d") {
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("exports"))
            .filter(|p| !(exclude_own && p == Path::new(MODELS_EXPORTS)))
            .collect();
        files.sort();
        for p in files {
            s.push('\n');
            s.push_str(&std::fs::read_to_string(&p).unwrap_or_default());
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Orchestrators: up / down / status.
// ---------------------------------------------------------------------------

pub struct UpOpts {
    /// Skip firewall management entirely.
    pub no_firewall: bool,
    /// Skip the post-up reachability self-check.
    pub skip_check: bool,
    pub json: bool,
}

/// Progress line for an orchestrator: humans read stdout; under `--json`
/// stdout is data-only (one JSON document), so progress goes to stderr.
fn say(json: bool, msg: &str) {
    if json {
        eprintln!("{msg}");
    } else {
        println!("{msg}");
    }
}

/// The `netboot up --json` summary (ONE stdout object). `reachability` is
/// `null` when the self-check was skipped. Pure for the unit test.
fn up_summary_json(st: &UpState, reachability: Option<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "up": true,
        "services_started": st.started,
        "services_enabled": st.enabled,
        "exports_file_added": st.exports_file_added,
        "firewall": {
            "backend": st.firewall_backend,
            "rules_added": st.fw_added,
        },
        "reachability": reachability,
    })
}

/// `netboot up`: services + exports + firewall + off-host reachability proof.
pub fn up(cfg: &Config, opts: &UpOpts) -> Result<()> {
    let nb = crate::netboot::require(cfg)?;
    let mut st = UpState {
        version: STATE_VERSION,
        ..Default::default()
    };
    // Fold a prior `up`'s record in FIRST (not after the phases): every
    // checkpoint below must already own what earlier runs added, so a
    // mid-phase failure can never orphan the prior record.
    if let Some(prior) = load_state(nb) {
        st.merge_prior(&prior);
    }
    // Checkpoint the record after EVERY mutation. `up` used to persist once,
    // after all three mutating phases - so a failure on (say) the 4th
    // firewall rule exited with rules/exports/services live and NO record,
    // and the conservative no-record `down` could never reverse them (the
    // confirmed M3 state-leak). A checkpoint failure is loud but non-fatal:
    // the final save below is the error-checked one.
    let checkpoint = |st: &UpState| {
        if let Err(e) = save_state(nb, st) {
            eprintln!(
                "WARNING: could not checkpoint up-state ({e:#}) - if this run \
                 fails before its final save, `netboot down` may not know \
                 about the additions made so far"
            );
        }
    };

    // Preflight: the pieces `up` orchestrates must exist before it starts
    // half a stack.
    if unit_fragment(HTTP_UNIT).is_empty() {
        bail!("{HTTP_UNIT} is not installed - run `llmtune netboot init --apply` first");
    }
    if !have_cmd("exportfs") {
        bail!(
            "nfs-utils is not installed (no exportfs found) - install it \
             (arch: pacman -S nfs-utils; debian: apt install nfs-kernel-server) \
             and re-run `llmtune netboot up`"
        );
    }
    if nb.dnsmasq && !have_cmd("dnsmasq") {
        bail!("[netboot] dnsmasq = true but dnsmasq is not installed");
    }

    // 1. Services: record what was ALREADY running/enabled so `down` only
    // reverses what this `up` changed.
    for svc in stack(nb) {
        let was_active = is_active(svc) == "active";
        let was_enabled = is_enabled(svc) == "enabled";
        crate::swap::sudo(&["systemctl", "enable", "--now", svc])
            .with_context(|| format!("starting {svc}"))?;
        if !was_active {
            st.started.push(svc.to_string());
        }
        if !was_enabled {
            st.enabled.push(svc.to_string());
        }
        if !was_active || !was_enabled {
            checkpoint(&st);
        }
        say(
            opts.json,
            &format!(
                "[ok] {svc} {}",
                if was_active {
                    "already active"
                } else {
                    "started"
                }
            ),
        );
    }

    // 2. NFS export of the model library. If the user already exports it to
    // this LAN (any options), respect that and add nothing.
    crate::swap::sudo(&["mkdir", "-p", &nb.models_dir])?;
    if exports_present(&read_exports_corpus(true), &nb.models_dir, &nb.lan_cidr) {
        say(
            opts.json,
            &format!(
                "[ok] {} already exported to {} (pre-existing; left as-is)",
                nb.models_dir, nb.lan_cidr
            ),
        );
    } else {
        crate::swap::sudo(&["mkdir", "-p", "/etc/exports.d"])?;
        crate::swap::sudo_tee(
            Path::new(MODELS_EXPORTS),
            &exports_file(&nb.models_dir, &nb.lan_cidr),
        )
        .with_context(|| format!("writing {MODELS_EXPORTS}"))?;
        st.exports_file_added = true;
        checkpoint(&st);
        say(
            opts.json,
            &format!(
                "[ok] exported {} ro to {} ({MODELS_EXPORTS})",
                nb.models_dir, nb.lan_cidr
            ),
        );
    }
    crate::swap::sudo(&["exportfs", "-ra"]).context("exportfs -ra")?;

    // 3. Firewall: open exactly the stack's ports, scoped to the LAN CIDR,
    // additive only. Rules that already exist are recorded as NOT ours.
    if opts.no_firewall {
        say(
            opts.json,
            "[..] firewall management skipped (--no-firewall)",
        );
    } else {
        let backend = detect_backend(true);
        st.firewall_backend = backend.as_str().to_string();
        open_firewall(nb, backend, &mut st, opts.json, &checkpoint)?;
    }

    // The authoritative save (the per-mutation checkpoints above are
    // best-effort): persisted BEFORE the self-check so a failed check still
    // leaves `down` able to reverse everything.
    save_state(nb, &st)?;

    // 4. Reachability: prove it from OFF-HOST, or say loudly that we could not.
    if opts.skip_check {
        say(
            opts.json,
            "[..] reachability self-check skipped (--skip-check)",
        );
        if opts.json {
            println!("{}", up_summary_json(&st, None));
        }
        return Ok(());
    }
    let report = reach_check(cfg, nb);
    let backend = if opts.no_firewall {
        Backend::None
    } else {
        st.firewall_backend.parse_backend().unwrap_or(Backend::None)
    };
    let (verdict, msg) = diagnose(&report, backend);
    if opts.json {
        println!(
            "{}",
            up_summary_json(&st, Some(reach_json_value(&report, &verdict, &msg)))
        );
    } else {
        print_reach(&report, &verdict, &msg);
    }
    match verdict {
        Verdict::Ok | Verdict::LocalOnlyUnproven => Ok(()),
        Verdict::Interference(_) | Verdict::NotListening(_) => {
            // A refusal, not an error: the stack is up but unproven from
            // off-host (an interfering firewall / nothing listening); `down`
            // reverses it, --skip-check overrides. Exit code 2.
            Err(crate::agentic::refusal(format!(
                "netboot up: reachability self-check failed - {msg} \
                 (fix the interference and re-run, or --skip-check to accept unproven)"
            )))
        }
    }
}

trait ParseBackend {
    fn parse_backend(&self) -> Option<Backend>;
}
impl ParseBackend for String {
    fn parse_backend(&self) -> Option<Backend> {
        match self.as_str() {
            "ufw" => Some(Backend::Ufw),
            "firewalld" => Some(Backend::Firewalld),
            "nftables" => Some(Backend::Nftables),
            "none" => Some(Backend::None),
            _ => None,
        }
    }
}

/// Open the stack's ports on the detected backend; record only what was added.
/// `checkpoint` persists the record after each addition, so an abort between
/// rules can never leave an unrecorded (irreversible) opening.
fn open_firewall(
    nb: &Netboot,
    backend: Backend,
    st: &mut UpState,
    json: bool,
    checkpoint: &dyn Fn(&UpState),
) -> Result<()> {
    let rules = required_ports(nb);
    let cidr = &nb.lan_cidr;
    match backend {
        Backend::None => {
            say(
                json,
                "[ok] no active firewall detected - nothing filters, nothing to open \
                 (the self-check still verifies off-host)",
            );
        }
        Backend::Ufw => {
            let status = priv_output(&["ufw", "status"], true).unwrap_or_default();
            for r in &rules {
                if ufw_rule_present(&status, cidr, r) {
                    say(
                        json,
                        &format!("[ok] ufw: {} from {cidr} already allowed", r.token()),
                    );
                    continue;
                }
                let argv = ufw_allow_argv(cidr, r);
                let argv_ref: Vec<&str> = argv.iter().map(String::as_str).collect();
                crate::swap::sudo(&argv_ref).with_context(|| format!("ufw allow {}", r.token()))?;
                st.fw_added.push(r.token());
                checkpoint(st);
                say(
                    json,
                    &format!("[ok] ufw: allowed {} from {cidr} ({})", r.token(), r.what),
                );
            }
        }
        Backend::Firewalld => {
            let list =
                priv_output(&["firewall-cmd", "--list-rich-rules"], true).unwrap_or_default();
            let mut changed = false;
            for r in &rules {
                let rule = firewalld_rich_rule(cidr, r);
                if firewalld_rule_present(&list, &rule) {
                    say(
                        json,
                        &format!("[ok] firewalld: {} already allowed", r.token()),
                    );
                    continue;
                }
                let arg = format!("--add-rich-rule={rule}");
                crate::swap::sudo(&["firewall-cmd", "--permanent", &arg])
                    .with_context(|| format!("firewalld add {}", r.token()))?;
                st.fw_added.push(r.token());
                checkpoint(st);
                changed = true;
                say(
                    json,
                    &format!(
                        "[ok] firewalld: allowed {} from {cidr} ({})",
                        r.token(),
                        r.what
                    ),
                );
            }
            if changed {
                crate::swap::sudo(&["firewall-cmd", "--reload"])
                    .context("firewall-cmd --reload")?;
            }
        }
        Backend::Nftables => {
            // llmtune's OWN table: created/deleted whole, other tables never
            // touched (and never flushed).
            let tables = priv_output(&["nft", "list", "tables"], true).unwrap_or_default();
            let table_existed = tables
                .lines()
                .any(|l| l.split_whitespace().collect::<Vec<_>>() == ["table", "inet", NFT_TABLE]);
            if !table_existed {
                crate::swap::sudo(&["nft", "add", "table", "inet", NFT_TABLE])?;
                let chain_spec = "{ type filter hook input priority -10 ; policy accept ; }";
                crate::swap::sudo(&[
                    "nft", "add", "chain", "inet", NFT_TABLE, "input", chain_spec,
                ])?;
                st.nft_table_added = true;
                checkpoint(st);
            }
            let listing =
                priv_output(&["nft", "list", "table", "inet", NFT_TABLE], true).unwrap_or_default();
            for r in &rules {
                if nft_rule_present(&listing, cidr, r) {
                    say(
                        json,
                        &format!("[ok] nft: {} from {cidr} already accepted", r.token()),
                    );
                    continue;
                }
                let argv = nft_add_rule_argv(cidr, r);
                let argv_ref: Vec<&str> = argv.iter().map(String::as_str).collect();
                crate::swap::sudo(&argv_ref).with_context(|| format!("nft add {}", r.token()))?;
                st.fw_added.push(r.token());
                checkpoint(st);
                say(
                    json,
                    &format!("[ok] nft: accepted {} from {cidr} ({})", r.token(), r.what),
                );
            }
            say(
                json,
                &format!(
                    "     note: rules live in llmtune's own table ({NFT_TABLE}); an accept \
                     here cannot override a drop in another table - the self-check verifies"
                ),
            );
        }
    }
    Ok(())
}

/// `netboot down`: stop only what `up` started; remove only what `up` added.
/// A removal that FAILS keeps its entry in the state file (rewritten as the
/// residual record) and fails the command - printing "[ok] removed" while a
/// rule was still live, then unconditionally deleting the only record of it,
/// was the terminal half of the M3 state-leak.
pub fn down(cfg: &Config, json: bool) -> Result<()> {
    let nb = crate::netboot::require(cfg)?;
    let st = load_state(nb);
    let mut undone: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();

    match &st {
        Some(st) => {
            // Everything whose removal fails stays owned here; the state file
            // is rewritten (not deleted) so a re-run can retry exactly that.
            let mut residual = UpState {
                version: STATE_VERSION,
                firewall_backend: st.firewall_backend.clone(),
                ..Default::default()
            };
            // Firewall rules first (they reference services' ports, order is
            // cosmetic - but do it while the record is in hand).
            if let Some(backend) = st.firewall_backend.parse_backend() {
                remove_firewall(nb, backend, st, &mut undone, &mut residual, &mut failed);
            }
            // Exports: only the drop-in this tool created.
            if st.exports_file_added && Path::new(MODELS_EXPORTS).exists() {
                let r = crate::swap::sudo(&["rm", "-f", MODELS_EXPORTS])
                    .and_then(|()| crate::swap::sudo(&["exportfs", "-ra"]));
                match r {
                    Ok(()) => undone.push(format!("exports drop-in {MODELS_EXPORTS}")),
                    Err(e) => {
                        residual.exports_file_added = true;
                        failed.push(format!("exports drop-in {MODELS_EXPORTS}: {e:#}"));
                    }
                }
            } else if !st.exports_file_added {
                skipped.push("exports (pre-existing, not ours)".into());
            }
            // Services: stop/disable only what `up` started/enabled.
            for svc in st.started.iter().rev() {
                match crate::swap::sudo(&["systemctl", "stop", svc]) {
                    Ok(()) => undone.push(format!("stopped {svc}")),
                    Err(e) => {
                        residual.started.push(svc.clone());
                        failed.push(format!("stop {svc}: {e:#}"));
                    }
                }
            }
            for svc in &st.enabled {
                if crate::swap::sudo(&["systemctl", "disable", svc]).is_err() {
                    residual.enabled.push(svc.clone());
                    failed.push(format!("disable {svc}"));
                }
            }
            for svc in stack(nb) {
                if !st.started.iter().any(|s| s == svc) {
                    skipped.push(format!("{svc} (was already running before up)"));
                }
            }
            if residual.owns_nothing() {
                crate::swap::sudo(&["rm", "-f", &state_path(nb).to_string_lossy()])?;
            } else {
                save_state(nb, &residual)?;
            }
        }
        None => {
            // No record: be conservative. Only artifacts that are
            // unambiguously llmtune's (our exports drop-in, our nft table, our
            // own HTTP unit) are removed; shared services and shared-backend
            // firewall rules are left alone.
            say(
                json,
                &format!(
                    "[..] no up-state record ({}) - conservative down: only \
                     unambiguously-llmtune artifacts are removed",
                    state_path(nb).display()
                ),
            );
            let _ = crate::swap::sudo(&["systemctl", "disable", "--now", HTTP_UNIT]);
            undone.push(format!("stopped {HTTP_UNIT}"));
            if Path::new(MODELS_EXPORTS).exists() {
                crate::swap::sudo(&["rm", "-f", MODELS_EXPORTS])?;
                crate::swap::sudo(&["exportfs", "-ra"])?;
                undone.push(format!("exports drop-in {MODELS_EXPORTS}"));
            }
            if priv_output(&["nft", "list", "table", "inet", NFT_TABLE], true).is_some() {
                crate::swap::sudo(&["nft", "delete", "table", "inet", NFT_TABLE])?;
                undone.push(format!("nft table {NFT_TABLE}"));
            }
            skipped.push("nfs-server/dnsmasq (unknown prior state)".into());
            skipped.push("ufw/firewalld rules (no record of which are ours)".into());
        }
    }

    if json {
        println!(
            "{}",
            serde_json::json!({
                "down": failed.is_empty(),
                "undone": undone,
                "left_alone": skipped,
                "failed": failed,
            })
        );
    } else {
        for u in &undone {
            println!("[ok] removed: {u}");
        }
        for s in &skipped {
            println!("[..] left alone: {s}");
        }
        for f in &failed {
            println!("[fail] still live: {f}");
        }
        if failed.is_empty() {
            println!("[ok] netboot stack down");
        }
    }
    if !failed.is_empty() {
        bail!(
            "netboot down: {} removal(s) FAILED and remain recorded in {} - re-run `llmtune netboot down` to retry them",
            failed.len(),
            state_path(nb).display()
        );
    }
    Ok(())
}

/// Remove exactly the rules `up` recorded as added. Failed removals go to
/// `residual`/`failed` instead of being swallowed - the rule is still open,
/// so its ownership record must survive.
fn remove_firewall(
    nb: &Netboot,
    backend: Backend,
    st: &UpState,
    undone: &mut Vec<String>,
    residual: &mut UpState,
    failed: &mut Vec<String>,
) {
    let cidr = &nb.lan_cidr;
    match backend {
        Backend::None => {}
        Backend::Ufw => {
            for tok in &st.fw_added {
                let Some(r) = parse_rule_token(tok) else {
                    // A token we cannot parse we cannot remove - keep it.
                    residual.fw_added.push(tok.clone());
                    failed.push(format!("ufw rule {tok} (unparseable token)"));
                    continue;
                };
                let argv = ufw_delete_argv(cidr, &r);
                let argv_ref: Vec<&str> = argv.iter().map(String::as_str).collect();
                match crate::swap::sudo(&argv_ref) {
                    Ok(()) => undone.push(format!("ufw rule {tok} from {cidr}")),
                    Err(e) => {
                        residual.fw_added.push(tok.clone());
                        failed.push(format!("ufw rule {tok}: {e:#}"));
                    }
                }
            }
        }
        Backend::Firewalld => {
            let mut changed = false;
            for tok in &st.fw_added {
                let Some(r) = parse_rule_token(tok) else {
                    residual.fw_added.push(tok.clone());
                    failed.push(format!("firewalld rule {tok} (unparseable token)"));
                    continue;
                };
                let arg = format!("--remove-rich-rule={}", firewalld_rich_rule(cidr, &r));
                match crate::swap::sudo(&["firewall-cmd", "--permanent", &arg]) {
                    Ok(()) => {
                        undone.push(format!("firewalld rule {tok} from {cidr}"));
                        changed = true;
                    }
                    Err(e) => {
                        residual.fw_added.push(tok.clone());
                        failed.push(format!("firewalld rule {tok}: {e:#}"));
                    }
                }
            }
            if changed && crate::swap::sudo(&["firewall-cmd", "--reload"]).is_err() {
                // The permanent config no longer carries the rules; only the
                // runtime reload failed. Not kept in the residual (a retry of
                // --remove would fail on the already-removed rules) - report
                // it so the operator reloads by hand.
                failed.push(
                    "firewall-cmd --reload (permanent rules removed; \
                             runtime not reloaded - run `sudo firewall-cmd --reload`)"
                        .to_string(),
                );
            }
        }
        Backend::Nftables => {
            if st.nft_table_added {
                match crate::swap::sudo(&["nft", "delete", "table", "inet", NFT_TABLE]) {
                    Ok(()) => undone.push(format!("nft table {NFT_TABLE} (whole table was ours)")),
                    Err(e) => {
                        residual.nft_table_added = true;
                        residual.fw_added.extend(st.fw_added.iter().cloned());
                        failed.push(format!("nft table {NFT_TABLE}: {e:#}"));
                    }
                }
            } else if !st.fw_added.is_empty() {
                // Rules added into our pre-existing table: delete the table's
                // rules we added is fiddly (handles); recreate-clean instead
                // is unsafe if someone else added rules. Flush ONLY our table.
                match crate::swap::sudo(&["nft", "flush", "table", "inet", NFT_TABLE]) {
                    Ok(()) => undone.push(format!("flushed llmtune's nft table {NFT_TABLE}")),
                    Err(e) => {
                        residual.fw_added.extend(st.fw_added.iter().cloned());
                        failed.push(format!("nft flush {NFT_TABLE}: {e:#}"));
                    }
                }
            }
        }
    }
}

/// The staged-image slice of a status snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageState {
    /// A verified staged image is active.
    Staged { id: String, init: String },
    /// No staged image (legacy kernel/initrd paths served).
    None,
    /// A staged image exists but fails verification.
    Error(String),
}

/// One probe of the whole boot-server surface (services, firewall, exports,
/// staged image, reachability) as DATA - the TUI dashboard renders this on a
/// worker thread; the CLI `netboot status` prints it.
#[derive(Debug, Clone)]
pub struct ServerStatus {
    pub server_ip: String,
    pub http_base: String,
    pub lan_cidr: String,
    pub models_dir: String,
    pub services: Vec<(String, String)>,
    pub backend: Backend,
    /// Per required port: open? (None = could not read, needs sudo).
    pub firewall: Vec<(PortRule, Option<bool>)>,
    pub exports_ok: bool,
    pub image: ImageState,
    pub reach: ReachReport,
    pub verdict: Verdict,
    pub detail: String,
}

/// Probe everything `netboot status` reports, returning the snapshot instead
/// of printing. Runs subprocesses and (for the off-host reachability proof)
/// an SSH probe - callers with a UI run this on a worker thread.
pub fn status_snapshot(cfg: &Config) -> Result<ServerStatus> {
    let nb = crate::netboot::require(cfg)?;
    let services: Vec<(String, String)> = stack(nb)
        .iter()
        .map(|s| (s.to_string(), is_active(s)))
        .collect();

    // Firewall (read-only, non-interactive sudo: degrade to unknown rather
    // than hang a TUI/agent on a password prompt).
    let backend = detect_backend(false);
    let rules = required_ports(nb);
    let fw_listing = match backend {
        Backend::Ufw => priv_output(&["ufw", "status"], false),
        Backend::Firewalld => priv_output(&["firewall-cmd", "--list-rich-rules"], false),
        Backend::Nftables => priv_output(&["nft", "list", "table", "inet", NFT_TABLE], false),
        Backend::None => None,
    };
    let firewall: Vec<(PortRule, Option<bool>)> = rules
        .iter()
        .map(|r| {
            let present = match (&backend, &fw_listing) {
                (Backend::None, _) => Some(true), // nothing filters
                (Backend::Ufw, Some(l)) => Some(ufw_rule_present(l, &nb.lan_cidr, r)),
                (Backend::Firewalld, Some(l)) => Some(firewalld_rule_present(
                    l,
                    &firewalld_rich_rule(&nb.lan_cidr, r),
                )),
                (Backend::Nftables, Some(l)) => Some(nft_rule_present(l, &nb.lan_cidr, r)),
                (_, None) => None, // could not read (sudo -n refused)
            };
            (*r, present)
        })
        .collect();

    let exports_ok = exports_present(&read_exports_corpus(false), &nb.models_dir, &nb.lan_cidr);

    let image = match crate::netboot_image::load_active(Path::new(&nb.images_dir)) {
        Ok(Some((m, _))) => ImageState::Staged {
            id: m.id.clone(),
            init: m.init.clone(),
        },
        Ok(None) => ImageState::None,
        Err(e) => ImageState::Error(format!("{e:#}")),
    };

    let reach = reach_check(cfg, nb);
    let (verdict, detail) = diagnose(&reach, backend);

    Ok(ServerStatus {
        server_ip: nb.server_ip.clone(),
        http_base: nb.http_base(),
        lan_cidr: nb.lan_cidr.clone(),
        models_dir: nb.models_dir.clone(),
        services,
        backend,
        firewall,
        exports_ok,
        image,
        reach,
        verdict,
        detail,
    })
}

/// `netboot status`: services + firewall port state + exports + staged image
/// + reachability, one view (rendered from [`status_snapshot`]).
pub fn status(cfg: &Config, json: bool) -> Result<()> {
    let st = status_snapshot(cfg)?;
    if json {
        let svcs: Vec<serde_json::Value> = st
            .services
            .iter()
            .map(|(s, a)| serde_json::json!({"service": s, "active": a}))
            .collect();
        let fw: Vec<serde_json::Value> = st
            .firewall
            .iter()
            .map(|(r, p)| {
                serde_json::json!({
                    "port": r.port, "proto": r.proto.as_str(), "what": r.what, "open": p,
                })
            })
            .collect();
        let img = match &st.image {
            ImageState::Staged { id, init } => serde_json::json!({"active": id, "init": init}),
            ImageState::None => serde_json::Value::Null,
            ImageState::Error(e) => serde_json::json!({"error": e}),
        };
        println!(
            "{}",
            serde_json::json!({
                "server_ip": st.server_ip,
                "http": st.http_base,
                "services": svcs,
                "firewall": {"backend": st.backend.as_str(), "rules": fw},
                "exports": {
                    "models_dir": st.models_dir,
                    "lan_cidr": st.lan_cidr,
                    "present": st.exports_ok,
                },
                "image": img,
                "reachability": reach_json_value(&st.reach, &st.verdict, &st.detail),
            })
        );
        return Ok(());
    }

    println!("netboot server {} ({})", st.server_ip, st.http_base);
    println!("services:");
    for (s, a) in &st.services {
        let mark = if a == "active" { "[ok]  " } else { "[fail]" };
        println!("  {mark} {s}: {a}");
    }
    println!("firewall ({}):", st.backend.as_str());
    for (r, p) in &st.firewall {
        let mark = match p {
            Some(true) => "[ok]  ",
            Some(false) => "[fail]",
            None => "[??]  ",
        };
        let state = match p {
            Some(true) => "open",
            Some(false) => "NOT open",
            None => "unknown (needs sudo)",
        };
        println!(
            "  {mark} {}/{} from {}: {state}  ({})",
            r.port,
            r.proto.as_str(),
            st.lan_cidr,
            r.what
        );
    }
    println!("exports:");
    let mark = if st.exports_ok { "[ok]  " } else { "[fail]" };
    println!(
        "  {mark} {} -> {} (ro): {}",
        st.models_dir,
        st.lan_cidr,
        if st.exports_ok {
            "exported"
        } else {
            "NOT exported"
        }
    );
    println!("image:");
    match &st.image {
        ImageState::Staged { id, init } => println!("  [ok]   staged {id} (init= {init})"),
        ImageState::None => {
            println!("  [--]   no staged image (legacy kernel/initrd paths served)")
        }
        ImageState::Error(e) => println!("  [fail] staged image fails verification: {e}"),
    }
    print_reach(&st.reach, &st.verdict, &st.detail);
    Ok(())
}

// ---------------------------------------------------------------------------
// Reachability rendering.
// ---------------------------------------------------------------------------

fn reach_json_value(r: &ReachReport, verdict: &Verdict, msg: &str) -> serde_json::Value {
    let checks = |v: &[PortCheck]| -> Vec<serde_json::Value> {
        v.iter()
            .map(|c| serde_json::json!({"port": c.port, "open": c.ok}))
            .collect()
    };
    serde_json::json!({
        "verdict": verdict.as_str(),
        "detail": msg,
        "local": checks(&r.local),
        "remote_host": r.remote_host,
        "remote": checks(&r.remote),
    })
}

fn print_reach(r: &ReachReport, verdict: &Verdict, msg: &str) {
    println!("reachability:");
    for c in &r.local {
        let mark = if c.ok { "[ok]  " } else { "[fail]" };
        println!(
            "  {mark} local  :{} {}",
            c.port,
            if c.ok { "open" } else { "closed" }
        );
    }
    match (&r.remote_host, r.remote.is_empty()) {
        (Some(host), false) => {
            for c in &r.remote {
                let mark = if c.ok { "[ok]  " } else { "[fail]" };
                println!(
                    "  {mark} remote :{} {} (probed from {host})",
                    c.port,
                    if c.ok { "open" } else { "closed" }
                );
            }
        }
        (Some(host), true) => println!("  [??]   remote probe from {host} failed (no evidence)"),
        (None, _) => println!("  [??]   no off-host probe node configured"),
    }
    let mark = match verdict {
        Verdict::Ok => "[ok]  ",
        Verdict::LocalOnlyUnproven => "[warn]",
        _ => "[fail]",
    };
    println!("  {mark} {}: {msg}", verdict.as_str());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn nb_toml(extra: &str) -> Netboot {
        let toml = format!(
            "[netboot]\n\
             interface = \"enp6s0\"\n\
             server_ip = \"198.51.100.225\"\n\
             subnet = \"198.51.100.0\"\n\
             {extra}"
        );
        Config::load_str(&toml).unwrap().netboot.unwrap()
    }

    fn nb() -> Netboot {
        nb_toml("")
    }

    #[test]
    fn config_defaults_lan_cidr_models_dir_dnsmasq() {
        let n = nb();
        assert_eq!(n.lan_cidr, "198.51.100.0/24", "defaults to subnet/prefix");
        assert_eq!(n.models_dir, "/var/lib/llmtune/models");
        assert!(!n.dnsmasq, "dnsmasq off by default");
        let n2 =
            nb_toml("lan_cidr = \"10.0.0.0/16\"\nmodels_dir = \"/srv/models\"\ndnsmasq = true\n");
        assert_eq!(n2.lan_cidr, "10.0.0.0/16");
        assert_eq!(n2.models_dir, "/srv/models");
        assert!(n2.dnsmasq);
    }

    #[test]
    fn config_rejects_bogus_lan_cidr() {
        for bad in ["not-a-cidr", "192.168.1.0/33", "192.168.1.0", "a.b.c.d/24"] {
            let toml = format!(
                "[netboot]\ninterface = \"e\"\nserver_ip = \"198.51.100.225\"\n\
                 subnet = \"198.51.100.0\"\nlan_cidr = \"{bad}\"\n"
            );
            assert!(
                Config::load_str(&toml).is_err(),
                "lan_cidr '{bad}' must be rejected"
            );
        }
    }

    #[test]
    fn required_ports_cover_http_and_nfs_and_gate_dnsmasq() {
        let base = required_ports(&nb());
        let toks: Vec<String> = base.iter().map(|r| r.token()).collect();
        assert_eq!(toks, vec!["8090/tcp", "2049/tcp", "111/tcp", "111/udp"]);
        let with = required_ports(&nb_toml("dnsmasq = true\n"));
        let toks: Vec<String> = with.iter().map(|r| r.token()).collect();
        assert!(toks.contains(&"67/udp".into()) && toks.contains(&"69/udp".into()));
        assert!(toks.contains(&"4011/udp".into()));
    }

    // -- ufw --------------------------------------------------------------

    #[test]
    fn ufw_argv_is_scoped_and_delete_mirrors_allow() {
        let r = PortRule {
            port: 8090,
            proto: Proto::Tcp,
            what: "",
        };
        let add = ufw_allow_argv("192.168.1.0/24", &r);
        assert_eq!(
            add,
            vec![
                "ufw",
                "allow",
                "from",
                "192.168.1.0/24",
                "to",
                "any",
                "port",
                "8090",
                "proto",
                "tcp",
                "comment",
                "llmtune-netboot"
            ]
        );
        let del = ufw_delete_argv("192.168.1.0/24", &r);
        assert_eq!(
            del,
            vec![
                "ufw",
                "delete",
                "allow",
                "from",
                "192.168.1.0/24",
                "to",
                "any",
                "port",
                "8090",
                "proto",
                "tcp"
            ],
            "delete matches the added spec exactly (no comment)"
        );
        assert!(
            !add.iter().any(|a| a.contains("flush") || a == "reset"),
            "additive only"
        );
    }

    #[test]
    fn ufw_status_parse_active_and_rule_presence() {
        let status = "Status: active\n\n\
             To                         Action      From\n\
             --                         ------      ----\n\
             22/tcp                     ALLOW       Anywhere\n\
             8090/tcp                   ALLOW       192.168.1.0/24\n\
             2049/tcp                   ALLOW       192.168.1.0/24\n";
        assert!(ufw_active(status));
        assert!(!ufw_active("Status: inactive\n"));
        let http = PortRule {
            port: 8090,
            proto: Proto::Tcp,
            what: "",
        };
        assert!(ufw_rule_present(status, "192.168.1.0/24", &http));
        // Same port, wrong scope: not ours.
        assert!(!ufw_rule_present(status, "10.0.0.0/8", &http));
        // Port not listed at all.
        let rpc = PortRule {
            port: 111,
            proto: Proto::Udp,
            what: "",
        };
        assert!(!ufw_rule_present(status, "192.168.1.0/24", &rpc));
        // A bare-port rule (added without proto) covers BOTH protocols: the
        // live cache host renders `111  ALLOW  192.168.1.0/24` that way.
        let bare = "Status: active\n111                        ALLOW       192.168.1.0/24\n";
        assert!(ufw_rule_present(bare, "192.168.1.0/24", &rpc));
        let rpc_tcp = PortRule {
            port: 111,
            proto: Proto::Tcp,
            what: "",
        };
        assert!(ufw_rule_present(bare, "192.168.1.0/24", &rpc_tcp));
        // But a bare 1111 must not match 111.
        let p111 = PortRule {
            port: 1111,
            proto: Proto::Tcp,
            what: "",
        };
        assert!(!ufw_rule_present(bare, "192.168.1.0/24", &p111));
        // 22/tcp is ALLOW Anywhere, not cidr-scoped: must not match a cidr query.
        let ssh = PortRule {
            port: 22,
            proto: Proto::Tcp,
            what: "",
        };
        assert!(!ufw_rule_present(status, "192.168.1.0/24", &ssh));
    }

    // -- firewalld ----------------------------------------------------------

    #[test]
    fn firewalld_rich_rule_and_presence() {
        let r = PortRule {
            port: 2049,
            proto: Proto::Tcp,
            what: "",
        };
        let rule = firewalld_rich_rule("192.168.1.0/24", &r);
        assert_eq!(
            rule,
            "rule family=\"ipv4\" source address=\"192.168.1.0/24\" \
             port port=\"2049\" protocol=\"tcp\" accept"
        );
        let listing = format!("rule family=\"ipv4\" source address=\"10.9.8.0/24\" port port=\"80\" protocol=\"tcp\" accept\n{rule}\n");
        assert!(firewalld_rule_present(&listing, &rule));
        assert!(!firewalld_rule_present(
            "rule family=\"ipv4\" accept\n",
            &rule
        ));
    }

    // -- nftables -----------------------------------------------------------

    #[test]
    fn nft_rules_live_in_llmtunes_own_table_only() {
        let r = PortRule {
            port: 8090,
            proto: Proto::Tcp,
            what: "",
        };
        let argv = nft_add_rule_argv("192.168.1.0/24", &r);
        assert_eq!(argv[3..6], ["inet", NFT_TABLE, "input"]);
        assert!(!argv.iter().any(|a| a == "flush"), "additive only");
        let listing = format!(
            "table inet llmtune_netboot {{\n\tchain input {{\n\t\ttype filter hook input \
             priority -10; policy accept;\n\t\t{}\n\t}}\n}}\n",
            nft_rule_needle("192.168.1.0/24", &r)
        );
        assert!(nft_rule_present(&listing, "192.168.1.0/24", &r));
        let other = PortRule {
            port: 2049,
            proto: Proto::Tcp,
            what: "",
        };
        assert!(!nft_rule_present(&listing, "192.168.1.0/24", &other));
    }

    // -- exports ------------------------------------------------------------

    #[test]
    fn exports_line_is_ro_scoped_root_squashed() {
        assert_eq!(
            exports_line("/var/lib/llmtune/models", "192.168.1.0/24"),
            "/var/lib/llmtune/models 192.168.1.0/24(ro,sync,no_subtree_check,root_squash)"
        );
    }

    #[test]
    fn exports_presence_is_idempotent_and_respects_hand_managed_lines() {
        let dir = "/var/lib/llmtune/models";
        let cidr = "192.168.1.0/24";
        // Our own generated file counts as present (re-running up adds nothing).
        assert!(exports_present(&exports_file(dir, cidr), dir, cidr));
        // A hand-managed export with DIFFERENT options still counts as present.
        let hand = "# my exports\n/var/lib/llmtune/models 192.168.1.0/24(rw,async)\n";
        assert!(exports_present(hand, dir, cidr));
        // Same dir, different client: NOT present for our cidr.
        let other = "/var/lib/llmtune/models 10.0.0.0/8(ro)\n";
        assert!(!exports_present(other, dir, cidr));
        // Different dir, same cidr: NOT present.
        let otherdir = "/srv/nfs/bc250-root 192.168.1.0/24(ro)\n";
        assert!(!exports_present(otherdir, dir, cidr));
        // Comments and blanks never match.
        assert!(!exports_present(
            "# /var/lib/llmtune/models 192.168.1.0/24(ro)\n\n",
            dir,
            cidr
        ));
        // Unrelated existing exports are untouched conceptually: presence of
        // ours does not depend on them.
        let corpus = format!("/srv/other 192.168.1.0/24(rw)\n{}", exports_line(dir, cidr));
        assert!(exports_present(&corpus, dir, cidr));
    }

    // -- state --------------------------------------------------------------

    #[test]
    fn up_state_roundtrips_and_tokens_parse() {
        let st = UpState {
            version: STATE_VERSION,
            firewall_backend: "ufw".into(),
            fw_added: vec!["8090/tcp".into(), "111/udp".into()],
            nft_table_added: false,
            exports_file_added: true,
            started: vec![HTTP_UNIT.into()],
            enabled: vec![HTTP_UNIT.into(), "nfs-server".into()],
        };
        let text = toml::to_string_pretty(&st).unwrap();
        let back: UpState = toml::from_str(&text).unwrap();
        assert_eq!(back, st);
        let r = parse_rule_token("8090/tcp").unwrap();
        assert_eq!((r.port, r.proto), (8090, Proto::Tcp));
        let r = parse_rule_token("111/udp").unwrap();
        assert_eq!((r.port, r.proto), (111, Proto::Udp));
        assert!(parse_rule_token("garbage").is_none());
        assert!(parse_rule_token("80/icmp").is_none());
        assert_eq!("ufw".to_string().parse_backend(), Some(Backend::Ufw));
        assert_eq!("none".to_string().parse_backend(), Some(Backend::None));
        assert_eq!("bogus".to_string().parse_backend(), None);
    }

    #[test]
    fn rerun_up_merges_prior_state_so_down_still_reverses_everything() {
        // First up added the rule + started the service.
        let first = UpState {
            version: STATE_VERSION,
            firewall_backend: "ufw".into(),
            fw_added: vec!["8091/tcp".into()],
            nft_table_added: false,
            exports_file_added: true,
            started: vec![HTTP_UNIT.into()],
            enabled: vec![HTTP_UNIT.into()],
        };
        // Second up saw everything "already present" and recorded nothing.
        let mut second = UpState {
            version: STATE_VERSION,
            firewall_backend: "ufw".into(),
            ..Default::default()
        };
        second.merge_prior(&first);
        assert_eq!(second.fw_added, vec!["8091/tcp".to_string()]);
        assert!(second.exports_file_added);
        assert_eq!(second.started, vec![HTTP_UNIT.to_string()]);
        assert_eq!(second.enabled, vec![HTTP_UNIT.to_string()]);
        // No duplicates when both runs recorded the same item.
        let mut third = first.clone();
        third.merge_prior(&first);
        assert_eq!(third.fw_added.len(), 1);
        assert_eq!(third.started.len(), 1);
    }

    // -- reachability -------------------------------------------------------

    fn ck(port: u16, ok: bool) -> PortCheck {
        PortCheck { port, ok }
    }

    #[test]
    fn diagnose_ok_when_remote_proves_all_ports() {
        let r = ReachReport {
            local: vec![ck(8090, true), ck(2049, true)],
            remote_host: Some("node-a (198.51.100.82)".into()),
            remote: vec![ck(8090, true), ck(2049, true)],
        };
        let (v, msg) = diagnose(&r, Backend::Ufw);
        assert_eq!(v, Verdict::Ok);
        assert!(msg.contains("node-a"), "names the probe host: {msg}");
    }

    #[test]
    fn diagnose_flags_interfering_firewall_on_local_ok_remote_dead() {
        // THE reference-session trap: localhost fine, LAN SYN silently dropped.
        let r = ReachReport {
            local: vec![ck(8090, true), ck(2049, true)],
            remote_host: Some("node-a (198.51.100.82)".into()),
            remote: vec![ck(8090, false), ck(2049, true)],
        };
        let (v, msg) = diagnose(&r, Backend::Ufw);
        assert_eq!(v, Verdict::Interference(vec![8090]));
        assert!(
            msg.contains("INTERFERING FIREWALL") && msg.contains(":8090"),
            "crisp diagnosis: {msg}"
        );
        assert!(msg.contains("ufw"), "backend-specific hint: {msg}");
        // nftables hint explains the cross-table drop limit.
        let (_, msg) = diagnose(&r, Backend::Nftables);
        assert!(msg.contains("another nft table"), "{msg}");
    }

    #[test]
    fn diagnose_warns_localhost_only_is_not_proof() {
        let r = ReachReport {
            local: vec![ck(8090, true)],
            remote_host: None,
            remote: vec![],
        };
        let (v, msg) = diagnose(&r, Backend::None);
        assert_eq!(v, Verdict::LocalOnlyUnproven);
        assert!(
            msg.contains("does NOT prove LAN reachability"),
            "explicit warning: {msg}"
        );
        // A remote host that failed to probe (empty results) is also unproven.
        let r2 = ReachReport {
            local: vec![ck(8090, true)],
            remote_host: Some("node-a".into()),
            remote: vec![],
        };
        assert_eq!(diagnose(&r2, Backend::None).0, Verdict::LocalOnlyUnproven);
    }

    #[test]
    fn diagnose_not_listening_beats_firewall_talk() {
        let r = ReachReport {
            local: vec![ck(8090, false), ck(2049, true)],
            remote_host: Some("node-a".into()),
            remote: vec![ck(8090, false), ck(2049, true)],
        };
        let (v, msg) = diagnose(&r, Backend::Ufw);
        assert_eq!(v, Verdict::NotListening(vec![8090]));
        assert!(
            msg.contains("service is down") && !msg.contains("INTERFERING"),
            "service problem, not firewall: {msg}"
        );
    }

    #[test]
    fn probe_script_and_output_roundtrip() {
        let s = probe_script("198.51.100.225", &[8090, 2049, 111]);
        assert!(s.contains("/dev/tcp/198.51.100.225/$p"));
        assert!(s.contains("8090 2049 111"));
        assert!(s.contains("timeout 5"), "bounded probe");
        let parsed = parse_probe_output("8090 open\n2049 closed\n111 open\njunk line\n");
        assert_eq!(parsed, vec![ck(8090, true), ck(2049, false), ck(111, true)]);
        assert!(parse_probe_output("bash: not found\n").is_empty());
    }

    #[test]
    fn probe_node_selection_skips_self_and_local() {
        let toml = "\
            [netboot]\n\
            interface = \"enp6s0\"\n\
            server_ip = \"198.51.100.225\"\n\
            subnet = \"198.51.100.0\"\n\
            [[node]]\nname = \"self\"\nhost = \"198.51.100.225\"\ntransport = \"ssh\"\n\
            [[node]]\nname = \"loop\"\nhost = \"localhost\"\ntransport = \"ssh\"\n\
            [[node]]\nname = \"local\"\ntransport = \"local\"\n\
            [[node]]\nname = \"node-a\"\nhost = \"198.51.100.82\"\ntransport = \"ssh\"\n";
        let cfg = Config::load_str(toml).unwrap();
        let nb = cfg.netboot.clone().unwrap();
        let n = pick_probe_node(&cfg, &nb).expect("finds the off-host node");
        assert_eq!(n.name, "node-a");
        // No candidates -> None (and the diagnosis path warns).
        let cfg2 = Config::load_str(
            "[netboot]\ninterface = \"e\"\nserver_ip = \"198.51.100.225\"\nsubnet = \"198.51.100.0\"\n",
        )
        .unwrap();
        let nb2 = cfg2.netboot.clone().unwrap();
        assert!(pick_probe_node(&cfg2, &nb2).is_none());
    }

    // -- status/json shaping --------------------------------------------------

    #[test]
    fn reach_json_shape_is_stable() {
        let r = ReachReport {
            local: vec![ck(8090, true)],
            remote_host: Some("node-a (198.51.100.82)".into()),
            remote: vec![ck(8090, false)],
        };
        let (v, msg) = diagnose(&r, Backend::Ufw);
        let val = reach_json_value(&r, &v, &msg);
        assert_eq!(val["verdict"], "interfering-firewall");
        assert_eq!(val["local"][0]["port"], 8090);
        assert_eq!(val["local"][0]["open"], true);
        assert_eq!(val["remote"][0]["open"], false);
        assert_eq!(val["remote_host"], "node-a (198.51.100.82)");
        assert!(val["detail"].as_str().unwrap().contains("INTERFERING"));
    }

    #[test]
    fn up_summary_json_is_one_object_with_or_without_reach() {
        let st = UpState {
            version: STATE_VERSION,
            firewall_backend: "ufw".into(),
            fw_added: vec!["8090/tcp".into(), "2049/tcp".into()],
            nft_table_added: false,
            exports_file_added: true,
            started: vec!["llmtune-netboot.service".into()],
            enabled: vec![],
        };
        let v = up_summary_json(&st, None);
        assert_eq!(v["up"], serde_json::Value::Bool(true));
        assert_eq!(v["services_started"][0], "llmtune-netboot.service");
        assert_eq!(v["exports_file_added"], serde_json::Value::Bool(true));
        assert_eq!(v["firewall"]["backend"], "ufw");
        assert_eq!(v["firewall"]["rules_added"][1], "2049/tcp");
        assert!(v["reachability"].is_null(), "--skip-check = null");

        let r = ReachReport {
            local: vec![ck(8090, true)],
            remote_host: None,
            remote: vec![],
        };
        let (verdict, msg) = diagnose(&r, Backend::None);
        let v = up_summary_json(&st, Some(reach_json_value(&r, &verdict, &msg)));
        assert!(v["reachability"]["verdict"].is_string());
    }

    // -- down residual-state semantics (#37) ----------------------------------

    #[test]
    fn owns_nothing_gates_state_file_deletion() {
        // `down` may delete the record only when NOTHING is still owned - any
        // residual (a failed removal) must keep the file so a re-run retries.
        let empty = UpState {
            version: STATE_VERSION,
            firewall_backend: "ufw".into(), // backend alone owns nothing
            ..Default::default()
        };
        assert!(empty.owns_nothing());
        for st in [
            UpState {
                fw_added: vec!["8090/tcp".into()],
                ..empty.clone()
            },
            UpState {
                nft_table_added: true,
                ..empty.clone()
            },
            UpState {
                exports_file_added: true,
                ..empty.clone()
            },
            UpState {
                started: vec!["nfs-server".into()],
                ..empty.clone()
            },
            UpState {
                enabled: vec!["dnsmasq".into()],
                ..empty.clone()
            },
        ] {
            assert!(!st.owns_nothing(), "{st:?} still owns something");
        }
    }

    #[test]
    fn residual_state_roundtrips_through_the_state_file_format() {
        // The residual `down` writes must read back as a normal UpState so
        // the NEXT down (or up-merge) can retry exactly the failed removals.
        let residual = UpState {
            version: STATE_VERSION,
            firewall_backend: "firewalld".into(),
            fw_added: vec!["2049/tcp".into()],
            nft_table_added: false,
            exports_file_added: true,
            started: vec!["nfs-server".into()],
            enabled: vec![],
        };
        let text = toml::to_string_pretty(&residual).unwrap();
        let back: UpState = toml::from_str(&text).unwrap();
        assert_eq!(back, residual);
    }

    // -- endpoint exposure argv builders (#39) --------------------------------

    #[test]
    fn expose_argv_per_backend() {
        assert_eq!(
            ufw_expose_argv(8080, true),
            vec!["ufw", "allow", "8080/tcp"]
        );
        assert_eq!(
            ufw_expose_argv(8080, false),
            vec!["ufw", "delete", "allow", "8080/tcp"]
        );
        assert_eq!(firewalld_expose_arg(8080, true), "--add-port=8080/tcp");
        assert_eq!(firewalld_expose_arg(8080, false), "--remove-port=8080/tcp");
        assert_eq!(nft_expose_needle(8080), "tcp dport 8080 accept");
    }

    #[test]
    fn expose_hint_matches_backend() {
        // The failure hint must be actionable on the user's ACTUAL distro -
        // "sudo ufw allow" on a firewalld box was the #39 misleading half.
        assert!(expose_hint(Backend::Ufw, 8080, true).contains("ufw allow 8080/tcp"));
        let fd = expose_hint(Backend::Firewalld, 8080, true);
        assert!(
            fd.contains("--add-port=8080/tcp") && fd.contains("--reload"),
            "{fd}"
        );
        let fd_off = expose_hint(Backend::Firewalld, 8080, false);
        assert!(fd_off.contains("--remove-port=8080/tcp"), "{fd_off}");
        let nft = expose_hint(Backend::Nftables, 8080, true);
        assert!(
            nft.contains(NFT_EXPOSE_TABLE) && nft.contains("dport 8080"),
            "{nft}"
        );
        assert!(expose_hint(Backend::None, 8080, true).contains("no active firewall"));
    }
}
