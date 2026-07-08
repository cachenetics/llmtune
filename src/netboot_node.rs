// SPDX-License-Identifier: GPL-2.0-only
//! Node lifecycle for netbooted boards: `netboot arm/boot/console`.
//!
//! `arm` sets a ONE-SHOT boot into the node's iPXE entry with `efibootmgr -n`
//! over SSH. That is the ONLY firmware surface this module touches: BootNext is
//! a standard UEFI runtime variable written through efivarfs by the stock
//! `efibootmgr` tool, cleared by the firmware after one boot. This module NEVER
//! writes a BIOS setup variable, never appends an NVAR, and never goes near the
//! vendor SMM interface - an SMM NVAR-append bricked a board to no-POST in the
//! reference session (recovered only by external SPI reflash). The invariant is
//! enforced structurally: every remote mutation goes through [`arm_argv`] /
//! [`disarm_argv`], which can only ever produce `efibootmgr -n <hex4>` or
//! `efibootmgr -N`.
//!
//! `boot` arms and then triggers power. A netboot NEEDS a COLD power cycle: a
//! warm `systemctl reboot` does not reliably reset the RTL8168 PHY, so iPXE
//! often gets no DHCP link and the one-shot is wasted. With a per-node
//! `power_cmd` (a smart-plug hook) the cycle is automated; without one, the
//! operator is told to pull power - `boot` never issues a warm reboot.
//!
//! `console` is the serial-free boot visibility: a netconsole UDP receiver
//! (default port 6666, matching the image's netconsole sender, which ships the
//! kernel ring buffer + journald ForwardToKMsg over UDP) that prints the node's
//! live boot log and can capture it timestamped to a file.
//!
//! The parsers/planners are pure functions (unit-tested); the orchestrators do
//! the SSH/UDP IO.

use anyhow::{bail, Context, Result};
use std::io::Write;
use std::net::{IpAddr, ToSocketAddrs, UdpSocket};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::config::{Config, Node, Transport};

/// Default netconsole UDP port (matches the image's netconsole sender).
pub const DEFAULT_CONSOLE_PORT: u16 = 6666;

// ---------------------------------------------------------------------------
// efibootmgr output parsing (pure).
// ---------------------------------------------------------------------------

/// One `BootXXXX` entry from `efibootmgr` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EfiEntry {
    pub num: u16,
    /// `*` after the number = active.
    pub active: bool,
    pub label: String,
    /// The device path half of the line ("" when efibootmgr printed none).
    pub path: String,
}

/// The parsed EFI boot state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EfiState {
    pub boot_next: Option<u16>,
    pub boot_current: Option<u16>,
    pub boot_order: Vec<u16>,
    pub entries: Vec<EfiEntry>,
}

fn parse_hex4(s: &str) -> Option<u16> {
    let t = s.trim();
    (t.len() == 4).then(|| u16::from_str_radix(t, 16).ok())?
}

/// Parse `efibootmgr` (no args) output. Unknown lines are skipped, so vendor
/// extras and future fields don't break the parser.
pub fn parse_efibootmgr(text: &str) -> EfiState {
    let mut st = EfiState::default();
    for line in text.lines() {
        let line = line.trim_end();
        if let Some(v) = line.strip_prefix("BootNext:") {
            st.boot_next = parse_hex4(v);
        } else if let Some(v) = line.strip_prefix("BootCurrent:") {
            st.boot_current = parse_hex4(v);
        } else if let Some(v) = line.strip_prefix("BootOrder:") {
            st.boot_order = v.split(',').filter_map(parse_hex4).collect();
        } else if let Some(rest) = line.strip_prefix("Boot") {
            // BootXXXX* Label\tDevicePath  (tab separates label from path;
            // older efibootmgr may print no path at all). Checked split: the
            // output comes from a REMOTE host, and a multibyte character
            // straddling byte 4 would make a plain `split_at` panic the whole
            // control plane on one garbled line.
            let Some((nums, tail)) = rest.split_at_checked(4) else {
                continue;
            };
            let Some(num) = parse_hex4(nums) else {
                continue;
            };
            let active = tail.starts_with('*');
            let tail = tail.trim_start_matches(['*', ' ']);
            let (label, path) = match tail.split_once('\t') {
                Some((l, p)) => (l.trim(), p.trim()),
                None => (tail.trim(), ""),
            };
            st.entries.push(EfiEntry {
                num,
                active,
                label: label.to_string(),
                path: path.to_string(),
            });
        }
    }
    st
}

/// Find the iPXE boot entry: first by device path (`ipxe.efi` anywhere in it,
/// case-insensitive - the reference node's ESP chainloader lives at
/// `\EFI\ipxe\ipxe.efi`), then by label containing "ipxe". Ambiguity (several
/// path matches) is an error rather than a guess: arming the wrong entry
/// one-shots the wrong loader.
pub fn find_ipxe_entry(st: &EfiState) -> Result<&EfiEntry> {
    let by_path: Vec<&EfiEntry> = st
        .entries
        .iter()
        .filter(|e| e.path.to_lowercase().contains("ipxe.efi"))
        .collect();
    match by_path.len() {
        1 => return Ok(by_path[0]),
        n if n > 1 => bail!(
            "ambiguous: {} boot entries reference ipxe.efi ({}) - arm the right \
             one by hand with `efibootmgr -n <num>` on the node",
            n,
            by_path
                .iter()
                .map(|e| format!("Boot{:04X} \"{}\"", e.num, e.label))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => {}
    }
    let by_label: Vec<&EfiEntry> = st
        .entries
        .iter()
        .filter(|e| e.label.to_lowercase().contains("ipxe"))
        .collect();
    match by_label.len() {
        1 => Ok(by_label[0]),
        0 => bail!(
            "no iPXE boot entry found on the node (looked for an ipxe.efi device \
             path, then an 'ipxe' label). Install the chainloader on the node's \
             ESP (\\EFI\\ipxe\\ipxe.efi) and add a boot entry for it first."
        ),
        n => bail!(
            "ambiguous: {n} boot entries are labelled ipxe - arm the right one \
             by hand with `efibootmgr -n <num>` on the node"
        ),
    }
}

/// The ONLY arming mutation this module can produce: `efibootmgr -n <hex4>`,
/// a one-shot BootNext write. Nothing here can emit `-o` (BootOrder), a setup
/// variable write, or any SMM/NVAR path.
pub fn arm_argv(bootnum: u16) -> Vec<String> {
    vec!["efibootmgr".into(), "-n".into(), format!("{bootnum:04X}")]
}

/// The disarm mutation: `efibootmgr -N` deletes BootNext (back to BootOrder).
pub fn disarm_argv() -> Vec<String> {
    vec!["efibootmgr".into(), "-N".into()]
}

// ---------------------------------------------------------------------------
// SSH plumbing: run a raw command on a fleet node (same option set as the
// fleet transport; the remote command is NOT `llmtune node ...` here).
// ---------------------------------------------------------------------------

/// Build the ssh argv to run `script` (a `sh -c` body) on a node. Pure.
pub fn ssh_script_argv(node: &Node, script: &str) -> Vec<String> {
    let mut v: Vec<String> = vec![
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "ConnectTimeout=8".into(),
        "-o".into(),
        format!(
            "StrictHostKeyChecking={}",
            crate::transport::hostkey_policy()
        ),
        "-o".into(),
        "ServerAliveInterval=5".into(),
        "-o".into(),
        "ServerAliveCountMax=3".into(),
    ];
    if let Some(k) = &node.ssh_key {
        v.push("-i".into());
        v.push(k.clone());
    }
    let host = node.host.clone().unwrap_or_default();
    // End-of-options guard: a target starting with `-` must never be read as
    // an ssh option (see `transport::ssh_argv`).
    v.push("--".into());
    v.push(match &node.ssh_user {
        Some(u) => format!("{u}@{host}"),
        None => host,
    });
    v.push("sh".into());
    v.push("-c".into());
    v.push(crate::transport::sh_quote(script));
    v
}

fn ssh_script(node: &Node, script: &str) -> Result<String> {
    let argv = ssh_script_argv(node, script);
    let mut cmd = std::process::Command::new("ssh");
    cmd.args(&argv);
    // Bounded like the fleet transport: efibootmgr output is a few KiB, so a
    // node streaming more is garbage, and one hung node must not stall an
    // arm/boot sweep forever.
    let out =
        crate::transport::output_capped(&mut cmd, 1024 * 1024, std::time::Duration::from_secs(60))
            .with_context(|| format!("running ssh for node `{}`", node.name))?;
    if !out.status.success() {
        bail!(
            "ssh `{}` failed: {}",
            node.name,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Resolve a named node that can be armed: configured, ssh transport, a host.
fn ssh_node<'a>(cfg: &'a Config, name: &str) -> Result<&'a Node> {
    let node = cfg.node(name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown node '{name}' - see `llmtune node list`, or discover+register \
             boards with `llmtune netboot nodes --register`"
        )
    })?;
    if node.transport != Transport::Ssh {
        bail!(
            "node '{name}' is not an ssh node - arm/boot drive a REMOTE board's \
             firmware; the control host itself is never armed"
        );
    }
    if node.host.as_deref().unwrap_or("").is_empty() {
        bail!("node '{name}' has no host configured");
    }
    Ok(node)
}

// ---------------------------------------------------------------------------
// arm: one-shot BootNext -> the iPXE entry. Idempotent, verified by re-read.
// ---------------------------------------------------------------------------

/// The result of an arm/disarm, for rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmReport {
    pub node: String,
    pub ipxe_bootnum: u16,
    pub ipxe_label: String,
    pub boot_next: Option<u16>,
    /// BootNext already pointed at the iPXE entry before this run.
    pub already_armed: bool,
}

/// The arm/disarm core, shared by the CLI and the TUI dashboard: discover the
/// iPXE entry from the node's own `efibootmgr` output, set (or clear) the
/// one-shot BootNext, and VERIFY by reading it back. Returns the report plus
/// a short action phrase for rendering. Never touches BootOrder, setup
/// variables, or any SMM/NVAR surface (see the module doc).
pub fn arm_node(node: &Node, disarm: bool) -> Result<(ArmReport, &'static str)> {
    if node.transport != Transport::Ssh {
        bail!(
            "node '{}' is not an ssh node - arm/boot drive a REMOTE board's \
             firmware; the control host itself is never armed",
            node.name
        );
    }
    if node.host.as_deref().unwrap_or("").is_empty() {
        bail!("node '{}' has no host configured", node.name);
    }
    let name = node.name.as_str();
    let before = parse_efibootmgr(&ssh_script(node, "efibootmgr").context("reading efibootmgr")?);
    let ipxe = find_ipxe_entry(&before)?.clone();

    let out = if disarm {
        if before.boot_next.is_none() {
            (
                ArmReport {
                    node: name.into(),
                    ipxe_bootnum: ipxe.num,
                    ipxe_label: ipxe.label.clone(),
                    boot_next: None,
                    already_armed: false,
                },
                "already disarmed (no BootNext set)",
            )
        } else {
            let argv = disarm_argv().join(" ");
            ssh_script(node, &argv).context("clearing BootNext")?;
            let after =
                parse_efibootmgr(&ssh_script(node, "efibootmgr").context("verifying disarm")?);
            if after.boot_next.is_some() {
                bail!("disarm did not stick: BootNext still set after `efibootmgr -N`");
            }
            (
                ArmReport {
                    node: name.into(),
                    ipxe_bootnum: ipxe.num,
                    ipxe_label: ipxe.label.clone(),
                    boot_next: None,
                    already_armed: false,
                },
                "disarmed (BootNext cleared)",
            )
        }
    } else if before.boot_next == Some(ipxe.num) {
        (
            ArmReport {
                node: name.into(),
                ipxe_bootnum: ipxe.num,
                ipxe_label: ipxe.label.clone(),
                boot_next: Some(ipxe.num),
                already_armed: true,
            },
            "already armed",
        )
    } else {
        let argv = arm_argv(ipxe.num).join(" ");
        ssh_script(node, &argv).context("setting BootNext")?;
        let after = parse_efibootmgr(&ssh_script(node, "efibootmgr").context("verifying arm")?);
        if after.boot_next != Some(ipxe.num) {
            bail!(
                "arm did not stick: BootNext is {:?} after `efibootmgr -n {:04X}`",
                after.boot_next,
                ipxe.num
            );
        }
        (
            ArmReport {
                node: name.into(),
                ipxe_bootnum: ipxe.num,
                ipxe_label: ipxe.label.clone(),
                boot_next: Some(ipxe.num),
                already_armed: false,
            },
            "armed",
        )
    };
    Ok(out)
}

/// `netboot arm <node>`: the CLI wrapper over [`arm_node`] (rendering only).
pub fn arm(cfg: &Config, name: &str, disarm: bool, json: bool) -> Result<()> {
    let node = ssh_node(cfg, name)?;
    let (report, action) = arm_node(node, disarm)?;

    if json {
        println!(
            "{}",
            serde_json::json!({
                "node": report.node,
                "ipxe_entry": format!("{:04X}", report.ipxe_bootnum),
                "ipxe_label": report.ipxe_label,
                "boot_next": report.boot_next.map(|n| format!("{n:04X}")),
                "armed": report.boot_next == Some(report.ipxe_bootnum),
                "already_armed": report.already_armed,
            })
        );
    } else {
        println!(
            "[ok] {name}: {action} - iPXE entry Boot{:04X} (\"{}\"), BootNext {}",
            report.ipxe_bootnum,
            report.ipxe_label,
            match report.boot_next {
                Some(n) => format!("{n:04X} (one shot: next boot only)"),
                None => "unset".into(),
            }
        );
        if !disarm {
            println!(
                "     next: a COLD power cycle boots it once (`llmtune netboot boot {name}`); \
                 a warm reboot may leave the NIC PHY unready for iPXE DHCP"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// boot: arm + trigger power (power_cmd hook, else manual COLD-cycle prompt).
// ---------------------------------------------------------------------------

/// How the power step will be executed. Pure decision from the node config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerPlan {
    /// Run this local command (a smart-plug hook) to cold-cycle the board.
    Cmd(String),
    /// No hook configured: the operator must pull power by hand.
    Manual,
}

/// Decide the power step for a node.
pub fn power_plan(power_cmd: Option<&str>) -> PowerPlan {
    match power_cmd {
        Some(c) if !c.trim().is_empty() => PowerPlan::Cmd(c.to_string()),
        _ => PowerPlan::Manual,
    }
}

/// The manual instruction: COLD cycle only. Rendered when no power_cmd exists.
pub fn manual_power_instructions(node: &str) -> String {
    format!(
        "[..] {node} is armed but has no power_cmd - COLD power-cycle it by hand now:\n\
         \x20    1. Pull power (or switch the plug off) and wait ~5 s.\n\
         \x20    2. Restore power. The board one-shots into iPXE and netboots.\n\
         \x20    Do NOT use a warm reboot (`systemctl reboot`): it does not reset the\n\
         \x20    RTL8168 PHY, so iPXE often gets no DHCP link and the one-shot is wasted.\n\
         \x20    Tip: set `power_cmd` on this node in fleet.toml (a smart-plug hook)\n\
         \x20    to automate this."
    )
}

/// Run a node's `power_cmd` hook (a local shell command - typically a
/// smart-plug cycle). Shared by the CLI `boot` and the TUI boot flow.
pub fn run_power_cmd(name: &str, cmd: &str) -> Result<()> {
    let out = std::process::Command::new("sh")
        .args(["-c", cmd])
        .output()
        .with_context(|| format!("running power_cmd for {name}"))?;
    if !out.status.success() {
        bail!(
            "power_cmd failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// The `netboot boot --json` payload: ONE stdout object covering both steps
/// (arm + power). Pure for the unit test.
pub fn boot_json(report: &ArmReport, power: &str, power_cmd: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "node": report.node,
        "armed": report.boot_next == Some(report.ipxe_bootnum),
        "ipxe_entry": format!("{:04X}", report.ipxe_bootnum),
        // "cycled" (power_cmd ran) | "manual" (cold-cycle by hand) |
        // "aborted" (operator declined the prompt; board stays armed)
        "power": power,
        "power_cmd": power_cmd,
    })
}

/// `netboot boot <node>`: arm, then cold-cycle. With a `power_cmd` the hook
/// runs (confirmed unless `--yes`; a non-TTY caller without `--yes` is
/// refused, exit 2); without one, the manual COLD-cycle instruction is
/// printed. A warm reboot is never issued. `--json` emits ONE object.
pub fn boot(cfg: &Config, name: &str, yes: bool, json: bool) -> Result<()> {
    let node = ssh_node(cfg, name)?;
    let plan = power_plan(node.power_cmd.as_deref());

    // 1. Arm (idempotent; hard-fails before any power action if it can't).
    // In json mode the report is folded into the single boot object below;
    // in text mode `arm` renders it directly.
    let report = if json {
        Some(arm_node(node, false)?.0)
    } else {
        arm(cfg, name, false, false)?;
        None
    };
    let report = |power: &str, cmd: Option<&str>| {
        boot_json(
            report.as_ref().expect("json mode carries the report"),
            power,
            cmd,
        )
    };

    // 2. Power.
    match plan {
        PowerPlan::Cmd(cmd) => {
            if !yes
                && !confirm(&format!(
                    "cold-cycle {name} now via power_cmd `{cmd}`? [y/N] "
                ))?
            {
                if json {
                    println!("{}", report("aborted", Some(&cmd)));
                } else {
                    println!("[..] aborted before the power step ({name} stays armed)");
                }
                return Ok(());
            }
            run_power_cmd(name, &cmd)?;
            if json {
                println!("{}", report("cycled", Some(&cmd)));
            } else {
                println!("[ok] {name}: power_cmd ran - board is cold-cycling into iPXE");
                println!(
                    "     watch it boot: `llmtune netboot console {name}` \
                     (then `llmtune netboot nodes` once it's up)"
                );
            }
        }
        PowerPlan::Manual => {
            if json {
                println!("{}", report("manual", None));
                eprintln!("{}", manual_power_instructions(name));
            } else {
                println!("{}", manual_power_instructions(name));
            }
        }
    }
    Ok(())
}

/// Ask a y/N question on the tty. Non-tty stdin refuses (exit 2 - the
/// non-interactive escape hatch is `--yes`).
fn confirm(prompt: &str) -> Result<bool> {
    crate::agentic::confirm_tty(
        prompt,
        "refusing to trigger power without a tty - pass --yes to confirm",
    )
}

// ---------------------------------------------------------------------------
// console: the netconsole UDP receiver (pure parsing + an IO loop).
// ---------------------------------------------------------------------------

/// Strip terminal control characters from a netconsole line, keeping tab
/// (kernel logs use it). The stream is unauthenticated UDP from the LAN, and a
/// compromised or garbled board is exactly what the console exists to watch -
/// so ESC/C0/C1 bytes must never reach the operator's terminal: an embedded
/// OSC/CSI sequence can retitle the window, poison the clipboard, or reposition
/// the cursor over earlier output. `char::is_control` covers C0, DEL, and the
/// C1 range (U+0080..U+009F, the 8-bit CSI/OSC introducers).
fn sanitize_console_line(l: &str) -> String {
    l.chars()
        .filter(|&c| c == '\t' || !c.is_control())
        .collect()
}

/// Split a netconsole datagram into printable lines. Netconsole sends one
/// kernel message per datagram (long ones fragmented); tolerate embedded
/// newlines and drop empties. Every line is control-stripped (see
/// [`sanitize_console_line`]) before it can reach a terminal or capture file.
pub fn datagram_lines(buf: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(buf)
        .lines()
        .map(|l| sanitize_console_line(l.trim_end()))
        .filter(|l| !l.is_empty())
        .collect()
}

/// Extract the kernel `[   12.345678]` uptime prefix if present. Returns
/// (uptime_secs, message-without-prefix). Lines without the prefix (printk
/// time off, or continuation fragments) come back untimestamped, unmodified.
pub fn parse_kernel_ts(line: &str) -> (Option<f64>, &str) {
    let t = line.trim_start();
    if let Some(rest) = t.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            let ts = rest[..end].trim();
            if let Ok(secs) = ts.parse::<f64>() {
                return (Some(secs), rest[end + 1..].trim_start());
            }
        }
    }
    (None, line)
}

/// Whether a line passes the `--since` filter: keep lines at/after the kernel
/// uptime bound; untimestamped lines always pass (journald floods aside, they
/// are usually live output, and dropping them silently would hide real logs).
pub fn passes_since(kernel_ts: Option<f64>, since: Option<f64>) -> bool {
    match (kernel_ts, since) {
        (Some(ts), Some(s)) => ts >= s,
        _ => true,
    }
}

/// Render one received line for the terminal: wallclock + the message.
pub fn console_line(unix: u64, subsec_ms: u32, line: &str) -> String {
    let secs = unix % 86_400;
    format!(
        "{:02}:{:02}:{:02}.{:03} | {line}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60,
        subsec_ms
    )
}

pub struct ConsoleOpts {
    /// UDP port to bind (default [`DEFAULT_CONSOLE_PORT`], matching the image's
    /// netconsole sender).
    pub port: u16,
    /// Only show lines whose kernel uptime is >= this (skips the flood of
    /// pre-existing ring-buffer messages when journald forwards history).
    pub since: Option<f64>,
    /// Also append every line (wallclock-timestamped) to this file.
    pub file: Option<String>,
    /// Exit after this many seconds (default: run until Ctrl-C).
    pub duration_secs: Option<u64>,
    /// NDJSON mode: one JSON object per received line on stdout (headers and
    /// the ignored-sender note go to stderr).
    pub json: bool,
}

/// One `netboot console --json` NDJSON record: wallclock, the kernel uptime
/// when the line carries one, and the raw line. Pure.
pub fn console_line_json(unix: u64, subsec_ms: u32, kernel_ts: Option<f64>, line: &str) -> String {
    serde_json::json!({
        "ts_unix_ms": unix * 1000 + subsec_ms as u64,
        "kernel_ts": kernel_ts,
        "line": line,
    })
    .to_string()
}

/// Resolve a node's configured host to an IP (for netconsole source
/// filtering). Errors on a hostless node or a resolution failure.
pub fn resolve_node_ip(node: &Node) -> Result<IpAddr> {
    let host = node.host.clone().unwrap_or_default();
    if host.is_empty() {
        bail!("node '{}' has no host configured", node.name);
    }
    resolve_ip(&host)
}

/// Resolve the node's host to an IP for source filtering.
fn resolve_ip(host: &str) -> Result<IpAddr> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(ip);
    }
    let mut addrs = (host, 0u16)
        .to_socket_addrs()
        .with_context(|| format!("resolving {host}"))?;
    addrs
        .next()
        .map(|a| a.ip())
        .ok_or_else(|| anyhow::anyhow!("no address for {host}"))
}

/// `netboot console <node>`: bind the control host's netconsole port and
/// stream the node's kernel+journal boot log. Filters datagrams to the node's
/// source IP (several boards can ship to the same port); other senders are
/// counted and reported once at exit.
pub fn console(cfg: &Config, name: &str, opts: &ConsoleOpts) -> Result<()> {
    let node = ssh_node(cfg, name)?;
    let host = node.host.clone().unwrap_or_default();
    let want_ip = resolve_ip(&host)?;

    let sock = UdpSocket::bind(("0.0.0.0", opts.port)).with_context(|| {
        format!(
            "binding UDP :{} (is another console/receiver already listening?)",
            opts.port
        )
    })?;
    sock.set_read_timeout(Some(Duration::from_millis(500)))?;

    let mut capture = match &opts.file {
        Some(p) => {
            if let Some(dir) = Path::new(p).parent() {
                std::fs::create_dir_all(dir).ok();
            }
            Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .with_context(|| format!("opening capture file {p}"))?,
            )
        }
        None => None,
    };

    // In NDJSON mode stdout carries ONLY the per-line objects.
    let note = |msg: String| {
        if opts.json {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };
    note(format!(
        "listening on udp :{} for {name} ({want_ip}){}{} - Ctrl-C to stop",
        opts.port,
        opts.since
            .map(|s| format!(", since kernel t={s}s"))
            .unwrap_or_default(),
        opts.file
            .as_deref()
            .map(|f| format!(", capturing to {f}"))
            .unwrap_or_default(),
    ));
    note(
        "(the node ships kernel+journal here via netconsole; nothing arriving \
         usually means the board is off, pre-network, or netconsole points at \
         a different receiver IP)"
            .to_string(),
    );

    let start = Instant::now();
    let mut buf = [0u8; 65536];
    let mut other_senders = 0u64;
    loop {
        if let Some(d) = opts.duration_secs {
            if start.elapsed() >= Duration::from_secs(d) {
                break;
            }
        }
        let (n, src) = match sock.recv_from(&mut buf) {
            Ok(v) => v,
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(e) => return Err(e).context("receiving netconsole datagram"),
        };
        if src.ip() != want_ip {
            other_senders += 1;
            continue;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        for line in datagram_lines(&buf[..n]) {
            let (kts, _) = parse_kernel_ts(&line);
            if !passes_since(kts, opts.since) {
                continue;
            }
            if opts.json {
                println!(
                    "{}",
                    console_line_json(now.as_secs(), now.subsec_millis(), kts, &line)
                );
            } else {
                println!(
                    "{}",
                    console_line(now.as_secs(), now.subsec_millis(), &line)
                );
            }
            if let Some(f) = capture.as_mut() {
                let _ = writeln!(
                    f,
                    "{} {line}",
                    crate::fmt::fmt_ts(now.as_secs()).replace(' ', "T")
                );
            }
        }
    }
    if other_senders > 0 {
        note(format!(
            "[..] ignored {other_senders} datagram(s) from other senders on :{port}",
            port = opts.port
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim from the live reference node (bc250-005363, 2026-07-07).
    const LIVE: &str = "BootCurrent: 0000\n\
        Timeout: 1 seconds\n\
        BootOrder: 0001,0000\n\
        Boot0000* BC250 iPXE oneshot\tHD(1,GPT,9841ad3b-e71c-4781-9d40-a0e7c61f7625,0x1000,0x96000)/\\EFI\\ipxe\\ipxe.efi\n\
        Boot0001* UEFI OS\tHD(1,GPT,9841ad3b-e71c-4781-9d40-a0e7c61f7625,0x1000,0x96000)/\\EFI\\BOOT\\BOOTX64.EFI0000424f\n";

    #[test]
    fn parses_live_efibootmgr_output() {
        let st = parse_efibootmgr(LIVE);
        assert_eq!(st.boot_current, Some(0));
        assert_eq!(st.boot_next, None, "no BootNext line = unarmed");
        assert_eq!(st.boot_order, vec![1, 0]);
        assert_eq!(st.entries.len(), 2);
        assert_eq!(st.entries[0].num, 0);
        assert!(st.entries[0].active);
        assert_eq!(st.entries[0].label, "BC250 iPXE oneshot");
        assert!(st.entries[0].path.ends_with("\\EFI\\ipxe\\ipxe.efi"));
        assert_eq!(st.entries[1].label, "UEFI OS");
    }

    #[test]
    fn parses_boot_next_and_skips_garbage() {
        let st = parse_efibootmgr(
            "BootNext: 0003\nBootCurrent: 0001\nnonsense line\nBootZZZZ* bad\n\
             Boot0003* netboot\n",
        );
        assert_eq!(st.boot_next, Some(3));
        assert_eq!(st.entries.len(), 1);
        assert_eq!(st.entries[0].label, "netboot");
        assert_eq!(st.entries[0].path, "", "no-path entry tolerated");
    }

    #[test]
    fn parses_multibyte_boot_lines_without_panicking() {
        // The output comes from a remote host: a multibyte character straddling
        // the 4-byte entry-number split must be SKIPPED, not panic the control
        // plane (a plain split_at(4) aborts on a non-char boundary).
        let st = parse_efibootmgr(
            "Boot\u{00e9}999* garbled vendor line\n\
             Boot\u{4e2d}\u{6587}* CJK garbage\n\
             Boot\u{00e9}* short multibyte\n\
             Boot0001* real entry\n",
        );
        assert_eq!(st.entries.len(), 1, "only the well-formed entry parses");
        assert_eq!(st.entries[0].num, 1);
        assert_eq!(st.entries[0].label, "real entry");
    }

    #[test]
    fn ipxe_discovery_prefers_device_path_over_label() {
        // A decoy entry LABELLED ipxe but pointing elsewhere must lose to the
        // real ipxe.efi device path.
        let st = parse_efibootmgr(
            "Boot0000* my ipxe-ish thing\tHD(1)/\\EFI\\BOOT\\BOOTX64.EFI\n\
             Boot0002* netboot\tHD(1)/\\EFI\\ipxe\\ipxe.efi\n",
        );
        let e = find_ipxe_entry(&st).unwrap();
        assert_eq!(e.num, 2);
    }

    #[test]
    fn ipxe_discovery_falls_back_to_label() {
        let st = parse_efibootmgr("Boot0005* iPXE oneshot\nBoot0001* UEFI OS\n");
        assert_eq!(find_ipxe_entry(&st).unwrap().num, 5);
    }

    #[test]
    fn ipxe_discovery_finds_live_entry() {
        let st = parse_efibootmgr(LIVE);
        let e = find_ipxe_entry(&st).unwrap();
        assert_eq!(e.num, 0);
        assert_eq!(e.label, "BC250 iPXE oneshot");
    }

    #[test]
    fn ipxe_discovery_errors_on_none_and_ambiguity() {
        let none = parse_efibootmgr("Boot0001* UEFI OS\tHD(1)/x\n");
        assert!(find_ipxe_entry(&none)
            .unwrap_err()
            .to_string()
            .contains("no iPXE boot entry"));
        let ambig = parse_efibootmgr(
            "Boot0001* a\tHD(1)/\\EFI\\ipxe\\ipxe.efi\nBoot0002* b\tHD(2)/\\EFI\\ipxe\\ipxe.efi\n",
        );
        assert!(find_ipxe_entry(&ambig)
            .unwrap_err()
            .to_string()
            .contains("ambiguous"));
    }

    #[test]
    fn arm_argv_is_bootnext_only_never_smm_or_bootorder() {
        // THE safety property: arm can only ever produce a one-shot BootNext
        // write. No BootOrder (-o), no setup-variable, no vendor SMM/NVAR path.
        assert_eq!(arm_argv(0), vec!["efibootmgr", "-n", "0000"]);
        assert_eq!(arm_argv(0x1a), vec!["efibootmgr", "-n", "001A"]);
        assert_eq!(disarm_argv(), vec!["efibootmgr", "-N"]);
        for argv in [arm_argv(0xffff), disarm_argv()] {
            assert_eq!(argv[0], "efibootmgr");
            assert!(
                !argv.contains(&"-o".to_string()),
                "never rewrites BootOrder"
            );
            assert!(
                argv.iter().all(|a| !a.to_lowercase().contains("nvar")),
                "never an NVAR write"
            );
        }
    }

    #[test]
    fn ssh_script_argv_is_batch_and_quoted() {
        let node = Node {
            name: "bc250-005363".into(),
            host: Some("198.51.100.223".into()),
            transport: Transport::Ssh,
            ssh_user: Some("root".into()),
            ssh_key: Some("/home/me/.ssh/id_ed25519".into()),
            models_dir: "/var/lib/llmtune/models".into(),
            llama_unit: "llama-server.service".into(),
            llama_url: "http://127.0.0.1:8080".into(),
            power_cmd: None,
        };
        let v = ssh_script_argv(&node, "efibootmgr -n 0000");
        assert!(v.contains(&"BatchMode=yes".to_string()));
        assert!(v.contains(&"root@198.51.100.223".to_string()));
        let i = v.iter().position(|a| a == "-i").unwrap();
        assert_eq!(v[i + 1], "/home/me/.ssh/id_ed25519");
        // `--` guard immediately before the target (option-injection defense)
        let dd = v
            .iter()
            .position(|a| a == "--")
            .expect("`--` guard missing");
        assert_eq!(v[dd + 1], "root@198.51.100.223");
        // the script rides as ONE quoted sh -c body
        assert_eq!(v[v.len() - 3], "sh");
        assert_eq!(v[v.len() - 2], "-c");
        assert_eq!(v[v.len() - 1], "'efibootmgr -n 0000'");
    }

    #[test]
    fn power_plan_cmd_vs_manual() {
        assert_eq!(
            power_plan(Some("plugctl cycle bc250-1")),
            PowerPlan::Cmd("plugctl cycle bc250-1".into())
        );
        assert_eq!(power_plan(None), PowerPlan::Manual);
        assert_eq!(power_plan(Some("")), PowerPlan::Manual);
        assert_eq!(power_plan(Some("   ")), PowerPlan::Manual);
    }

    #[test]
    fn manual_instructions_mandate_cold_cycle() {
        let s = manual_power_instructions("bc250-x");
        assert!(s.contains("COLD power-cycle"));
        assert!(s.contains("RTL8168"), "explains WHY warm reboot fails");
        assert!(s.contains("power_cmd"), "points at the automation hook");
    }

    // -- netconsole parsing -------------------------------------------------

    #[test]
    fn datagram_splits_lines_and_drops_empties() {
        let lines = datagram_lines(b"[    1.234567] amdgpu: ring gfx ready\n\n");
        assert_eq!(lines, vec!["[    1.234567] amdgpu: ring gfx ready"]);
        // invalid utf8 degrades lossily, never panics
        let lossy = datagram_lines(&[0xff, 0xfe, b'o', b'k']);
        assert_eq!(lossy.len(), 1);
        assert!(lossy[0].ends_with("ok"));
    }

    #[test]
    fn datagram_strips_terminal_escapes() {
        // A malicious/garbled board must not be able to inject terminal
        // sequences: OSC (window title / clipboard), CSI (cursor movement),
        // BEL, and 8-bit C1 introducers are all stripped; tab is kept.
        let osc = datagram_lines(b"\x1b]0;pwned\x07boot ok");
        assert_eq!(osc, vec!["]0;pwnedboot ok"], "ESC and BEL stripped");
        let csi = datagram_lines(b"\x1b[2Jcleared\x1b[H");
        assert_eq!(csi, vec!["[2Jcleared[H"], "ESC stripped from CSI");
        // 8-bit C1 CSI (U+009B) and OSC (U+009D) as UTF-8
        let c1 = datagram_lines("\u{9b}31mred\u{9d}0;t".as_bytes());
        assert_eq!(c1, vec!["31mred0;t"]);
        // tab survives; embedded CR does not
        let tabs = datagram_lines(b"a\tb\rc\n");
        assert_eq!(tabs, vec!["a\tbc"]);
        // a datagram that is ONLY control bytes yields no lines
        assert!(datagram_lines(b"\x1b\x07\x08\n").is_empty());
    }

    #[test]
    fn kernel_ts_parses_and_tolerates_absence() {
        let (ts, msg) = parse_kernel_ts("[   12.345678] systemd[1]: Reached target basic");
        assert_eq!(ts, Some(12.345678));
        assert_eq!(msg, "systemd[1]: Reached target basic");
        let (ts, msg) = parse_kernel_ts("raw continuation fragment");
        assert_eq!(ts, None);
        assert_eq!(msg, "raw continuation fragment");
        // a bracketed-but-not-numeric prefix is NOT a timestamp
        let (ts, _) = parse_kernel_ts("[UFW BLOCK] IN=eth0");
        assert_eq!(ts, None);
    }

    #[test]
    fn since_filter_keeps_late_and_untimestamped() {
        assert!(passes_since(Some(20.0), Some(10.0)));
        assert!(!passes_since(Some(5.0), Some(10.0)));
        assert!(passes_since(None, Some(10.0)), "untimestamped always shown");
        assert!(passes_since(Some(5.0), None), "no filter = everything");
    }

    #[test]
    fn boot_json_is_one_object_covering_arm_and_power() {
        let r = ArmReport {
            node: "bc250-a".into(),
            ipxe_bootnum: 0,
            ipxe_label: "BC250 iPXE oneshot".into(),
            boot_next: Some(0),
            already_armed: false,
        };
        let v = boot_json(&r, "cycled", Some("plugctl cycle bc250-a"));
        assert_eq!(v["node"], "bc250-a");
        assert_eq!(v["armed"], serde_json::Value::Bool(true));
        assert_eq!(v["ipxe_entry"], "0000");
        assert_eq!(v["power"], "cycled");
        assert_eq!(v["power_cmd"], "plugctl cycle bc250-a");
        let v = boot_json(&r, "manual", None);
        assert_eq!(v["power"], "manual");
        assert!(v["power_cmd"].is_null());
    }

    #[test]
    fn console_line_json_is_ndjson_parseable() {
        let s = console_line_json(1_783_000_000, 42, Some(12.345678), "amdgpu: ring gfx ready");
        let v: serde_json::Value = serde_json::from_str(&s).expect("one valid JSON object");
        assert_eq!(v["ts_unix_ms"], 1_783_000_000_042u64);
        assert!((v["kernel_ts"].as_f64().unwrap() - 12.345678).abs() < 1e-9);
        assert_eq!(v["line"], "amdgpu: ring gfx ready");
        // untimestamped lines carry null, and hostile bytes are escaped
        let s = console_line_json(1, 0, None, "raw \"quoted\" fragment");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(v["kernel_ts"].is_null());
        assert_eq!(v["line"], "raw \"quoted\" fragment");
    }

    #[test]
    fn console_line_is_wallclock_prefixed() {
        // 12:34:56 UTC on some day
        let unix = 86_400 * 20_000 + 12 * 3600 + 34 * 60 + 56;
        assert_eq!(
            console_line(unix, 7, "[  1.0] hello"),
            "12:34:56.007 | [  1.0] hello"
        );
    }
}
