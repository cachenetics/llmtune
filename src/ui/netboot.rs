// SPDX-License-Identifier: GPL-2.0-only
//! The fleet/netboot dashboard ([N]): one interactive view that drives the
//! whole netboot loop without dropping to the CLI - the boot-server health
//! strip (services/firewall/exports/image/reachability, probed on a worker),
//! a selectable table of fleet nodes (reachability, served model, GPU
//! temp/power, last bench tok/s - all from the cockpit's existing poll data,
//! no extra IO), the discovered-but-unregistered boards with one-key
//! register, per-node arm/disarm, the "boot a board" flow (arm + power_cmd,
//! or the manual cold-cycle instruction), and a live netconsole pane.
//!
//! Style (house rule): colored SECTION HEADER lines only; plain body text.
//! Everything slow (the server probe, arm/boot SSH round-trips) runs on a
//! worker thread - the state machine here only folds results; the netconsole
//! receiver streams over a channel. The row/format/summary helpers are pure
//! and unit-tested; the draw is a thin fold over them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use super::widgets::*;
use super::{CockpitView, NetbootBoard, NetbootView, NodeCard};
use crate::config::{Config, Node, Transport};
use crate::fmt::fit;
use crate::netboot_node::{self, PowerPlan};
use crate::netboot_server::{Backend, ImageState, PortRule, ServerStatus, Verdict};

/// Console scrollback depth (lines kept for the pane).
const CONSOLE_KEEP: usize = 400;

// ---------------------------------------------------------------------------
// Pure row/format helpers (unit-tested; the draw is a fold over these).
// ---------------------------------------------------------------------------

/// What a dashboard row points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RowKind {
    /// Index into `cfg.nodes` / the fleet cards.
    Node(usize),
    /// Index into the netboot view's discovered boards (unregistered only -
    /// a registered board already appears as a fleet node).
    Board(usize),
}

/// The selectable rows: every fleet node, then every discovered board that is
/// NOT yet a fleet node (registered boards would be duplicates).
pub(super) fn row_targets(n_nodes: usize, boards: &[NetbootBoard]) -> Vec<RowKind> {
    let mut v: Vec<RowKind> = (0..n_nodes).map(RowKind::Node).collect();
    v.extend(
        boards
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.registered)
            .map(|(i, _)| RowKind::Board(i)),
    );
    v
}

/// One node row's cell strings (placeholders for anything unknown - a dead
/// node never shows stale numbers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NodeCells {
    pub name: String,
    pub state: String,
    pub served: String,
    pub temp: String,
    pub power: String,
    pub toks: String,
}

pub(super) fn node_cells(node: &Node, card: &NodeCard) -> NodeCells {
    let state = if !card.probed {
        "probing"
    } else if !card.status.reachable {
        "OFFLINE"
    } else if card.status.healthy {
        "llama up"
    } else if card.status.benchmarking {
        // The bench intentionally paused llama-server; not a fault.
        "benchmarking"
    } else {
        "llama down"
    };
    let (temp, power) = match &card.telem {
        Some(t) => (
            format!("{:.0}C", t.temp_c),
            t.power_w
                .map(|w| format!("{w:.1}W"))
                .unwrap_or_else(|| "-".into()),
        ),
        None => ("-".into(), "-".into()),
    };
    NodeCells {
        name: node.name.clone(),
        state: state.into(),
        served: card.status.served.clone().unwrap_or_else(|| "-".into()),
        temp,
        power,
        toks: card
            .status
            .last_gen_tok_s
            .map(|t| format!("{t:.1}"))
            .unwrap_or_else(|| "-".into()),
    }
}

/// Whether this node can be armed/booted from the dashboard (a remote ssh
/// board with a host; the control host itself is never armed).
pub(super) fn can_arm(node: &Node) -> Result<(), String> {
    if node.transport != Transport::Ssh {
        return Err(format!(
            "{}: the control host cannot be armed - arm/boot drive remote boards",
            node.name
        ));
    }
    if node.host.as_deref().unwrap_or("").is_empty() {
        return Err(format!("{}: no host configured", node.name));
    }
    Ok(())
}

/// One-line firewall rollup: "ufw: all 4 port(s) open", or the closed/unknown
/// ports called out.
pub(super) fn fw_summary(backend: Backend, fw: &[(PortRule, Option<bool>)]) -> String {
    if fw.is_empty() {
        return format!("{}: no rules required", backend.as_str());
    }
    let closed: Vec<String> = fw
        .iter()
        .filter(|(_, p)| *p == Some(false))
        .map(|(r, _)| r.token())
        .collect();
    let unknown = fw.iter().filter(|(_, p)| p.is_none()).count();
    if closed.is_empty() && unknown == 0 {
        return format!("{}: all {} port(s) open", backend.as_str(), fw.len());
    }
    let mut s = backend.as_str().to_string();
    if !closed.is_empty() {
        s.push_str(&format!(": NOT open {}", closed.join(", ")));
    }
    if unknown > 0 {
        s.push_str(&format!("  ({unknown} unknown - needs sudo)"));
    }
    s
}

/// One-line staged-image rollup.
pub(super) fn image_summary(img: &ImageState) -> String {
    match img {
        ImageState::Staged { id, init } => format!("[ok]   staged {id} (init= {init})"),
        ImageState::None => "[--]   no staged image (legacy kernel/initrd paths served)".into(),
        ImageState::Error(e) => format!("[fail] staged image fails verification: {e}"),
    }
}

/// One-line reachability rollup: mark + the diagnosis.
pub(super) fn reach_summary(st: &ServerStatus) -> String {
    let mark = match st.verdict {
        Verdict::Ok => "[ok]  ",
        Verdict::LocalOnlyUnproven => "[??]  ",
        _ => "[fail]",
    };
    format!("{mark} {}", st.detail)
}

// ---------------------------------------------------------------------------
// State: the dash, its background probe/action workers, the console pane.
// ---------------------------------------------------------------------------

/// The boot-server probe (services + firewall + exports + image +
/// off-host reachability). Runs `netboot_server::status_snapshot` on a
/// worker - it shells out and SSHes, so it must never run on the draw path.
pub(super) enum ServerProbe {
    /// No `[netboot]` in fleet.toml: the nodes table still works.
    Unconfigured,
    Running(JoinHandle<Result<ServerStatus, String>>),
    Ready(Box<ServerStatus>),
    Failed(String),
}

/// What a finished background action reports.
pub(super) struct TaskDone {
    pub headline: String,
    /// A multi-line note pinned into the dash (e.g. the manual cold-cycle
    /// instruction after arming a node with no power_cmd).
    pub note: Option<String>,
}

/// What a key press asks the cockpit to do with the dash.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum DashOutcome {
    Stay,
    Close,
    /// Close the dash and focus this fleet node in the cockpit (Enter on a
    /// node row - the swap/bench flow continues in the main view).
    FocusNode(usize),
    /// Register the discovered-but-unregistered boards as fleet nodes.
    Register,
}

/// The live netconsole pane: a UDP receiver on a worker thread, filtered to
/// the selected node's source IP, streaming rendered lines over a channel.
pub(super) struct ConsolePane {
    pub node: String,
    pub port: u16,
    pub lines: VecDeque<String>,
    rx: Receiver<String>,
    stop: Arc<AtomicBool>,
}

impl ConsolePane {
    /// Bind the netconsole port and start the receiver worker. Fails fast on
    /// an unresolvable node or an occupied port.
    pub(super) fn start(node: &Node, port: u16) -> Result<ConsolePane, String> {
        let want_ip = netboot_node::resolve_node_ip(node).map_err(|e| e.to_string())?;
        let sock = std::net::UdpSocket::bind(("0.0.0.0", port)).map_err(|e| {
            format!("binding udp :{port}: {e} (is another console/receiver listening?)")
        })?;
        sock.set_read_timeout(Some(Duration::from_millis(300)))
            .map_err(|e| e.to_string())?;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = [0u8; 65536];
            while !flag.load(Ordering::Relaxed) {
                let (n, src) = match sock.recv_from(&mut buf) {
                    Ok(v) => v,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut =>
                    {
                        continue;
                    }
                    Err(_) => break,
                };
                if src.ip() != want_ip {
                    continue; // several boards can ship to the same port
                }
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                for line in netboot_node::datagram_lines(&buf[..n]) {
                    let rendered =
                        netboot_node::console_line(now.as_secs(), now.subsec_millis(), &line);
                    if tx.send(rendered).is_err() {
                        return; // pane closed
                    }
                }
            }
        });
        Ok(ConsolePane {
            node: node.name.clone(),
            port,
            lines: VecDeque::new(),
            rx,
            stop,
        })
    }

    /// A pane with no worker (tests of the key routing / draw only).
    #[cfg(test)]
    pub(super) fn fake(node: &str) -> ConsolePane {
        let (_tx, rx) = std::sync::mpsc::channel();
        ConsolePane {
            node: node.into(),
            port: 6666,
            lines: VecDeque::new(),
            rx,
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Drain newly-received lines into the bounded scrollback.
    fn poll(&mut self) {
        while let Ok(l) = self.rx.try_recv() {
            if self.lines.len() == CONSOLE_KEEP {
                self.lines.pop_front();
            }
            self.lines.push_back(l);
        }
    }
}

impl Drop for ConsolePane {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// The fleet/netboot dashboard state (owned by the cockpit while [N] is open).
pub(super) struct NetbootDash {
    /// The netboot snapshot (server config + discovered boards); None when
    /// `[netboot]` is unconfigured - the nodes table still renders.
    pub(super) view: Option<NetbootView>,
    /// Selection over `row_targets(...)`.
    pub(super) sel: usize,
    pub(super) server: ServerProbe,
    /// One background action (arm/disarm/boot) at a time.
    task: Option<(String, JoinHandle<TaskDone>)>,
    /// [o] is confirm-armed for this cfg.nodes index: the next [o] boots it.
    pub(super) boot_confirm: Option<usize>,
    /// A pinned multi-line note (manual cold-cycle instructions).
    pub(super) note: Option<String>,
    pub(super) console: Option<ConsolePane>,
}

impl NetbootDash {
    pub(super) fn new(view: Option<NetbootView>) -> NetbootDash {
        NetbootDash {
            view,
            sel: 0,
            server: ServerProbe::Unconfigured,
            task: None,
            boot_confirm: None,
            note: None,
            console: None,
        }
    }

    fn boards(&self) -> &[NetbootBoard] {
        self.view
            .as_ref()
            .map(|v| v.boards.as_slice())
            .unwrap_or(&[])
    }

    fn console_port(&self) -> u16 {
        self.view
            .as_ref()
            .map(|v| v.console_port)
            .unwrap_or(netboot_node::DEFAULT_CONSOLE_PORT)
    }

    /// True while anything is in flight (drives the fast tick).
    pub(super) fn busy(&self) -> bool {
        self.task.is_some()
            || self.console.is_some()
            || matches!(self.server, ServerProbe::Running(_))
    }

    /// (Re)dispatch the boot-server probe on a worker.
    pub(super) fn spawn_server_probe(&mut self, cfg: &Config) {
        if cfg.netboot.is_none() {
            self.server = ServerProbe::Unconfigured;
            return;
        }
        if matches!(self.server, ServerProbe::Running(_)) {
            return;
        }
        let cfg = cfg.clone();
        self.server = ServerProbe::Running(std::thread::spawn(move || {
            crate::netboot_server::status_snapshot(&cfg).map_err(|e| format!("{e:#}"))
        }));
    }

    pub(super) fn clamp_sel(&mut self, n_nodes: usize) {
        let n = row_targets(n_nodes, self.boards()).len();
        if self.sel >= n {
            self.sel = n.saturating_sub(1);
        }
    }

    /// Fold finished workers + drain the console. Returns a headline for the
    /// status line when an action lands.
    pub(super) fn poll(&mut self) -> Option<String> {
        let mut headline = None;
        if matches!(&self.server, ServerProbe::Running(h) if h.is_finished()) {
            if let ServerProbe::Running(h) =
                std::mem::replace(&mut self.server, ServerProbe::Unconfigured)
            {
                self.server = match h.join() {
                    Ok(Ok(st)) => ServerProbe::Ready(Box::new(st)),
                    Ok(Err(e)) => ServerProbe::Failed(e),
                    Err(_) => ServerProbe::Failed("status probe panicked".into()),
                };
            }
        }
        if matches!(&self.task, Some((_, h)) if h.is_finished()) {
            if let Some((_, h)) = self.task.take() {
                match h.join() {
                    Ok(done) => {
                        headline = Some(done.headline);
                        if done.note.is_some() {
                            self.note = done.note;
                        }
                    }
                    Err(_) => headline = Some("netboot action panicked".into()),
                }
            }
        }
        if let Some(c) = self.console.as_mut() {
            c.poll();
        }
        headline
    }

    /// All key routing while the dash is open. Returns what the cockpit
    /// should do, plus an optional status-line message.
    pub(super) fn on_key(&mut self, code: KeyCode, cfg: &Config) -> (DashOutcome, Option<String>) {
        let rows = row_targets(cfg.nodes.len(), self.boards());
        // Any key other than a second [o] cancels an armed boot confirm.
        let confirm = self.boot_confirm.take();
        match code {
            KeyCode::Esc => {
                if self.console.is_some() {
                    self.console = None;
                    return (DashOutcome::Stay, Some("console closed".into()));
                }
                (DashOutcome::Close, None)
            }
            KeyCode::Char('q') => (DashOutcome::Close, None),
            KeyCode::Up => {
                self.sel = self.sel.saturating_sub(1);
                (DashOutcome::Stay, None)
            }
            KeyCode::Down => {
                if self.sel + 1 < rows.len() {
                    self.sel += 1;
                }
                (DashOutcome::Stay, None)
            }
            KeyCode::Enter => match rows.get(self.sel) {
                Some(RowKind::Node(i)) => (DashOutcome::FocusNode(*i), None),
                Some(RowKind::Board(_)) => (DashOutcome::Register, None),
                None => (DashOutcome::Stay, None),
            },
            KeyCode::Char('r') | KeyCode::Char('R') => (DashOutcome::Register, None),
            KeyCode::Char('s') => {
                if self.view.is_none() {
                    return (
                        DashOutcome::Stay,
                        Some("netboot not configured - add [netboot] to fleet.toml".into()),
                    );
                }
                self.spawn_server_probe(cfg);
                (
                    DashOutcome::Stay,
                    Some(
                        "re-probing boot server (services/firewall/exports/reachability)...".into(),
                    ),
                )
            }
            KeyCode::Char('a') => (DashOutcome::Stay, self.start_arm(cfg, &rows, false)),
            KeyCode::Char('d') => (DashOutcome::Stay, self.start_arm(cfg, &rows, true)),
            KeyCode::Char('o') => (DashOutcome::Stay, self.boot_key(cfg, &rows, confirm)),
            KeyCode::Char('c') => (DashOutcome::Stay, self.toggle_console(cfg, &rows)),
            _ => (DashOutcome::Stay, None),
        }
    }

    /// The selected row's node, if it is a fleet node.
    fn sel_node<'a>(&self, cfg: &'a Config, rows: &[RowKind]) -> Option<&'a Node> {
        match rows.get(self.sel)? {
            RowKind::Node(i) => cfg.nodes.get(*i),
            RowKind::Board(_) => None,
        }
    }

    /// [a]/[d]: arm (or disarm) the selected node's one-shot iPXE BootNext on
    /// a worker (an SSH round-trip; never inline).
    fn start_arm(&mut self, cfg: &Config, rows: &[RowKind], disarm: bool) -> Option<String> {
        let verb = if disarm { "disarm" } else { "arm" };
        let Some(node) = self.sel_node(cfg, rows) else {
            return Some(format!(
                "{verb}: select a fleet node (register boards first)"
            ));
        };
        if let Err(e) = can_arm(node) {
            return Some(e);
        }
        if self.task.is_some() {
            return Some("an action is already running...".into());
        }
        let n = node.clone();
        self.task = Some((
            format!("{verb}ing {}...", n.name),
            std::thread::spawn(move || match netboot_node::arm_node(&n, disarm) {
                Ok((rep, action)) => TaskDone {
                    headline: format!(
                        "{}: {action} - iPXE entry Boot{:04X} \"{}\"",
                        n.name, rep.ipxe_bootnum, rep.ipxe_label
                    ),
                    note: None,
                },
                Err(e) => TaskDone {
                    headline: format!("{}: {verb} failed: {e}", n.name),
                    note: None,
                },
            }),
        ));
        Some(format!("{verb}ing {} (one-shot BootNext)...", node.name))
    }

    /// [o]: the boot-a-board flow. First press arms the confirm (describing
    /// the power step); the second dispatches arm + power on a worker. A
    /// power_cmd hook cold-cycles automatically; without one the board is
    /// armed and the manual cold-cycle instruction is pinned as a note.
    fn boot_key(
        &mut self,
        cfg: &Config,
        rows: &[RowKind],
        confirm: Option<usize>,
    ) -> Option<String> {
        let Some(node) = self.sel_node(cfg, rows) else {
            return Some("boot: select a fleet node (register boards first)".into());
        };
        if let Err(e) = can_arm(node) {
            return Some(e);
        }
        if self.task.is_some() {
            return Some("an action is already running...".into());
        }
        let idx = match rows.get(self.sel) {
            Some(RowKind::Node(i)) => *i,
            _ => return None,
        };
        if confirm != Some(idx) {
            self.boot_confirm = Some(idx);
            let power = match netboot_node::power_plan(node.power_cmd.as_deref()) {
                PowerPlan::Cmd(c) => format!("then COLD-CYCLE via power_cmd `{c}`"),
                PowerPlan::Manual => "no power_cmd - you will cold-cycle by hand".into(),
            };
            return Some(format!(
                "boot {}? arms iPXE one-shot, {power} - press o again to confirm",
                node.name
            ));
        }
        let n = node.clone();
        self.task = Some((
            format!("booting {}...", n.name),
            std::thread::spawn(move || {
                match netboot_node::arm_node(&n, false) {
                Err(e) => TaskDone {
                    headline: format!("{}: arm failed (no power action taken): {e}", n.name),
                    note: None,
                },
                Ok(_) => match netboot_node::power_plan(n.power_cmd.as_deref()) {
                    PowerPlan::Cmd(cmd) => match netboot_node::run_power_cmd(&n.name, &cmd) {
                        Ok(()) => TaskDone {
                            headline: format!(
                                "{}: armed + power_cmd ran - cold-cycling into iPXE ([c] console to watch)",
                                n.name
                            ),
                            note: None,
                        },
                        Err(e) => TaskDone {
                            headline: format!("{}: armed, but power_cmd failed: {e}", n.name),
                            note: None,
                        },
                    },
                    PowerPlan::Manual => TaskDone {
                        headline: format!(
                            "{}: armed - cold power-cycle it by hand now ([c] console to watch)",
                            n.name
                        ),
                        note: Some(netboot_node::manual_power_instructions(&n.name)),
                    },
                },
            }
            }),
        ));
        Some(format!("booting {} (arm + power)...", node.name))
    }

    /// [c]: toggle the netconsole pane for the selected node.
    fn toggle_console(&mut self, cfg: &Config, rows: &[RowKind]) -> Option<String> {
        if self.console.is_some() {
            self.console = None;
            return Some("console closed".into());
        }
        let Some(node) = self.sel_node(cfg, rows) else {
            return Some("console: select a fleet node".into());
        };
        let port = self.console_port();
        match ConsolePane::start(node, port) {
            Ok(c) => {
                let name = node.name.clone();
                self.console = Some(c);
                Some(format!("console: listening on udp :{port} for {name}"))
            }
            Err(e) => Some(format!("console: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Draw (thin over the pure helpers; headers colored, body plain).
// ---------------------------------------------------------------------------

/// A colored SECTION HEADER line (the only body-area color besides the
/// selection cursor, per the house style).
fn header(text: String) -> Line<'static> {
    Line::from(Span::styled(
        text,
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    ))
}

fn plain(text: String) -> Line<'static> {
    Line::from(Span::raw(text))
}

fn dimmed(text: String) -> Line<'static> {
    Line::from(Span::styled(text, Style::default().fg(DIM)))
}

/// The fleet/netboot dashboard overlay.
pub(super) fn draw_netboot(f: &mut Frame, area: Rect, app: &CockpitView) {
    let Some(dash) = app.netboot else {
        return;
    };
    let w = area.width.saturating_sub(2).min(102);
    let h = area.height.saturating_sub(2).max(8);
    let inner = super::overlays::open_overlay(f, area, "fleet / netboot", w, h);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let iw = inner.width as usize;

    // The console pane (when open) gets the bottom of the box.
    let console_h = dash
        .console
        .as_ref()
        .map(|_| (inner.height / 2).clamp(5, 14))
        .unwrap_or(0);
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(4), Constraint::Length(console_h)])
        .split(inner);

    let mut lines: Vec<Line> = Vec::new();
    let mut sel_end = 0usize;

    // -- boot server ---------------------------------------------------------
    match &dash.view {
        None => {
            lines.push(header("BOOT SERVER".into()));
            lines.push(dimmed(
                "  netboot not configured (add [netboot] to fleet.toml) - nodes below still drivable"
                    .into(),
            ));
        }
        Some(v) => {
            lines.push(header(format!(
                "BOOT SERVER  {}  {}  iface {}",
                v.server_ip, v.http_base, v.interface
            )));
            match &dash.server {
                ServerProbe::Unconfigured | ServerProbe::Running(_) => {
                    // The always-cheap probe (unit states) while the full one runs.
                    let svcs: Vec<String> = v
                        .services
                        .iter()
                        .map(|(s, st)| {
                            format!("{s} {}", if st == "active" { "[ok]" } else { "[--]" })
                        })
                        .collect();
                    lines.push(plain(format!("  services  {}", svcs.join("   "))));
                    lines.push(dimmed(
                        "  probing firewall/exports/image/reachability...".into(),
                    ));
                }
                ServerProbe::Failed(e) => {
                    for seg in
                        wrap_display(&format!("status probe failed: {e}"), iw.saturating_sub(2))
                    {
                        lines.push(plain(format!("  {seg}")));
                    }
                }
                ServerProbe::Ready(st) => {
                    let svcs: Vec<String> = st
                        .services
                        .iter()
                        .map(|(s, a)| {
                            format!("{s} {}", if a == "active" { "[ok]" } else { "[--]" })
                        })
                        .collect();
                    lines.push(plain(format!("  services  {}", svcs.join("   "))));
                    lines.push(plain(format!(
                        "  firewall  {}",
                        fw_summary(st.backend, &st.firewall)
                    )));
                    lines.push(plain(format!(
                        "  exports   {} -> {} (ro): {}",
                        st.models_dir,
                        st.lan_cidr,
                        if st.exports_ok {
                            "exported"
                        } else {
                            "NOT exported"
                        }
                    )));
                    lines.push(plain(format!("  image     {}", image_summary(&st.image))));
                    let mut reach = wrap_display(&reach_summary(st), iw.saturating_sub(12));
                    if let Some(first) = reach.first() {
                        lines.push(plain(format!("  reach     {first}")));
                    }
                    for seg in reach.drain(..).skip(1).take(2) {
                        lines.push(dimmed(format!("            {seg}")));
                    }
                }
            }
        }
    }
    lines.push(plain(String::new()));

    // -- nodes table ----------------------------------------------------------
    let rows = row_targets(app.fleet.cfg.nodes.len(), dash.boards());
    lines.push(header(format!("NODES ({})", app.fleet.cfg.nodes.len())));
    // Column budget: name gets what the fixed metric columns leave.
    let name_w = 16usize.max(
        app.fleet
            .cfg
            .nodes
            .iter()
            .map(|n| dwidth(&n.name))
            .max()
            .unwrap_or(0),
    );
    // State column fits the widest state ("benchmarking", 12) plus a gap.
    let served_w = iw
        .saturating_sub(4 + name_w + 2 + 13 + 5 + 7 + 7)
        .clamp(8, 44);
    lines.push(dimmed(format!(
        "    {:<name_w$}  {:<13}{:<served_w$} {:>4} {:>6} {:>6}",
        "name", "state", "served", "temp", "power", "tok/s"
    )));
    for (ri, row) in rows.iter().enumerate() {
        let selected = ri == dash.sel;
        match row {
            RowKind::Node(i) => {
                let (Some(node), Some(card)) =
                    (app.fleet.cfg.nodes.get(*i), app.fleet.cards.get(*i))
                else {
                    continue;
                };
                let c = node_cells(node, card);
                let text = format!(
                    "{:<name_w$}  {:<13}{:<served_w$} {:>4} {:>6} {:>6}",
                    fit(&c.name, name_w),
                    c.state,
                    fit(&c.served, served_w),
                    c.temp,
                    c.power,
                    c.toks
                );
                lines.push(row_line(selected, text));
            }
            RowKind::Board(_) => {} // drawn under DISCOVERED below
        }
        if selected && matches!(row, RowKind::Node(_)) {
            sel_end = lines.len();
        }
    }

    // -- discovered boards ------------------------------------------------
    if let Some(v) = &dash.view {
        let new = v.unregistered();
        lines.push(plain(String::new()));
        lines.push(header(format!(
            "DISCOVERED ({} board(s), {new} new)",
            v.boards.len()
        )));
        if v.boards.is_empty() {
            lines.push(dimmed(format!("    none seen yet ({})", v.leases_path)));
        }
        for b in v.boards.iter().filter(|b| b.registered) {
            lines.push(dimmed(format!(
                "    {:<16} {:<16} {}  registered (in NODES above)",
                b.name, b.ip, b.mac
            )));
        }
        for (ri, row) in rows.iter().enumerate() {
            let RowKind::Board(bi) = row else { continue };
            let Some(b) = v.boards.get(*bi) else { continue };
            let selected = ri == dash.sel;
            let text = format!(
                "{:<16} {:<16} {}  new - [r] or Enter registers",
                b.name, b.ip, b.mac
            );
            lines.push(row_line(selected, text));
            if selected {
                sel_end = lines.len();
            }
        }
    }

    // -- pinned note (manual cold-cycle instructions) -----------------------
    if let Some(note) = &dash.note {
        lines.push(plain(String::new()));
        lines.push(header("NOTE".into()));
        for l in note.lines() {
            for seg in wrap_display(l.trim_end(), iw.saturating_sub(2)) {
                lines.push(plain(format!("  {seg}")));
            }
        }
    }

    // -- in-flight action ----------------------------------------------------
    if let Some((what, _)) = &dash.task {
        lines.push(plain(String::new()));
        lines.push(Line::from(Span::styled(
            format!(
                "{} {what}",
                spinner_ch(app.fleet.started.elapsed().as_millis())
            ),
            Style::default().fg(WARN).add_modifier(Modifier::BOLD),
        )));
    }

    // -- key hints ------------------------------------------------------------
    lines.push(plain(String::new()));
    lines.push(key_line(&[
        ("Enter", "drive node"),
        ("a", "arm"),
        ("d", "disarm"),
        ("o", "boot (x2)"),
        ("c", "console"),
        ("r", "register"),
        ("s", "recheck server"),
        ("Esc", "close"),
    ]));

    // Scroll so the selected row stays visible.
    let first = sel_end.saturating_sub(split[0].height as usize);
    f.render_widget(Paragraph::new(lines).scroll((first as u16, 0)), split[0]);

    // -- console pane ----------------------------------------------------------
    if let Some(c) = &dash.console {
        let r = split[1];
        if r.height >= 2 {
            let mut clines: Vec<Line> = vec![header(format!(
                "CONSOLE {}  udp :{}  (c or Esc closes)",
                c.node, c.port
            ))];
            let budget = (r.height as usize).saturating_sub(1);
            if c.lines.is_empty() {
                clines.push(dimmed(
                    "  waiting for netconsole datagrams (nothing = board off, pre-network, \
                     or netconsole pointed elsewhere)"
                        .into(),
                ));
            }
            let start = c.lines.len().saturating_sub(budget);
            for l in c.lines.iter().skip(start) {
                clines.push(plain(fit(l, iw)));
            }
            f.render_widget(Paragraph::new(clines), r);
        }
    }
}

/// A selectable row: cursor + bold when selected, plain otherwise.
fn row_line(selected: bool, text: String) -> Line<'static> {
    if selected {
        Line::from(vec![
            Span::styled(
                "  ▸ ".to_string(),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(text, Style::default().add_modifier(Modifier::BOLD)),
        ])
    } else {
        Line::from(vec![Span::raw("    "), Span::raw(text)])
    }
}

// ---------------------------------------------------------------------------
// Tests (the pure state/formatting layer).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netboot_server::{Proto, ReachReport};

    fn board(name: &str, registered: bool) -> NetbootBoard {
        NetbootBoard {
            name: name.into(),
            ip: "192.168.1.120".into(),
            mac: "58:11:22:aa:bb:cc".into(),
            registered,
        }
    }

    fn ssh_node(name: &str) -> Node {
        Node {
            name: name.into(),
            host: Some("192.0.2.10".into()),
            transport: Transport::Ssh,
            ssh_user: Some("root".into()),
            ssh_key: None,
            models_dir: "/var/lib/llmtune/models".into(),
            llama_unit: "llama-server.service".into(),
            llama_url: "http://127.0.0.1:8080".into(),
            power_cmd: None,
        }
    }

    fn local_node() -> Node {
        let mut n = ssh_node("localhost");
        n.transport = Transport::Local;
        n.host = None;
        n
    }

    fn cfg_with(nodes: Vec<Node>) -> Config {
        Config {
            nodes,
            clusters: vec![],
            netboot: None,
        }
    }

    #[test]
    fn rows_are_nodes_then_unregistered_boards_only() {
        let boards = vec![board("bc250-01", true), board("bc250-02", false)];
        let rows = row_targets(2, &boards);
        assert_eq!(
            rows,
            vec![RowKind::Node(0), RowKind::Node(1), RowKind::Board(1)],
            "registered boards must not duplicate their fleet-node row"
        );
        assert!(row_targets(0, &[]).is_empty());
    }

    #[test]
    fn node_cells_format_and_placeholders() {
        let node = ssh_node("bc250-005363");
        // healthy, fully-reported card
        let mut card = NodeCard::new("bc250-005363");
        card.probed = true;
        card.status = crate::transport::NodeStatus {
            name: "bc250-005363".into(),
            reachable: true,
            healthy: true,
            benchmarking: false,
            served: Some("Qwen3-30B-A3B-IQ2.gguf".into()),
            models: 3,
            last_model: None,
            last_gen_tok_s: Some(41.23),
        };
        card.telem = Some(crate::telemetry::Telemetry {
            gfxclk_mhz: 2230,
            uclk_mhz: 450,
            temp_c: 61.4,
            power_w: Some(38.06),
            ..Default::default()
        });
        let c = node_cells(&node, &card);
        assert_eq!(c.state, "llama up");
        assert_eq!(c.served, "Qwen3-30B-A3B-IQ2.gguf");
        assert_eq!(c.temp, "61C");
        assert_eq!(c.power, "38.1W");
        assert_eq!(c.toks, "41.2");
        // unprobed / unreachable cards: placeholders, never stale numbers
        let fresh = NodeCard::new("x");
        let c = node_cells(&node, &fresh);
        assert_eq!(c.state, "probing");
        assert_eq!((c.served.as_str(), c.temp.as_str()), ("-", "-"));
        let mut dead = NodeCard::new("x");
        dead.probed = true;
        let c = node_cells(&node, &dead);
        assert_eq!(c.state, "OFFLINE");
        assert_eq!((c.power.as_str(), c.toks.as_str()), ("-", "-"));
    }

    #[test]
    fn node_cells_bench_pause_is_not_down() {
        let node = ssh_node("bc250-005363");
        let mut card = NodeCard::new("bc250-005363");
        card.probed = true;
        card.status.reachable = true;
        card.status.healthy = false;
        card.status.benchmarking = true;
        // Down + marker: the intentional pause, not a fault.
        assert_eq!(node_cells(&node, &card).state, "benchmarking");
        // Down without the marker stays a real down.
        card.status.benchmarking = false;
        assert_eq!(node_cells(&node, &card).state, "llama down");
        // A healthy server wins over a stale marker.
        card.status.benchmarking = true;
        card.status.healthy = true;
        assert_eq!(node_cells(&node, &card).state, "llama up");
    }

    #[test]
    fn can_arm_refuses_local_and_hostless() {
        assert!(can_arm(&ssh_node("b")).is_ok());
        assert!(can_arm(&local_node()).unwrap_err().contains("control host"));
        let mut hostless = ssh_node("b");
        hostless.host = None;
        assert!(can_arm(&hostless).unwrap_err().contains("no host"));
    }

    #[test]
    fn fw_summary_rollups() {
        let r = |port, ok| {
            (
                PortRule {
                    port,
                    proto: Proto::Tcp,
                    what: "http",
                },
                ok,
            )
        };
        assert_eq!(
            fw_summary(Backend::Ufw, &[r(8090, Some(true)), r(2049, Some(true))]),
            "ufw: all 2 port(s) open"
        );
        let s = fw_summary(Backend::Nftables, &[r(8090, Some(false)), r(2049, None)]);
        assert!(s.contains("NOT open 8090/tcp"), "{s}");
        assert!(s.contains("1 unknown"), "{s}");
        assert_eq!(fw_summary(Backend::None, &[]), "none: no rules required");
    }

    #[test]
    fn image_and_reach_summaries() {
        assert!(image_summary(&ImageState::Staged {
            id: "bc250-1".into(),
            init: "/nix/store/x/init".into()
        })
        .contains("staged bc250-1"));
        assert!(image_summary(&ImageState::None).contains("no staged image"));
        assert!(image_summary(&ImageState::Error("bad".into())).contains("[fail]"));
        let st = ServerStatus {
            server_ip: "192.168.1.91".into(),
            http_base: "http://192.168.1.91:8090".into(),
            lan_cidr: "192.168.1.0/24".into(),
            models_dir: "/srv/models".into(),
            services: vec![],
            backend: Backend::None,
            firewall: vec![],
            exports_ok: true,
            image: ImageState::None,
            reach: ReachReport::default(),
            verdict: Verdict::Ok,
            detail: "reachable from bc250-005363".into(),
        };
        assert!(reach_summary(&st).starts_with("[ok]"));
        let st2 = ServerStatus {
            verdict: Verdict::LocalOnlyUnproven,
            detail: "no off-host proof".into(),
            ..st
        };
        assert!(reach_summary(&st2).starts_with("[??]"));
    }

    #[test]
    fn boot_flow_is_two_step_and_cancellable() {
        let cfg = cfg_with(vec![local_node()]);
        let mut d = NetbootDash::new(None);
        // A LOCAL node refuses to arm/boot (no thread ever spawns).
        let (out, msg) = d.on_key(KeyCode::Char('o'), &cfg);
        assert_eq!(out, DashOutcome::Stay);
        assert!(msg.unwrap().contains("control host"));
        assert_eq!(d.boot_confirm, None);
        // A remote node: first [o] arms the confirm and names the power plan.
        let mut remote = ssh_node("bc250-x");
        remote.power_cmd = Some("plugctl cycle x".into());
        let cfg = cfg_with(vec![remote]);
        let (_, msg) = d.on_key(KeyCode::Char('o'), &cfg);
        assert_eq!(d.boot_confirm, Some(0));
        let m = msg.unwrap();
        assert!(m.contains("press o again"), "{m}");
        assert!(m.contains("plugctl cycle x"), "{m}");
        // any OTHER key cancels the armed confirm
        let _ = d.on_key(KeyCode::Down, &cfg);
        assert_eq!(d.boot_confirm, None, "non-[o] key cancels the confirm");
    }

    #[test]
    fn arm_and_console_refuse_board_rows_and_local() {
        let mut view = NetbootView {
            server_ip: "192.168.1.91".into(),
            http_base: "http://192.168.1.91:8090".into(),
            interface: "enp6s0".into(),
            leases_path: "/tmp/leases".into(),
            console_port: 6666,
            services: vec![],
            boards: vec![board("bc250-new", false)],
        };
        view.boards[0].registered = false;
        let cfg = cfg_with(vec![local_node()]);
        let mut d = NetbootDash::new(Some(view));
        // move onto the board row (row 1)
        let _ = d.on_key(KeyCode::Down, &cfg);
        assert_eq!(d.sel, 1);
        let (_, msg) = d.on_key(KeyCode::Char('a'), &cfg);
        assert!(msg.unwrap().contains("register boards first"));
        // Enter on the board row asks the cockpit to register
        let (out, _) = d.on_key(KeyCode::Enter, &cfg);
        assert_eq!(out, DashOutcome::Register);
        // console on the local (hostless) row: refused with a message
        let _ = d.on_key(KeyCode::Up, &cfg);
        let (_, msg) = d.on_key(KeyCode::Char('c'), &cfg);
        assert!(msg.unwrap().contains("no host"));
    }

    #[test]
    fn esc_closes_console_before_dash_and_enter_focuses_node() {
        let cfg = cfg_with(vec![ssh_node("bc250-x")]);
        let mut d = NetbootDash::new(None);
        d.console = Some(ConsolePane::fake("bc250-x"));
        let (out, _) = d.on_key(KeyCode::Esc, &cfg);
        assert_eq!(out, DashOutcome::Stay, "first Esc closes the console pane");
        assert!(d.console.is_none());
        let (out, _) = d.on_key(KeyCode::Esc, &cfg);
        assert_eq!(out, DashOutcome::Close, "second Esc closes the dash");
        let (out, _) = d.on_key(KeyCode::Enter, &cfg);
        assert_eq!(
            out,
            DashOutcome::FocusNode(0),
            "Enter on a node row drives it in the cockpit"
        );
    }

    #[test]
    fn selection_clamps_and_never_overruns() {
        let cfg = cfg_with(vec![ssh_node("a"), ssh_node("b")]);
        let mut d = NetbootDash::new(None);
        for _ in 0..10 {
            let _ = d.on_key(KeyCode::Down, &cfg);
        }
        assert_eq!(d.sel, 1, "Down clamps to the last row");
        d.sel = 5;
        d.clamp_sel(cfg.nodes.len());
        assert_eq!(d.sel, 1, "clamp_sel repairs an out-of-range selection");
        d.clamp_sel(0);
        assert_eq!(d.sel, 0);
    }
}
