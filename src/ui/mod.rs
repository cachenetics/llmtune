// SPDX-License-Identifier: GPL-2.0-only
//! Interactive TUI - the unified cockpit. ONE screen for any node count:
//! the node telemetry rail (left, the selected node's card expanded with its
//! full detail), the focused node's model list (middle), and the selected
//! model's card (right) are all visible at the same time; the model card is
//! the elastic pane that grows with the terminal. Tab / Left / Right move
//! focus between the nodes and models panes, Up/Down moves within the
//! focused pane, Enter acts on the selection (nodes: focus the models pane;
//! models: load the selection on the focused node). Everything runs through
//! `NodeTransport`, so a remote (SSH) node is driven exactly like the local
//! one, and every probe/list/load runs on a worker thread so a down node can
//! never freeze the cockpit.
//!
//! Layout: this module holds the app state + the event loop; `widgets` is the
//! shared look/helper library; `fleet` holds the cockpit layout + rail
//! draws, `node` the model list/card draws, `overlays` the modal draws.

mod fleet;
mod netboot;
mod node;
mod overlays;
mod widgets;

use std::io::stdout;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;

use crate::bench;
use crate::build;
use crate::config::{ClusterCfg, Config, Node, Transport};
use crate::doctor;
use crate::endpoint::Endpoint;
use crate::history::Record;
use crate::nodeops;
use crate::profile;
use crate::telemetry::Telemetry;
use crate::transport::{self, ModelInfo, NodeStatus, SwapReport};

use fleet::draw_cockpit;
use widgets::split_flags;
#[cfg(test)]
use widgets::{dwidth, wrap_display};

/// Best-effort terminal teardown: leave raw mode + the alternate screen. Safe to
/// call more than once, and from a panic hook (so a panic mid-draw doesn't leave
/// the user's terminal in raw mode with no echo).
fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = stdout().execute(LeaveAlternateScreen);
}

pub fn run(cfg: Config, config_path: Option<String>) -> Result<()> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    // Chain a panic hook that restores the terminal before the default handler
    // prints the panic - otherwise a panic in the draw/event loop corrupts the tty.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        prev(info);
    }));
    let mut term = Terminal::new(CrosstermBackend::new(stdout()))?;
    let res = drive(&mut term, cfg, config_path);
    restore_terminal();
    let _ = std::panic::take_hook(); // drop our hook on the clean path
    res
}

fn drive<B: Backend>(
    term: &mut Terminal<B>,
    cfg: Config,
    config_path: Option<String>,
) -> Result<()> {
    // One unified cockpit for ANY node count: nodes rail + model list + model
    // card, all on screen together. Key routing lives in CockpitApp::on_key.
    let mut app = CockpitApp::load(cfg, config_path);
    loop {
        app.fleet.poll();
        app.node.poll();
        app.poll_fetch();
        app.poll_netboot();
        app.fold_status();
        term.draw(|f| draw_cockpit(f, &app.view()))?;
        // Fast tick while anything animates: a node op, or the netboot dash
        // with an in-flight action / live console stream.
        let netboot_busy = app.netboot.as_ref().is_some_and(|d| d.busy());
        let tick = if app.node.busy() || netboot_busy {
            90
        } else {
            300
        };
        if !event::poll(Duration::from_millis(tick))? {
            continue;
        }
        let Event::Key(k) = event::read()? else {
            continue;
        };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        match app.on_key(k.code) {
            KeyOutcome::Continue => {}
            KeyOutcome::Quit => return Ok(()),
            KeyOutcome::BenchAll => run_bench_all(term, &mut app)?,
        }
    }
}

/// The confirmed bench-all: bench every node sequentially, redrawing the
/// status line per node (deliberately blocking - it was confirmed twice).
fn run_bench_all<B: Backend>(term: &mut Terminal<B>, app: &mut CockpitApp) -> Result<()> {
    let nodes = app.fleet.cfg.nodes.clone();
    let mut failed = 0usize;
    for (i, n) in nodes.iter().enumerate() {
        app.fleet.status = format!("benchmarking {} ({}/{})...", n.name, i + 1, nodes.len());
        app.fold_status();
        term.draw(|f| draw_cockpit(f, &app.view()))?;
        if transport::for_node(n)
            .and_then(|t| t.bench(&bench::BenchSpec::default()))
            .is_err()
        {
            failed += 1;
        }
    }
    // re-poll every node (async) + refetch the models pane, headline the result
    app.fleet.refresh();
    app.queue_fetch();
    app.fleet.status = if failed == 0 {
        format!("bench-all complete ({} nodes)", nodes.len())
    } else {
        format!("bench-all: {}/{} node(s) failed", failed, nodes.len())
    };
    Ok(())
}

// ---------------------------------------------------------------------------
// Fleet view
// ---------------------------------------------------------------------------

/// How often each fleet node's status + telemetry are re-polled. Every poll
/// runs on a one-shot worker thread (an SSH round-trip per call on a remote
/// node), never inline in the event loop.
const FLEET_POLL_INTERVAL: Duration = Duration::from_secs(5);
/// An in-flight poll older than this is presumed lost (worker died without
/// reporting); the node becomes eligible for a fresh dispatch.
const FLEET_POLL_TIMEOUT: Duration = Duration::from_secs(30);

/// Sparkline history depth per node/metric: 60 samples at the 5 s poll cadence
/// is ~5 minutes of trend, enough to see a slow climb into thermal-wedge
/// territory. Bounded per node, so a 12-node rack holds at most
/// 12 x 3 x 60 f64s.
const SPARK_SAMPLES: usize = 60;

/// A bounded ring buffer of the last `cap` samples (per-node sparkline
/// history). Non-finite samples are dropped at the door so the min/max fold
/// (and the sparkline normalization) can never see a NaN.
pub(super) struct Ring {
    buf: std::collections::VecDeque<f64>,
    cap: usize,
}

impl Ring {
    fn new(cap: usize) -> Ring {
        Ring {
            buf: std::collections::VecDeque::with_capacity(cap),
            cap: cap.max(1),
        }
    }

    pub(super) fn push(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(v);
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.buf.len()
    }

    pub(super) fn last(&self) -> Option<f64> {
        self.buf.back().copied()
    }

    /// (min, max) over the buffer; None when empty.
    pub(super) fn min_max(&self) -> Option<(f64, f64)> {
        self.buf
            .iter()
            .fold(None, |acc: Option<(f64, f64)>, &v| match acc {
                None => Some((v, v)),
                Some((mn, mx)) => Some((mn.min(v), mx.max(v))),
            })
    }

    /// The last `n` samples, oldest first (what a width-limited sparkline shows).
    pub(super) fn tail(&self, n: usize) -> Vec<f64> {
        self.buf
            .iter()
            .copied()
            .skip(self.buf.len().saturating_sub(n))
            .collect()
    }
}

/// Event-log depth: the last ~50 state transitions, enough to answer "what
/// happened while I wasn't looking" without growing unbounded.
const EVENT_LOG_CAP: usize = 50;

/// Event severity - the message color in the [E]vents overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EventSev {
    /// Recovery / back-to-normal.
    Info,
    /// Degraded but running (llama down, slow).
    Warn,
    /// Crossing temp_crit / unreachable.
    Crit,
}

/// One logged fleet event: `14:32  node-03  81°C (crit)`.
pub(super) struct FleetEvent {
    /// Unix seconds (rendered as HH:MM).
    pub(super) ts: u64,
    pub(super) node: String,
    pub(super) msg: String,
    pub(super) sev: EventSev,
}

/// The bounded fleet event log. Appended ONLY by the poll fold on a state
/// TRANSITION (edge-triggered: a node sitting at 85 C logs once when it
/// crosses, not every poll).
pub(super) struct EventLog {
    buf: std::collections::VecDeque<FleetEvent>,
    cap: usize,
}

impl EventLog {
    fn new(cap: usize) -> EventLog {
        EventLog {
            buf: std::collections::VecDeque::with_capacity(cap),
            cap: cap.max(1),
        }
    }

    pub(super) fn push(&mut self, ev: FleetEvent) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(ev);
    }

    pub(super) fn len(&self) -> usize {
        self.buf.len()
    }

    /// The last `n` events, oldest first (chronological read).
    pub(super) fn tail(&self, n: usize) -> impl Iterator<Item = &FleetEvent> {
        self.buf.iter().skip(self.buf.len().saturating_sub(n))
    }
}

/// A node's alert state vs the configured thresholds (see
/// `fleet::alert_flags`). Stored per card so the poll fold can edge-detect
/// transitions into/out of each condition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AlertFlags {
    pub(super) hot: bool,
    pub(super) unreachable: bool,
    pub(super) down: bool,
    pub(super) slow: bool,
}

impl AlertFlags {
    pub(super) fn any(self) -> bool {
        self.hot || self.unreachable || self.down || self.slow
    }
}

/// Fold one node's before/after alert flags into the event log: rising edges
/// log the condition, and a previously-alerting node going fully clear logs
/// one "recovered". Pure (caller supplies the timestamp), so it is testable
/// without a clock.
fn fold_alert_events(
    events: &mut EventLog,
    ts: u64,
    name: &str,
    prev: AlertFlags,
    cur: AlertFlags,
    card: &NodeCard,
) {
    let mut push = |sev: EventSev, msg: String| {
        events.push(FleetEvent {
            ts,
            node: name.to_string(),
            msg,
            sev,
        });
    };
    if cur.hot && !prev.hot {
        let t = card.telem.as_ref().map(|t| t.temp_c).unwrap_or(0.0);
        push(EventSev::Crit, format!("{t:.0}°C (crit)"));
    }
    if cur.unreachable && !prev.unreachable {
        push(EventSev::Crit, "unreachable".into());
    }
    if cur.down && !prev.down {
        push(EventSev::Warn, "llama down (model loaded)".into());
    }
    if cur.slow && !prev.slow {
        let t = card.status.last_gen_tok_s.unwrap_or(0.0);
        push(EventSev::Warn, format!("slow: {t:.1} t/s"));
    }
    if prev.any() && !cur.any() {
        push(EventSev::Info, "recovered".into());
    }
}

/// One node's efficiency-bench outcome ([F]): gen tok/s + average socket watts
/// sampled across the run. `err` = the bench failed on that node.
pub(super) struct EffResult {
    pub(super) tok_s: Option<f64>,
    pub(super) avg_w: Option<f64>,
    pub(super) err: Option<String>,
}

/// The fleet efficiency bench: one worker thread per node (each box has its
/// own GPU, so cross-node parallelism is safe), results folded by poll(),
/// ranked overlay opened when the last worker lands. Indexed by node.
pub(super) enum EffBench {
    Idle,
    Running { results: Vec<Option<EffResult>> },
    Done(Vec<Option<EffResult>>),
}

/// One rail card: the latest background-poll results for a node.
struct NodeCard {
    status: NodeStatus,
    /// False until the first poll lands (renders as probing, not OFFLINE).
    probed: bool,
    telem: Option<Telemetry>,
    /// Consecutive polls without telemetry (>= TELEM_STALE_AFTER drops the
    /// reading so a dead node never shows stale numbers).
    telem_fails: u32,
    /// Some(model) = a dashboard-dispatched `load` is in flight on this node's
    /// worker. Guards against a second load racing the first, and renders as
    /// a spinner on the rail card. Cleared when the worker reports.
    loading: Option<String>,
    /// Sparkline rings, appended from the poll fold (no extra IO): GPU temp,
    /// UMA memory pressure (percent of pool) and socket power ride every
    /// telemetry reading; gen tok/s only when a NEW bench figure lands
    /// (status repeats the last bench every poll).
    temp_hist: Ring,
    mem_hist: Ring,
    power_hist: Ring,
    tok_hist: Ring,
    /// When the node first failed to answer (None while reachable). Drives the
    /// optional `unreachable_after_secs` alert debounce.
    unreachable_since: Option<Instant>,
    /// The alert flags after the LAST poll fold - the edge detector's memory,
    /// so a persisting condition logs once, not every poll.
    alerted: AlertFlags,
}

impl NodeCard {
    fn new(name: &str) -> NodeCard {
        NodeCard {
            status: NodeStatus::unreachable(name),
            probed: false,
            telem: None,
            telem_fails: 0,
            loading: None,
            temp_hist: Ring::new(SPARK_SAMPLES),
            mem_hist: Ring::new(SPARK_SAMPLES),
            power_hist: Ring::new(SPARK_SAMPLES),
            tok_hist: Ring::new(SPARK_SAMPLES),
            unreachable_since: None,
            alerted: AlertFlags::default(),
        }
    }
}

/// Fleet-side modal state: the read-only [E]vent-log and efficiency-results
/// overlays (the model list lives on screen now - no picker overlay).
enum FleetMode {
    Normal,
    /// The event-log overlay ([E]); any key closes.
    Events,
    /// The ranked efficiency-bench results ([F], on completion); any key closes.
    EffResults,
}

/// What one fleet poll worker reports back: node index, status, telemetry.
type FleetPoll = (usize, NodeStatus, Option<Telemetry>);

struct FleetApp {
    cfg: Config,
    cards: Vec<NodeCard>,
    sel: usize,
    status: String,
    /// `B` (bench-all) is armed: the next `B` fires it, any other key cancels.
    /// A minutes-long, GPU-thrashing fleet op shouldn't run on one stray keypress.
    bench_all_armed: bool,
    /// `F` (efficiency bench) is armed - same two-step confirm as bench-all.
    eff_armed: bool,
    /// Alert thresholds (settings.toml `[alerts]`, defaults preserve the old
    /// hardcoded 70/80). Loaded once per session - the draw path never reads
    /// the settings file.
    alerts_cfg: crate::settings::AlertsCfg,
    /// Bounded, edge-triggered event log (the [E] overlay).
    events: EventLog,
    /// The in-flight/finished fleet efficiency bench ([F]).
    eff: EffBench,
    /// [z]: force EVERY rail card compact - one line per node, the selected
    /// one included (a full 12-node chassis on one screen). Off (the default),
    /// the selected node's card expands with its full detail while the rest
    /// stay compact. Session-scoped.
    compact: bool,
    /// Modal state (events / efficiency-results overlays).
    mode: FleetMode,
    poll_tx: Sender<FleetPoll>,
    poll_rx: Receiver<FleetPoll>,
    /// Load workers report (node index, load outcome) here; poll() folds them.
    load_tx: Sender<(usize, Result<SwapReport>)>,
    load_rx: Receiver<(usize, Result<SwapReport>)>,
    /// Efficiency-bench workers report (node index, bench outcome) here.
    eff_tx: Sender<(usize, Result<Record>)>,
    eff_rx: Receiver<(usize, Result<Record>)>,
    /// Some(when) = a poll worker for node i has been in flight since `when`.
    inflight: Vec<Option<Instant>>,
    /// When node i was last dispatched (None = due immediately).
    polled_at: Vec<Option<Instant>>,
    /// Session start - drives the loading spinner animation.
    started: Instant,
}

impl FleetApp {
    fn load(cfg: Config) -> FleetApp {
        let n = cfg.nodes.len();
        let (poll_tx, poll_rx) = std::sync::mpsc::channel();
        let (load_tx, load_rx) = std::sync::mpsc::channel();
        let (eff_tx, eff_rx) = std::sync::mpsc::channel();
        let cards = cfg.nodes.iter().map(|nd| NodeCard::new(&nd.name)).collect();
        // Open on the first RAIL card (clusters reorder the display, so config
        // index 0 may not be the top of the rail).
        let sel = fleet::rail_order(&cfg).first().copied().unwrap_or(0);
        let mut f = FleetApp {
            cfg,
            cards,
            sel,
            status: "probing nodes...".into(),
            bench_all_armed: false,
            eff_armed: false,
            alerts_cfg: crate::settings::alerts(),
            events: EventLog::new(EVENT_LOG_CAP),
            eff: EffBench::Idle,
            compact: false,
            mode: FleetMode::Normal,
            poll_tx,
            poll_rx,
            load_tx,
            load_rx,
            eff_tx,
            eff_rx,
            inflight: vec![None; n],
            polled_at: vec![None; n],
            started: Instant::now(),
        };
        f.poll(); // dispatch the first round immediately
        f
    }

    /// Drain finished pollers and dispatch due ones. Never blocks: each node's
    /// probe (status + telemetry) runs on a one-shot worker and reports over
    /// `poll_rx`, so a down node times out on its worker thread while the rail
    /// keeps rendering.
    fn poll(&mut self) {
        while let Ok((i, status, telem)) = self.poll_rx.try_recv() {
            if let Some(fl) = self.inflight.get_mut(i) {
                *fl = None;
            }
            let Some(c) = self.cards.get_mut(i) else {
                continue;
            };
            c.probed = true;
            c.status = status;
            match telem {
                Some(t) => {
                    // Sparkline history rides the poll fold - no extra IO.
                    c.temp_hist.push(t.temp_c);
                    if let (Some(used), Some(total)) = (t.mem_used_mib, t.mem_total_mib) {
                        if total > 0 {
                            c.mem_hist.push(used as f64 / total as f64 * 100.0);
                        }
                    }
                    if let Some(w) = t.power_w {
                        c.power_hist.push(w);
                    }
                    c.telem = Some(t);
                    c.telem_fails = 0;
                }
                None => {
                    c.telem_fails = c.telem_fails.saturating_add(1);
                    if c.telem_fails >= TELEM_STALE_AFTER {
                        c.telem = None;
                    }
                }
            }
            // tok/s history: only when a NEW bench figure lands (the status
            // repeats the last bench on every poll - appending each time would
            // draw a flat line of duplicates).
            if let Some(tok) = c.status.last_gen_tok_s {
                if c.tok_hist.last() != Some(tok) {
                    c.tok_hist.push(tok);
                }
            }
            // Alert edge detection: track the outage start (for the optional
            // unreachable debounce), recompute the flags, and log TRANSITIONS
            // only - a node sitting hot logs once, not every 5 s.
            if !c.status.reachable {
                if c.unreachable_since.is_none() {
                    c.unreachable_since = Some(Instant::now());
                }
            } else {
                c.unreachable_since = None;
            }
            let prev = c.alerted;
            let cur = fleet::alert_flags(c, &self.alerts_cfg);
            c.alerted = cur;
            if prev != cur {
                let name = self
                    .cfg
                    .nodes
                    .get(i)
                    .map(|n| n.name.as_str())
                    .unwrap_or("?");
                fold_alert_events(
                    &mut self.events,
                    crate::history::now_unix(),
                    name,
                    prev,
                    cur,
                    c,
                );
            }
        }
        // Fold finished load workers: clear the card's loading state, headline
        // the outcome, and mark the node due so the served model refreshes.
        while let Ok((i, res)) = self.load_rx.try_recv() {
            if let Some(c) = self.cards.get_mut(i) {
                c.loading = None;
            }
            let name = self
                .cfg
                .nodes
                .get(i)
                .map(|n| n.name.clone())
                .unwrap_or_else(|| "?".into());
            self.status = match res {
                Ok(rep) if rep.ok => format!("{name}: loaded `{}`", rep.to),
                Ok(rep) => format!("{name}: load reverted: {}", rep.detail),
                Err(e) => format!("{name}: load failed: {e}"),
            };
            if let Some(p) = self.polled_at.get_mut(i) {
                *p = None; // due now: fold the new served model promptly
            }
        }
        // Fold finished efficiency-bench workers; open the ranked overlay when
        // the last one lands. The workers ran the benches (and the power
        // sampling) off-thread - this only collects results.
        while let Ok((i, res)) = self.eff_rx.try_recv() {
            let EffBench::Running { results } = &mut self.eff else {
                continue; // stale report (shouldn't happen; never panic)
            };
            let Some(slot) = results.get_mut(i) else {
                continue;
            };
            if slot.is_none() {
                *slot = Some(match res {
                    Ok(rec) => EffResult {
                        tok_s: (rec.perf.gen_tok_s > 0.0).then_some(rec.perf.gen_tok_s),
                        avg_w: rec.perf.telemetry.power_avg_w,
                        err: None,
                    },
                    Err(e) => EffResult {
                        tok_s: None,
                        avg_w: None,
                        err: Some(e.to_string()),
                    },
                });
            }
            let done = results.iter().filter(|r| r.is_some()).count();
            self.status = format!("efficiency bench: {done}/{} node(s) done...", results.len());
        }
        if matches!(&self.eff, EffBench::Running { results } if results.iter().all(|r| r.is_some()))
        {
            if let EffBench::Running { results } = std::mem::replace(&mut self.eff, EffBench::Idle)
            {
                self.eff = EffBench::Done(results);
                self.mode = FleetMode::EffResults;
                self.status = "efficiency bench complete".into();
                // fresh bench figures exist on every node: re-poll on the next
                // ticks (workers; never inline here)
                for p in &mut self.polled_at {
                    *p = None;
                }
            }
        }
        if self.status == "probing nodes..." && self.cards.iter().all(|c| c.probed) {
            self.status = format!("{} node(s)", self.cards.len());
        }
        for i in 0..self.cfg.nodes.len() {
            if let Some(since) = self.inflight[i] {
                if since.elapsed() < FLEET_POLL_TIMEOUT {
                    continue;
                }
                self.inflight[i] = None; // presumed lost; re-dispatch below
            }
            let due = self.polled_at[i].is_none_or(|t| t.elapsed() >= FLEET_POLL_INTERVAL);
            if !due {
                continue;
            }
            let now = Instant::now();
            self.polled_at[i] = Some(now);
            self.inflight[i] = Some(now);
            let node = self.cfg.nodes[i].clone();
            let tx = self.poll_tx.clone();
            std::thread::spawn(move || {
                let (status, telem) = match transport::for_node(&node) {
                    Ok(t) => (t.status(), t.telemetry()),
                    Err(_) => (NodeStatus::unreachable(&node.name), None),
                };
                let _ = tx.send((i, status, telem));
            });
        }
    }

    /// Force an immediate re-poll of every node. The probes still run on
    /// workers; this only marks them due and dispatches.
    fn refresh(&mut self) {
        for p in &mut self.polled_at {
            *p = None;
        }
        self.poll();
    }

    /// Jump-to-node keys: `1`-`9` select rail cards 1-9, `0` the 10th (cards
    /// 11-12 of a full chassis stay on Up/Down). Positions are RAIL order
    /// (cluster grouping may reorder the config), and out-of-range digits are
    /// a no-op, never a panic or a clamp.
    fn jump(&mut self, c: char) {
        let p = match c {
            '0' => 9,
            '1'..='9' => (c as u8 - b'1') as usize,
            _ => return,
        };
        if let Some(&i) = fleet::rail_order(&self.cfg).get(p) {
            self.sel = i;
        }
    }

    /// Move the selection through the RAIL order, so visual neighbors are
    /// selection neighbors even when cluster grouping reorders the config.
    fn move_sel(&mut self, delta: i32) {
        let order = fleet::rail_order(&self.cfg);
        if order.is_empty() {
            return;
        }
        let pos = order.iter().position(|&i| i == self.sel).unwrap_or(0) as i32;
        let np = (pos + delta).clamp(0, order.len() as i32 - 1) as usize;
        self.sel = order[np];
    }

    /// [F] (confirmed): the fleet efficiency bench. One worker thread PER
    /// node: each box has its own GPU, so cross-node parallelism is safe, and
    /// a down node's timeout can never stall the rest. Per node the bench
    /// stops llama-server, runs llama-bench while sampling socket power, then
    /// restarts, which is why the key is confirm-gated. poll() folds the
    /// results and opens the ranked overlay.
    fn start_eff_bench(&mut self) {
        if !matches!(self.eff, EffBench::Idle) {
            self.status = "efficiency bench already running...".into();
            return;
        }
        let n = self.cfg.nodes.len();
        for (i, node) in self.cfg.nodes.iter().cloned().enumerate() {
            let tx = self.eff_tx.clone();
            std::thread::spawn(move || {
                let res =
                    transport::for_node(&node).and_then(|t| t.bench(&bench::BenchSpec::default()));
                let _ = tx.send((i, res));
            });
        }
        self.eff = EffBench::Running {
            results: (0..n).map(|_| None).collect(),
        };
        self.status = format!("efficiency bench: 0/{n} node(s) done...");
    }

    /// Dispatch `load(name)` on each target node, one worker thread per node
    /// (the whole point: a live llama-server swap takes tens of seconds, and a
    /// down node's ssh timeout takes 8 - neither may touch the event loop).
    /// Nodes already mid-load are skipped, never raced. Returns the number of
    /// loads actually dispatched.
    fn dispatch_load(&mut self, targets: &[usize], name: &str) -> usize {
        let mut n = 0;
        for &i in targets {
            let Some(card) = self.cards.get_mut(i) else {
                continue;
            };
            let Some(node) = self.cfg.nodes.get(i).cloned() else {
                continue;
            };
            if card.loading.is_some() {
                continue; // guard: one load per node at a time
            }
            card.loading = Some(name.to_string());
            let name = name.to_string();
            let tx = self.load_tx.clone();
            std::thread::spawn(move || {
                let res = transport::for_node(&node).and_then(|t| t.load(&name));
                let _ = tx.send((i, res));
            });
            n += 1;
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Cockpit (the unified app: fleet rail + focused node's models + model card)
// ---------------------------------------------------------------------------

/// Which pane Up/Down/Enter act on. Only navigation is focus-scoped - every
/// action key works from either pane (on the focused node / selected model).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focus {
    Nodes,
    Models,
}

/// What a key press asks the event loop to do. Key routing itself lives in
/// `CockpitApp::on_key` (pure state, testable); only these need the loop.
#[derive(Debug, PartialEq, Eq)]
enum KeyOutcome {
    Continue,
    Quit,
    /// Run the sequential bench-all (needs the terminal for per-node redraws).
    BenchAll,
}

/// The focused node's model list + history being fetched on a worker thread
/// (moving the node selection must never block on an SSH round-trip).
enum ModelsFetch {
    Idle,
    Running {
        node_idx: usize,
        handle: JoinHandle<(Vec<ModelInfo>, Vec<Record>)>,
    },
}

/// A borrowed draw-view over the cockpit state - the draw path never mutates,
/// and tests can render any (fleet, node, focus) combination directly.
struct CockpitView<'a> {
    fleet: &'a FleetApp,
    node: &'a NodeApp,
    focus: Focus,
    headline: &'a str,
    /// The fleet/netboot dashboard (Some while [N] is open).
    netboot: Option<&'a netboot::NetbootDash>,
}

struct CockpitApp {
    fleet: FleetApp,
    /// The focused node's model surface (list/card/actions). `node.node`
    /// always mirrors `fleet.cfg.nodes[fleet.sel]`.
    node: NodeApp,
    focus: Focus,
    fetch: ModelsFetch,
    /// The selection moved while a fetch was in flight: re-dispatch on land.
    fetch_queued: bool,
    /// The focused node had a fleet-dispatched load in flight last tick (a
    /// Some->None edge means it finished: refetch so the served marker moves).
    was_loading: bool,
    /// The one visible status line: whichever of the two apps' statuses
    /// changed most recently.
    headline: String,
    prev_fleet_status: String,
    prev_node_status: String,
    /// The `--config` path the TUI was launched with (so a netboot register
    /// writes the file that was actually loaded, not just the default).
    config_path: Option<String>,
    /// The fleet/netboot dashboard state (Some while [N] is open).
    netboot: Option<netboot::NetbootDash>,
}

impl CockpitApp {
    fn load(cfg: Config, config_path: Option<String>) -> CockpitApp {
        let fleet = FleetApp::load(cfg);
        // Config::load synthesizes a localhost node when none are configured,
        // so the fleet is never empty; fall back to index 0 defensively.
        let nd = fleet
            .cfg
            .nodes
            .get(fleet.sel)
            .or_else(|| fleet.cfg.nodes.first())
            .cloned()
            .expect("config guarantees at least one node");
        let mut node = NodeApp::new(nd);
        node.fleet_nodes = fleet.cfg.nodes.clone();
        node.clusters = fleet.cfg.clusters.clone();
        node.status = format!("listing models on {}...", node.node.name);
        let headline = fleet.status.clone();
        let prev_fleet_status = fleet.status.clone();
        let prev_node_status = node.status.clone();
        let mut app = CockpitApp {
            fleet,
            node,
            focus: Focus::Nodes,
            fetch: ModelsFetch::Idle,
            fetch_queued: false,
            was_loading: false,
            headline,
            prev_fleet_status,
            prev_node_status,
            config_path,
            netboot: None,
        };
        app.queue_fetch();
        app
    }

    /// Register newly-discovered boards as fleet nodes (writes the loaded config)
    /// and refresh the netboot snapshot so their flags flip. The new nodes become
    /// drivable on the next launch (cards are built at load); this session's view
    /// reflects them immediately.
    fn netboot_register(&mut self) {
        match crate::netboot::register_discovered(&self.fleet.cfg, self.config_path.as_deref()) {
            Ok(added) if added.is_empty() => {
                self.node.status = "netboot: all discovered boards already registered".into();
            }
            Ok(added) => {
                self.node.status = format!(
                    "netboot: registered {} board(s): {} - restart to drive them",
                    added.len(),
                    added.join(", ")
                );
                // Rebuild the view from the freshly-written config (registered
                // flags now true) WITHOUT mutating the live fleet (its cards are
                // built at load, so a live cfg swap would desync them).
                if let Ok(cfg) = Config::load(self.config_path.as_deref()) {
                    if let Some(d) = self.netboot.as_mut() {
                        d.view = NetbootView::build(&cfg);
                        d.clamp_sel(self.fleet.cfg.nodes.len());
                    }
                }
            }
            Err(e) => self.node.status = format!("netboot register failed: {e}"),
        }
    }

    /// Fold the netboot dash's background workers (server probe, arm/boot
    /// actions, console stream) into the UI state.
    fn poll_netboot(&mut self) {
        if let Some(d) = self.netboot.as_mut() {
            if let Some(headline) = d.poll() {
                self.node.status = headline;
            }
        }
    }

    fn view(&self) -> CockpitView<'_> {
        CockpitView {
            fleet: &self.fleet,
            node: &self.node,
            focus: self.focus,
            headline: &self.headline,
            netboot: self.netboot.as_ref(),
        }
    }

    /// Dispatch (or queue) a worker fetch of the focused node's models +
    /// bench history. One at a time; a landed fetch re-dispatches if queued.
    fn queue_fetch(&mut self) {
        if matches!(self.fetch, ModelsFetch::Running { .. }) {
            self.fetch_queued = true;
            return;
        }
        let node_idx = self.fleet.sel;
        let node = self.node.node.clone();
        self.fetch = ModelsFetch::Running {
            node_idx,
            handle: std::thread::spawn(move || match transport::for_node(&node) {
                Ok(t) => (
                    t.list().unwrap_or_default(),
                    t.history(200).unwrap_or_default(),
                ),
                Err(_) => (Vec::new(), Vec::new()),
            }),
        };
    }

    /// Fold a finished models fetch into the models pane; discard stale
    /// results (the selection moved on) and re-dispatch if one is queued.
    /// Also edge-detects a fleet-dispatched load finishing on the focused
    /// node - the cue to refetch so the served marker updates.
    fn poll_fetch(&mut self) {
        let loading = self
            .fleet
            .cards
            .get(self.fleet.sel)
            .is_some_and(|c| c.loading.is_some());
        if self.was_loading && !loading {
            self.queue_fetch();
        }
        self.was_loading = loading;

        let done =
            matches!(&self.fetch, ModelsFetch::Running { handle, .. } if handle.is_finished());
        if !done {
            return;
        }
        if let ModelsFetch::Running { node_idx, handle } =
            std::mem::replace(&mut self.fetch, ModelsFetch::Idle)
        {
            if let Ok((models, history)) = handle.join() {
                if node_idx == self.fleet.sel {
                    self.node.models = models;
                    self.node.history = history;
                    self.node.clamp_sel();
                    self.node.status = format!(
                        "{}: {} model(s)",
                        self.node.node.name,
                        self.node.models.len()
                    );
                }
            }
        }
        if std::mem::take(&mut self.fetch_queued) {
            self.queue_fetch();
        }
    }

    /// Point the models pane at the (new) fleet selection: clear the stale
    /// list and dispatch a worker fetch. No-op when already there.
    fn sync_node(&mut self) {
        let Some(nd) = self.fleet.cfg.nodes.get(self.fleet.sel) else {
            return;
        };
        if self.node.node.name == nd.name {
            return;
        }
        self.node.node = nd.clone();
        self.node.models.clear();
        self.node.history.clear();
        self.node.sel = 0;
        self.node.filter.clear();
        self.node.status = format!("listing models on {}...", nd.name);
        self.queue_fetch();
    }

    /// Guard for node-selection moves: a bench/expose worker is bound to the
    /// current node, so the pane must not be re-pointed under it.
    fn node_pinned(&mut self) -> bool {
        if self.node.busy() {
            self.node.status = format!(
                "{} is busy - wait for the current op to finish",
                self.node.node.name
            );
            return true;
        }
        false
    }

    fn select_node(&mut self, delta: i32) {
        if self.node_pinned() {
            return;
        }
        self.fleet.move_sel(delta);
        self.sync_node();
    }

    fn jump_node(&mut self, c: char) {
        if self.node_pinned() {
            return;
        }
        self.fleet.jump(c);
        self.sync_node();
    }

    /// Keep the visible status line on whichever app spoke last.
    fn fold_status(&mut self) {
        if self.fleet.status != self.prev_fleet_status {
            self.prev_fleet_status = self.fleet.status.clone();
            self.headline = self.fleet.status.clone();
        }
        if self.node.status != self.prev_node_status {
            self.prev_node_status = self.node.status.clone();
            self.headline = self.node.status.clone();
        }
    }

    /// Confirmed load: dispatch the selected model onto the focused node (or
    /// the whole fleet) via the per-node load workers - the rail cards show
    /// spinners and the event loop never blocks.
    fn confirm_load(&mut self, vis_idx: usize, all: bool) {
        let Some(name) = self
            .node
            .visible()
            .get(vis_idx)
            .and_then(|&i| self.node.models.get(i))
            .map(|m| m.name.clone())
        else {
            return;
        };
        let targets: Vec<usize> = if all {
            (0..self.fleet.cfg.nodes.len()).collect()
        } else {
            vec![self.fleet.sel]
        };
        let n = self.fleet.dispatch_load(&targets, &name);
        self.fleet.status = if n == 0 {
            "no load dispatched: target node(s) already loading".into()
        } else {
            format!("loading `{name}` on {n} node(s)...")
        };
    }

    /// All key routing. Modal modes first (they capture every key), then the
    /// armed two-step confirms, then the Normal-mode map: navigation is
    /// focus-scoped, action keys are global.
    fn on_key(&mut self, code: KeyCode) -> KeyOutcome {
        if self.on_key_modal(code) {
            return KeyOutcome::Continue;
        }
        // Read-only fleet overlays (event log / efficiency results): any key
        // closes them.
        if matches!(self.fleet.mode, FleetMode::Events | FleetMode::EffResults) {
            self.fleet.mode = FleetMode::Normal;
            return KeyOutcome::Continue;
        }
        // Any key other than a second `B` cancels an armed bench-all; same
        // two-step confirm for the efficiency bench.
        if self.fleet.bench_all_armed && !matches!(code, KeyCode::Char('B')) {
            self.fleet.bench_all_armed = false;
            self.fleet.status = "bench-all cancelled".into();
        }
        if self.fleet.eff_armed && !matches!(code, KeyCode::Char('F')) {
            self.fleet.eff_armed = false;
            self.fleet.status = "efficiency bench cancelled".into();
        }
        match code {
            KeyCode::Char('q') => return KeyOutcome::Quit,
            KeyCode::Esc => {
                if self.node.filter.is_empty() {
                    return KeyOutcome::Quit;
                }
                self.node.filter.clear();
                self.node.clamp_sel();
            }
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = match self.focus {
                    Focus::Nodes => Focus::Models,
                    Focus::Models => Focus::Nodes,
                };
            }
            KeyCode::Left => self.focus = Focus::Nodes,
            KeyCode::Right => self.focus = Focus::Models,
            KeyCode::Up => match self.focus {
                Focus::Nodes => self.select_node(-1),
                Focus::Models => move_sel(&mut self.node, -1),
            },
            KeyCode::Down => match self.focus {
                Focus::Nodes => self.select_node(1),
                Focus::Models => move_sel(&mut self.node, 1),
            },
            KeyCode::Enter => match self.focus {
                Focus::Nodes => self.focus = Focus::Models,
                Focus::Models => {
                    if self.node.selected_model().is_some() && !self.node.busy() {
                        self.node.mode = Mode::ConfirmSwap(self.node.sel);
                    }
                }
            },
            KeyCode::Char(c @ '0'..='9') => self.jump_node(c),
            KeyCode::Char('z') => self.fleet.compact = !self.fleet.compact,
            KeyCode::Char('E') => self.fleet.mode = FleetMode::Events,
            KeyCode::Char('F') => {
                // Two-step like bench-all: an efficiency bench stops/benches/
                // restarts llama-server on EVERY node - not a stray-key op.
                if !self.fleet.eff_armed {
                    self.fleet.eff_armed = true;
                    self.fleet.status = format!(
                        "efficiency-bench ALL {} node(s)? press F again to confirm (any other key cancels)",
                        self.fleet.cfg.nodes.len()
                    );
                } else {
                    self.fleet.eff_armed = false;
                    self.fleet.start_eff_bench();
                }
            }
            KeyCode::Char('B') => {
                // Two-step: arm on the first B, fire on the second. Bench-all
                // is minutes-long and thrashes every node's GPU, so it must
                // not run on a single stray keypress.
                if !self.fleet.bench_all_armed {
                    self.fleet.bench_all_armed = true;
                    self.fleet.status = format!(
                        "bench ALL {} nodes? press B again to confirm (any other key cancels)",
                        self.fleet.cfg.nodes.len()
                    );
                } else {
                    self.fleet.bench_all_armed = false;
                    return KeyOutcome::BenchAll;
                }
            }
            KeyCode::Char('r') => {
                // Marks every node due and dispatches workers; results land
                // on later ticks (the event loop never blocks on a probe).
                self.fleet.refresh();
                self.queue_fetch();
                self.fleet.status = "refreshing...".into();
            }
            KeyCode::Char('/') => {
                self.focus = Focus::Models;
                self.node.mode = Mode::Filter;
                self.node.sel = 0;
            }
            KeyCode::Char('?') | KeyCode::Char('d') => self.node.mode = Mode::Help,
            KeyCode::Char('b') => self.node.start_bench(),
            KeyCode::Char('e') => self.node.start_flag_edit(),
            KeyCode::Char('o') => {
                if !self.node.busy() {
                    // Probe THIS node's endpoint through its transport (a
                    // remote node reports its own URL/auth, not the
                    // controller's).
                    match transport::for_node(&self.node.node).and_then(|t| t.endpoint()) {
                        Ok(ep) => {
                            self.node.endpoint_view = Some(ep);
                            self.node.mode = Mode::Endpoint;
                        }
                        Err(e) => self.node.status = format!("endpoint probe failed: {e}"),
                    }
                }
            }
            KeyCode::Char('u') => {
                if !self.node.busy() {
                    self.node.mode = Mode::ConfirmUnload;
                }
            }
            KeyCode::Char('s') => {
                if !self.node.busy() {
                    // Probe active-state ONCE here, not on every dialog frame.
                    self.node.server_active = matches!(self.node.node.transport, Transport::Local)
                        && std::process::Command::new("systemctl")
                            .args(["is-active", "--quiet", &self.node.node.llama_unit])
                            .status()
                            .map(|s| s.success())
                            .unwrap_or(false);
                    self.node.mode = Mode::ConfirmServer;
                }
            }
            KeyCode::Char('n') => {
                if !self.node.busy() {
                    // Refuse up front for a remote node (the confirm dialog
                    // reads LOCAL settings state and the toggle writes local
                    // config).
                    if matches!(self.node.node.transport, Transport::Local) {
                        self.node.probe_expose_state();
                        self.node.mode = Mode::ConfirmExpose;
                    } else {
                        self.node.status =
                            "network exposure is local-only - run llmtune on the node".into();
                    }
                }
            }
            KeyCode::Char('h') => self.node.mode = Mode::History,
            KeyCode::Char('L') => self.node.mode = Mode::Leaderboard,
            KeyCode::Char('k') => self.node.mode = Mode::Compare,
            KeyCode::Char('y') => {
                if !self.node.busy() {
                    // Run doctor ON the focused node (remote preflight over SSH).
                    match transport::for_node(&self.node.node).and_then(|t| t.doctor()) {
                        Ok(checks) => {
                            self.node.doctor_view = Some(checks);
                            self.node.mode = Mode::Doctor;
                        }
                        Err(e) => self.node.status = format!("doctor failed: {e}"),
                    }
                }
            }
            KeyCode::Char('p') => {
                // profile::load() reads the CONTROLLER's config - showing it
                // for a remote node would misrepresent that node's profiles,
                // so refuse there (like expose/server toggle).
                if matches!(self.node.node.transport, Transport::Local) {
                    self.node.profiles_view = Some(profile::load().unwrap_or_default());
                    self.node.mode = Mode::Profiles;
                } else {
                    self.node.status =
                        "profiles view is controller-local - run llmtune on the node".into();
                }
            }
            KeyCode::Char('g') => {
                // build::load()/list() inspect the CONTROLLER's managed
                // builds; same honesty rule as the profiles overlay above.
                if matches!(self.node.node.transport, Transport::Local) {
                    let specs = build::load().unwrap_or_default();
                    self.node.builds_view = Some(build::list(&specs));
                    self.node.mode = Mode::Builds;
                } else {
                    self.node.status =
                        "builds view is controller-local - run llmtune on the node".into();
                }
            }
            KeyCode::Char('K') => self.node.mode = Mode::Clusters,
            KeyCode::Char('N') => {
                // The fleet/netboot dashboard. Server/lease probes read the
                // CONTROLLER's state, which is right wherever the TUI runs (the
                // control host by definition); with no [netboot] configured the
                // nodes table still works (arm/console are plain SSH/UDP).
                let mut dash = netboot::NetbootDash::new(NetbootView::build(&self.fleet.cfg));
                // Open on the cockpit's focused node (dash rows are config order).
                dash.sel = self
                    .fleet
                    .sel
                    .min(self.fleet.cfg.nodes.len().saturating_sub(1));
                dash.spawn_server_probe(&self.fleet.cfg);
                self.netboot = Some(dash);
                self.node.mode = Mode::Netboot;
            }
            KeyCode::Char('A') => {
                if self.node.selected_model().is_some() && !self.node.busy() {
                    self.node.mode = Mode::ConfirmSwapAll;
                }
            }
            _ => {}
        }
        KeyOutcome::Continue
    }

    /// Modal-mode key handling (confirms, endpoint, read-only overlays, the
    /// filter input, the flag editor). Returns true when the key was consumed
    /// by a modal - the Normal-mode map must not see it.
    fn on_key_modal(&mut self, code: KeyCode) -> bool {
        if let Mode::ConfirmSwap(idx) = self.node.mode {
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.node.mode = Mode::Normal;
                    self.confirm_load(idx, false);
                }
                _ => {
                    self.node.mode = Mode::Normal;
                    self.node.status = "cancelled".into();
                }
            }
            return true;
        }
        if matches!(self.node.mode, Mode::ConfirmSwapAll) {
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.node.mode = Mode::Normal;
                    self.confirm_load(self.node.sel, true);
                }
                _ => {
                    self.node.mode = Mode::Normal;
                    self.node.status = "cancelled".into();
                }
            }
            return true;
        }
        if matches!(self.node.mode, Mode::ConfirmUnload) {
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.node.mode = Mode::Normal;
                    self.node.do_unload();
                }
                _ => {
                    self.node.mode = Mode::Normal;
                    self.node.status = "cancelled".into();
                }
            }
            return true;
        }
        if matches!(self.node.mode, Mode::ConfirmServer) {
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.node.mode = Mode::Normal;
                    self.node.do_server_toggle();
                }
                KeyCode::Char('r') | KeyCode::Char('R') => {
                    self.node.mode = Mode::Normal;
                    self.node.do_server("restart");
                }
                _ => {
                    self.node.mode = Mode::Normal;
                    self.node.status = "cancelled".into();
                }
            }
            return true;
        }
        if matches!(self.node.mode, Mode::ConfirmExpose) {
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.node.mode = Mode::Normal;
                    self.node.do_expose();
                }
                _ => {
                    self.node.mode = Mode::Normal;
                    self.node.status = "cancelled".into();
                }
            }
            // If this confirm came from the endpoint overlay, reopen it (now
            // refreshed) so the owner sees the exposed base + LAN commands.
            if std::mem::take(&mut self.node.reopen_endpoint_after_expose) {
                self.node.reopen_endpoint();
            }
            return true;
        }
        // Endpoint overlay is interactive: e toggles exposure, a toggles auth,
        // i toggles identity (re-probing in place); any other key closes it.
        if matches!(self.node.mode, Mode::Endpoint) {
            // A key other than a second `a` cancels an armed auth-off.
            if !matches!(code, KeyCode::Char('a')) {
                self.node.auth_off_armed = false;
            }
            match code {
                KeyCode::Char('e') => self.node.expose_from_overlay(),
                // [a] toggles auth: on -> generate+apply a key, off -> keyless.
                // Disabling auth while the endpoint is EXPOSED needs a second
                // `a` (it would otherwise leave the LAN endpoint open in one
                // keystroke).
                KeyCode::Char('a') => {
                    let turning_off = crate::settings::api_key().is_some();
                    if turning_off && crate::settings::exposed() && !self.node.auth_off_armed {
                        self.node.auth_off_armed = true;
                        self.node.status =
                            "disable auth on an EXPOSED endpoint? press a again to confirm".into();
                    } else {
                        self.node.auth_off_armed = false;
                        let key = if turning_off {
                            None
                        } else {
                            Some(crate::settings::generate_key())
                        };
                        self.node.do_api_key(key);
                    }
                }
                // [i] toggles identity: branded -> strip to harness-controlled;
                // harness-controlled -> restore the embedded (branded) template.
                KeyCode::Char('i') => {
                    let to_harness = self
                        .node
                        .endpoint_view
                        .as_ref()
                        .is_some_and(|ep| ep.identity_branded);
                    self.node.do_identity(to_harness);
                }
                _ => {
                    self.node.mode = Mode::Normal;
                    self.node.endpoint_view = None;
                }
            }
            return true;
        }
        // The fleet/netboot dashboard: interactive (selection, arm/boot/console,
        // register); its key routing lives in the dash itself.
        if matches!(self.node.mode, Mode::Netboot) {
            let outcome = match self.netboot.as_mut() {
                Some(d) => {
                    let (outcome, msg) = d.on_key(code, &self.fleet.cfg);
                    if let Some(m) = msg {
                        self.node.status = m;
                    }
                    outcome
                }
                None => netboot::DashOutcome::Close,
            };
            match outcome {
                netboot::DashOutcome::Stay => {}
                netboot::DashOutcome::Close => {
                    self.node.mode = Mode::Normal;
                    self.netboot = None;
                }
                netboot::DashOutcome::FocusNode(i) => {
                    // Close the dash and drive that node in the cockpit (the
                    // swap/bench flow continues in the main view).
                    self.node.mode = Mode::Normal;
                    self.netboot = None;
                    if i < self.fleet.cfg.nodes.len() && !self.node_pinned() {
                        self.fleet.sel = i;
                        self.sync_node();
                        self.focus = Focus::Models;
                    }
                }
                netboot::DashOutcome::Register => self.netboot_register(),
            }
            return true;
        }
        // Read-only overlays: any key closes them (and clears their state).
        if matches!(
            self.node.mode,
            Mode::Help
                | Mode::History
                | Mode::Leaderboard
                | Mode::Compare
                | Mode::Doctor
                | Mode::Profiles
                | Mode::Builds
                | Mode::Clusters
        ) {
            self.node.mode = Mode::Normal;
            self.node.endpoint_view = None;
            self.node.doctor_view = None;
            self.node.profiles_view = None;
            self.node.builds_view = None;
            return true;
        }
        // Filter-input mode: type to narrow the model list.
        if matches!(self.node.mode, Mode::Filter) {
            match code {
                KeyCode::Esc => {
                    self.node.filter.clear();
                    self.node.mode = Mode::Normal;
                    self.node.clamp_sel();
                }
                KeyCode::Enter => self.node.mode = Mode::Normal,
                KeyCode::Backspace => {
                    self.node.filter.pop();
                    self.node.sel = 0;
                }
                KeyCode::Char(c) => {
                    self.node.filter.push(c);
                    self.node.sel = 0;
                }
                _ => {}
            }
            return true;
        }
        // Flag editor: edit the profile's flags one-per-line.
        if matches!(self.node.mode, Mode::EditFlags) {
            self.on_key_flag_editor(code);
            return true;
        }
        false
    }

    /// The in-card flag editor's key handling (unchanged semantics).
    fn on_key_flag_editor(&mut self, code: KeyCode) {
        let app = &mut self.node;
        let (mut save, mut cancel, mut reset) = (false, false, false);
        if let Some(fe) = app.flag_edit.as_mut() {
            if let Some(buf) = fe.buf.as_mut() {
                // editing a single flag line
                match code {
                    KeyCode::Char(c) => buf.push(c),
                    KeyCode::Backspace => {
                        buf.pop();
                    }
                    KeyCode::Enter => {
                        let v = fe.buf.take().unwrap().trim().to_string();
                        if v.is_empty() {
                            fe.flags.remove(fe.sel);
                            fe.sel = fe.sel.min(fe.flags.len().saturating_sub(1));
                        } else {
                            fe.flags[fe.sel] = v;
                        }
                    }
                    KeyCode::Esc => {
                        let v = fe.buf.take().unwrap();
                        if v.is_empty() {
                            fe.flags.remove(fe.sel);
                            fe.sel = fe.sel.min(fe.flags.len().saturating_sub(1));
                        }
                    }
                    _ => {}
                }
            } else {
                match code {
                    KeyCode::Up => fe.sel = fe.sel.saturating_sub(1),
                    KeyCode::Down => {
                        if fe.sel + 1 < fe.flags.len() {
                            fe.sel += 1;
                        }
                    }
                    KeyCode::Enter | KeyCode::Char('i') => {
                        if !fe.flags.is_empty() {
                            fe.buf = Some(fe.flags[fe.sel].clone());
                        }
                    }
                    KeyCode::Char('a') => {
                        let at = (fe.sel + 1).min(fe.flags.len());
                        fe.flags.insert(at, String::new());
                        fe.sel = at;
                        fe.buf = Some(String::new());
                    }
                    KeyCode::Char('d') | KeyCode::Delete => {
                        if !fe.flags.is_empty() {
                            fe.flags.remove(fe.sel);
                            fe.sel = fe.sel.min(fe.flags.len().saturating_sub(1));
                        }
                    }
                    KeyCode::Char('s') => save = true,
                    KeyCode::Char('x') => reset = true,
                    KeyCode::Esc => cancel = true,
                    _ => {}
                }
            }
        }
        if save {
            if let Some(fe) = app.flag_edit.take() {
                let joined = fe
                    .flags
                    .iter()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                // Persist ON the focused node (a remote node's override lands
                // in ITS config, not the controller's - no silent no-op).
                let res = transport::for_node(&app.node)
                    .and_then(|t| t.set_model_flags(&fe.model_name, Some(&joined)));
                app.status = match res {
                    Ok(()) => format!("saved override for `{}`", fe.model_name),
                    Err(e) => format!("save failed: {e}"),
                };
            }
            app.mode = Mode::Normal;
            app.refresh();
        } else if reset {
            if let Some(fe) = app.flag_edit.take() {
                let res = transport::for_node(&app.node)
                    .and_then(|t| t.set_model_flags(&fe.model_name, None));
                app.status = match res {
                    Ok(()) => format!("reset `{}` to profile default", fe.model_name),
                    Err(e) => format!("reset failed: {e}"),
                };
            }
            app.mode = Mode::Normal;
            app.refresh();
        } else if cancel {
            app.flag_edit = None;
            app.mode = Mode::Normal;
            app.status = "edit cancelled".into();
        }
    }
}

enum Mode {
    Normal,
    /// Typing a model-list filter.
    Filter,
    ConfirmSwap(usize),
    /// Confirming removal of llmtune's drop-in (revert the unit to base config).
    ConfirmUnload,
    /// Confirming a llama-server stop/start (free/restore the GPU).
    ConfirmServer,
    /// Confirming a network-exposure toggle (bind 0.0.0.0 + open firewall).
    ConfirmExpose,
    /// Confirming a swap-all: load the selected model on every fleet node.
    ConfirmSwapAll,
    /// Editing the selected model's profile flags (state in `flag_edit`).
    EditFlags,
    Help,
    /// Showing the OpenAI endpoint + connection snippets (state in `endpoint_view`).
    Endpoint,
    /// Read-only overlay: the selected model's bench history.
    History,
    /// Read-only overlay: best gen tok/s per model.
    Leaderboard,
    /// Read-only overlay: quant families / throughput trade-off.
    Compare,
    /// Read-only overlay: preflight checks (state in `doctor_view`).
    Doctor,
    /// Read-only overlay: configured launch profiles (state in `profiles_view`).
    Profiles,
    /// Read-only overlay: managed llama.cpp builds (state in `builds_view`).
    Builds,
    /// Read-only overlay: configured RPC clusters (from `clusters`).
    Clusters,
    /// The fleet/netboot dashboard (state in `CockpitApp::netboot`): boot
    /// server health, node table, discovered boards, arm/boot/console.
    Netboot,
}

/// In-card flag editor: the model's effective flags as an editable one-per-line
/// list. Saving stores a per-model override keyed by `model_name`.
struct FlagEdit {
    model_name: String,
    flags: Vec<String>,
    sel: usize,
    /// Some = currently typing into `flags[sel]`.
    buf: Option<String>,
}

enum BenchState {
    Idle,
    Running(JoinHandle<Result<Record>>),
    Done,
    Failed,
}

/// A generic background task (expose/auth toggle) run off the UI thread
/// so a multi-second model reload doesn't freeze rendering. The worker returns
/// the finished status line; the poll loop applies it + the fast UI-thread
/// finalizers (reprobe/refresh).
enum BgState {
    Idle,
    Running(JoinHandle<String>),
}

/// After this many consecutive failed telemetry polls the last reading is
/// dropped (shown as unavailable) rather than displayed stale forever.
const TELEM_STALE_AFTER: u32 = 3;

struct NodeApp {
    node: Node,
    models: Vec<ModelInfo>,
    /// Benchmark history (shown per-model in the card; no separate pane).
    history: Vec<Record>,
    sel: usize,
    /// Model-list substring filter (empty = show all).
    filter: String,
    /// Active in-card flag editor (when `mode == EditFlags`).
    flag_edit: Option<FlagEdit>,
    bench: BenchState,
    /// Live per-run progress while a bench is running.
    bench_rx: Option<Receiver<bench::BenchProgress>>,
    bench_progress: Option<bench::BenchProgress>,
    /// A background task (expose/auth) running off the UI thread.
    bg: BgState,
    mode: Mode,
    /// llama-server active-state, probed ONCE when the confirm-server dialog
    /// opens - so the dialog doesn't fork `systemctl is-active` on every frame.
    server_active: bool,
    /// Disabling auth on an EXPOSED endpoint is armed: the next `a` confirms it.
    /// Prevents one keystroke from leaving a LAN-exposed endpoint unauthenticated.
    auth_off_armed: bool,
    /// (exposed, has_api_key) probed ONCE when the confirm-expose dialog opens -
    /// so that dialog doesn't read settings.toml twice on every render frame.
    expose_state: (bool, bool),
    status: String,
    started: Instant,
    /// Probed endpoint (set when the endpoint overlay opens).
    endpoint_view: Option<Endpoint>,
    /// The expose-confirm dialog was opened FROM the endpoint overlay - reopen it
    /// (refreshed) after the toggle so the owner sees the new base/commands.
    reopen_endpoint_after_expose: bool,
    /// Doctor checks (set when the doctor overlay opens).
    doctor_view: Option<Vec<doctor::Check>>,
    /// Loaded launch profiles (set when the profiles overlay opens).
    profiles_view: Option<Vec<profile::Profile>>,
    /// Managed build statuses (set when the builds overlay opens).
    builds_view: Option<Vec<build::BuildStatus>>,
    /// The whole fleet's node list (for swap-all). Empty over a lone local node.
    fleet_nodes: Vec<Node>,
    /// Configured RPC clusters (for the clusters overlay).
    clusters: Vec<ClusterCfg>,
}

/// A snapshot for the cockpit's netboot view: the boot-server config, the stack
/// service states, and the discovered boards (with whether each is already a
/// registered fleet node). Computed when the overlay opens (`N`).
pub(super) struct NetbootView {
    pub server_ip: String,
    pub http_base: String,
    pub interface: String,
    pub leases_path: String,
    /// The netconsole UDP port boards ship their boot log to.
    pub console_port: u16,
    pub services: Vec<(&'static str, String)>,
    pub boards: Vec<NetbootBoard>,
}

pub(super) struct NetbootBoard {
    pub name: String,
    pub ip: String,
    pub mac: String,
    pub registered: bool,
}

impl NetbootView {
    /// Build the snapshot from the controller's config (systemctl probes + lease
    /// discovery). `None` when netboot is unconfigured.
    fn build(cfg: &Config) -> Option<NetbootView> {
        let nb = cfg.netboot.as_ref()?;
        let registered: std::collections::HashSet<String> =
            cfg.nodes.iter().filter_map(|n| n.host.clone()).collect();
        let boards = crate::netboot::discovered(nb)
            .iter()
            .map(|l| NetbootBoard {
                name: crate::netboot::node_name(nb, l),
                ip: l.ip.clone(),
                mac: l.mac.clone(),
                registered: registered.contains(&l.ip),
            })
            .collect();
        Some(NetbootView {
            server_ip: nb.server_ip.clone(),
            http_base: nb.http_base(),
            interface: nb.interface.clone(),
            leases_path: nb.leases.clone(),
            console_port: nb.console_port,
            services: crate::netboot::service_states(),
            boards,
        })
    }

    /// Count of discovered boards not yet registered as fleet nodes.
    pub(super) fn unregistered(&self) -> usize {
        self.boards.iter().filter(|b| !b.registered).count()
    }
}

impl NodeApp {
    /// Indices into `models` matching the current filter (all if empty).
    fn visible(&self) -> Vec<usize> {
        if self.filter.is_empty() {
            return (0..self.models.len()).collect();
        }
        let f = self.filter.to_lowercase();
        self.models
            .iter()
            .enumerate()
            .filter(|(_, m)| m.name.to_lowercase().contains(&f))
            .map(|(i, _)| i)
            .collect()
    }

    /// The model under the cursor (mapped through the filter), if any.
    fn selected_model(&self) -> Option<&ModelInfo> {
        self.visible()
            .get(self.sel)
            .and_then(|&i| self.models.get(i))
    }
}

impl NodeApp {
    /// A fresh, EMPTY node surface - no IO. The cockpit fills models/history
    /// via a worker fetch (see `CockpitApp::queue_fetch`), so pointing the
    /// pane at a slow/remote node never blocks the event loop.
    fn new(node: Node) -> NodeApp {
        NodeApp {
            node,
            models: Vec::new(),
            history: Vec::new(),
            sel: 0,
            filter: String::new(),
            flag_edit: None,
            bench: BenchState::Idle,
            bench_rx: None,
            bench_progress: None,
            bg: BgState::Idle,
            mode: Mode::Normal,
            server_active: false,
            auth_off_armed: false,
            expose_state: (false, false),
            status: "ready".into(),
            started: Instant::now(),
            endpoint_view: None,
            reopen_endpoint_after_expose: false,
            doctor_view: None,
            profiles_view: None,
            builds_view: None,
            fleet_nodes: Vec::new(),
            clusters: Vec::new(),
        }
    }

    fn busy(&self) -> bool {
        matches!(self.bench, BenchState::Running(_)) || matches!(self.bg, BgState::Running(_))
    }

    fn refresh(&mut self) {
        if let Ok(t) = transport::for_node(&self.node) {
            self.models = t.list().unwrap_or_default();
            self.history = t.history(200).unwrap_or_default();
        }
        self.clamp_sel();
    }

    fn clamp_sel(&mut self) {
        let n = self.visible().len();
        if self.sel >= n {
            self.sel = n.saturating_sub(1);
        }
    }

    fn start_bench(&mut self) {
        if self.busy() {
            return;
        }
        let node = self.node.clone();
        self.status = "benchmarking served model...".into();
        self.started = Instant::now();
        self.bench_progress = None;
        let spec = bench::BenchSpec::default();
        if matches!(self.node.transport, Transport::Local) {
            // Local: stream per-run progress (tok/s ticks up live).
            let (tx, rx) = std::sync::mpsc::channel();
            self.bench_rx = Some(rx);
            self.bench = BenchState::Running(std::thread::spawn(move || {
                nodeops::bench_streaming(&node, &spec, tx)
            }));
        } else {
            // Remote: route through the transport (no live streaming).
            self.bench_rx = None;
            self.bench = BenchState::Running(std::thread::spawn(move || {
                transport::for_node(&node)?.bench(&spec)
            }));
        }
    }

    /// Open the in-card flag editor for the selected model's profile.
    fn start_flag_edit(&mut self) {
        if self.busy() {
            return;
        }
        let Some((name, flags)) = self
            .selected_model()
            .map(|m| (m.name.clone(), m.flags.clone()))
        else {
            return;
        };
        self.flag_edit = Some(FlagEdit {
            model_name: name,
            flags: split_flags(&flags),
            sel: 0,
            buf: None,
        });
        self.mode = Mode::EditFlags;
        self.status =
            "editing flags - [Enter] edit  [a] add  [d] del  [s] save  [x] reset  [Esc] cancel"
                .into();
    }

    /// Remove llmtune's drop-in (revert the unit to base config). Fast (a file
    /// op / one ssh call), so it runs inline rather than on a worker thread.
    fn do_unload(&mut self) {
        if self.busy() {
            return;
        }
        match transport::for_node(&self.node).and_then(|t| t.unload()) {
            Ok(()) => {
                self.status =
                    "unloaded - drop-in removed; unit reverts to base config on restart".into();
                self.refresh();
            }
            Err(e) => self.status = format!("unload failed: {e}"),
        }
    }

    /// Toggle the local llama-server: stop it to free the GPU for manual testing,
    /// start it to restore inference. Local-only (remote units need ssh).
    fn do_server_toggle(&mut self) {
        if self.busy() {
            return;
        }
        if !matches!(self.node.transport, Transport::Local) {
            self.status = "server control is local-only".into();
            return;
        }
        let unit = self.node.llama_unit.clone();
        let active = std::process::Command::new("systemctl")
            .args(["is-active", "--quiet", &unit])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let action = if active { "stop" } else { "start" };
        match crate::swap::sudo(&["systemctl", action, &unit]) {
            Ok(()) => {
                self.status = if active {
                    format!("stopped {unit} - GPU freed for testing ([s] to start)")
                } else {
                    format!("started {unit}")
                };
                self.refresh();
            }
            Err(e) => self.status = format!("server {action} failed: {e}"),
        }
    }

    /// Explicit server action (used for restart; stop/start go through the
    /// toggle). Routes via the transport so it works local or over SSH.
    fn do_server(&mut self, action: &str) {
        if self.busy() {
            return;
        }
        match transport::for_node(&self.node).and_then(|t| t.server_ctl(action)) {
            Ok(()) => {
                self.status = match action {
                    "restart" => format!("restarted {}", self.node.llama_unit),
                    "stop" => format!("stopped {} - GPU freed", self.node.llama_unit),
                    "start" => format!("started {}", self.node.llama_unit),
                    other => format!("server {other} ok"),
                };
                self.refresh();
            }
            Err(e) => self.status = format!("server {action} failed: {e}"),
        }
    }

    /// Toggle network exposure (bind 0.0.0.0 + open the firewall, or back to
    /// localhost). Reloads the served model to apply. Local node only.
    fn do_expose(&mut self) {
        if self.busy() {
            return;
        }
        if !matches!(self.node.transport, Transport::Local) {
            self.status = "network exposure is local-only".into();
            return;
        }
        let target = !crate::settings::exposed();
        let node = self.node.clone();
        self.status = if target {
            "exposing to the LAN...".into()
        } else {
            "restricting to localhost...".into()
        };
        self.started = Instant::now();
        // set_exposure reloads the served model (tens of seconds); run it off the
        // UI thread so rendering doesn't freeze. Completion is applied in poll().
        self.bg = BgState::Running(std::thread::spawn(move || {
            match nodeops::set_exposure(&node, target) {
                Ok((_applied, fw_ok, _backend)) => {
                    if target {
                        let port = nodeops::parse_bind(&node.llama_url).1;
                        let addr = crate::settings::lan_ip()
                            .map(|ip| format!("http://{ip}:{port}"))
                            .unwrap_or_else(|| format!("0.0.0.0:{port}"));
                        let auth = if crate::settings::api_key().is_some() {
                            "api-key required"
                        } else {
                            "UNAUTHENTICATED"
                        };
                        let fw = if fw_ok {
                            ""
                        } else {
                            " (open firewall manually)"
                        };
                        format!("exposed to the LAN at {addr} - {auth}{fw}")
                    } else {
                        "restricted to localhost".into()
                    }
                }
                Err(e) => format!("expose failed: {e}"),
            }
        }));
    }

    /// Re-probe the served endpoint so an open overlay reflects the current
    /// exposure/auth/served state in place. No-op when nothing is open.
    fn reprobe_endpoint(&mut self) {
        if self.endpoint_view.is_some() {
            self.endpoint_view = transport::for_node(&self.node)
                .and_then(|t| t.endpoint())
                .ok();
        }
    }

    /// Probe fresh and (re)open the endpoint overlay - used to return to it after
    /// the expose-confirm gate resolves.
    fn reopen_endpoint(&mut self) {
        self.endpoint_view = transport::for_node(&self.node)
            .and_then(|t| t.endpoint())
            .ok();
        if self.endpoint_view.is_some() {
            self.mode = Mode::Endpoint;
        }
    }

    /// `[e]` in the endpoint overlay: toggle LAN exposure. Turning OFF is
    /// harmless and applies immediately; turning ON binds `0.0.0.0` (possibly
    /// unauthenticated), so it routes through the same confirm gate as `n` -
    /// with a flag to reopen this overlay, refreshed, once the toggle lands.
    /// Probe (exposed, has_key) ONCE so the confirm-expose dialog can render from
    /// stored state instead of reading settings.toml twice per frame.
    fn probe_expose_state(&mut self) {
        self.expose_state = (
            crate::settings::exposed(),
            crate::settings::api_key().is_some(),
        );
    }

    fn expose_from_overlay(&mut self) {
        if self.busy() {
            return;
        }
        if !matches!(self.node.transport, Transport::Local) {
            self.status = "network exposure is local-only".into();
            return;
        }
        if crate::settings::exposed() {
            self.do_expose(); // restrict to localhost, refresh in place
        } else {
            self.reopen_endpoint_after_expose = true;
            self.probe_expose_state();
            self.mode = Mode::ConfirmExpose;
        }
    }

    /// Turn auth on (`Some` key) or off (`None`) from the endpoint overlay, apply
    /// it (reload), and re-probe so the overlay updates in place.
    fn do_api_key(&mut self, key: Option<String>) {
        if self.busy() {
            return;
        }
        if !matches!(self.node.transport, Transport::Local) {
            self.status = "auth control is local-only".into();
            return;
        }
        // Persisting the key is a fast file write; the restage (a full model
        // reload) is the slow part, so only that runs on the worker thread.
        if let Err(e) = crate::settings::set_api_key(key.as_deref()) {
            self.status = format!("auth: {e}");
            return;
        }
        let node = self.node.clone();
        let keyset = key.is_some();
        self.status = "applying auth (reloading model)...".into();
        self.started = Instant::now();
        self.bg = BgState::Running(std::thread::spawn(move || {
            let applied = nodeops::restage_served(&node);
            let tail = if applied {
                ""
            } else {
                " (applies on next load)"
            };
            if keyset {
                format!("auth ON - key required{tail}")
            } else {
                format!("auth OFF - keyless{tail}")
            }
        }));
    }

    /// `[i]` in the endpoint overlay: flip who defines the served model's
    /// identity. `to_harness` installs the stripped override template (the
    /// model obeys the harness's system prompt); `false` restores the embedded
    /// (branded) one. Applies by reloading, then re-probes the overlay in
    /// place. Local node only (the profile + template live on the node).
    fn do_identity(&mut self, to_harness: bool) {
        if self.busy() {
            return;
        }
        if !matches!(self.node.transport, Transport::Local) {
            self.status = "identity control is local-only".into();
            return;
        }
        let res = if to_harness {
            crate::identity::set_identity_harness(&self.node).map(|(applied, _path)| applied)
        } else {
            crate::identity::set_identity_model(&self.node)
        };
        match res {
            Ok(applied) => {
                let tail = if applied {
                    ""
                } else {
                    " (applies on next load)"
                };
                self.status = if to_harness {
                    format!("identity harness-controlled - override template applied{tail}")
                } else {
                    format!("identity model-branded - embedded template restored{tail}")
                };
                self.reprobe_endpoint();
                self.refresh();
            }
            Err(e) => self.status = format!("identity: {e}"),
        }
    }

    fn poll(&mut self) {
        // Drain live bench progress.
        if let Some(rx) = &self.bench_rx {
            while let Ok(p) = rx.try_recv() {
                self.bench_progress = Some(p);
            }
        }
        if let BenchState::Running(h) = &self.bench {
            if h.is_finished() {
                if let BenchState::Running(h) = std::mem::replace(&mut self.bench, BenchState::Idle)
                {
                    self.bench_rx = None;
                    self.bench_progress = None;
                    match h.join() {
                        Ok(Ok(rec)) => {
                            self.status = format!(
                                "bench: {:.1} gen tok/s, ttft {:.0} ms",
                                rec.perf.gen_tok_s, rec.perf.ttft_ms
                            );
                            self.bench = BenchState::Done;
                            self.refresh();
                        }
                        Ok(Err(e)) => {
                            self.status = format!("bench failed: {e}");
                            self.bench = BenchState::Failed;
                        }
                        Err(_) => {
                            self.status = "bench thread panicked".into();
                            self.bench = BenchState::Failed;
                        }
                    }
                }
            }
        }
        // Background task (expose/auth) done: apply its status line + the
        // fast UI-thread finalizers (reprobe an open overlay, refresh the list).
        if let BgState::Running(h) = &self.bg {
            if h.is_finished() {
                if let BgState::Running(h) = std::mem::replace(&mut self.bg, BgState::Idle) {
                    self.status = h
                        .join()
                        .unwrap_or_else(|_| "background task panicked".into());
                    self.reprobe_endpoint();
                    self.refresh();
                }
            }
        }
    }
}

fn move_sel(app: &mut NodeApp, delta: i32) {
    let len = app.visible().len();
    if len == 0 {
        app.sel = 0;
        return;
    }
    app.sel = (app.sel as i32 + delta).clamp(0, len as i32 - 1) as usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Transport;
    use ratatui::backend::TestBackend;

    fn node() -> Node {
        Node {
            name: "localhost".into(),
            host: None,
            transport: Transport::Local,
            ssh_user: None,
            ssh_key: None,
            models_dir: "/nonexistent".into(),
            llama_unit: "llama-server.service".into(),
            llama_url: "http://127.0.0.1:1".into(),
            power_cmd: None,
        }
    }

    fn node_app() -> NodeApp {
        NodeApp {
            node: node(),
            models: vec![],
            history: vec![],
            sel: 0,
            filter: String::new(),
            flag_edit: None,
            bench: BenchState::Idle,
            bench_rx: None,
            bench_progress: None,
            bg: BgState::Idle,
            mode: Mode::Normal,
            server_active: false,
            auth_off_armed: false,
            expose_state: (false, false),
            status: "test".into(),
            started: Instant::now(),
            endpoint_view: None,
            reopen_endpoint_after_expose: false,
            doctor_view: None,
            profiles_view: None,
            builds_view: None,
            fleet_nodes: vec![],
            clusters: vec![],
        }
    }

    /// A FleetApp with `n` canned nodes/cards and NO live pollers: `polled_at`
    /// is fresh, so calling poll() in a test won't dispatch worker threads.
    fn fleet_app_n(n: usize) -> FleetApp {
        let nodes: Vec<Node> = (0..n)
            .map(|i| {
                let mut nd = node();
                nd.name = format!("node-{i:02}");
                if i > 0 {
                    nd.host = Some(format!("192.0.2.{}", 10 + i));
                    nd.transport = Transport::Ssh;
                }
                nd
            })
            .collect();
        let cards = nodes
            .iter()
            .map(|nd| {
                let mut c = NodeCard::new(&nd.name);
                c.status = NodeStatus {
                    name: nd.name.clone(),
                    reachable: true,
                    healthy: true,
                    benchmarking: false,
                    served: Some("Qwen3-30B-A3B-IQ2.gguf".into()),
                    models: 3,
                    last_model: Some("Qwen3-30B-A3B-IQ2.gguf".into()),
                    last_gen_tok_s: Some(41.2),
                };
                c.probed = true;
                c.telem = Some(Telemetry {
                    gfxclk_mhz: 2230,
                    uclk_mhz: 450,
                    temp_c: 61.0,
                    power_w: Some(38.1),
                    ..Default::default()
                });
                c
            })
            .collect();
        let (poll_tx, poll_rx) = std::sync::mpsc::channel();
        let (load_tx, load_rx) = std::sync::mpsc::channel();
        let (eff_tx, eff_rx) = std::sync::mpsc::channel();
        FleetApp {
            cfg: Config {
                nodes,
                clusters: vec![],
                netboot: None,
            },
            cards,
            sel: 0,
            status: "test".into(),
            bench_all_armed: false,
            eff_armed: false,
            alerts_cfg: crate::settings::AlertsCfg::default(),
            events: EventLog::new(EVENT_LOG_CAP),
            eff: EffBench::Idle,
            compact: false,
            mode: FleetMode::Normal,
            poll_tx,
            poll_rx,
            load_tx,
            load_rx,
            eff_tx,
            eff_rx,
            inflight: vec![None; n],
            polled_at: vec![Some(Instant::now()); n],
            started: Instant::now(),
        }
    }

    /// A full cockpit over `fleet` with an empty, non-fetching models pane
    /// (deterministic: no worker threads until a test dispatches one).
    fn cockpit(fleet: FleetApp) -> CockpitApp {
        let mut na = node_app();
        if let Some(nd) = fleet.cfg.nodes.get(fleet.sel) {
            na.node = nd.clone();
        }
        na.fleet_nodes = fleet.cfg.nodes.clone();
        na.clusters = fleet.cfg.clusters.clone();
        CockpitApp {
            headline: fleet.status.clone(),
            prev_fleet_status: fleet.status.clone(),
            prev_node_status: na.status.clone(),
            fleet,
            node: na,
            focus: Focus::Nodes,
            fetch: ModelsFetch::Idle,
            fetch_queued: false,
            was_loading: false,
            config_path: None,
            netboot: None,
        }
    }

    fn cockpit_n(n: usize) -> CockpitApp {
        cockpit(fleet_app_n(n))
    }

    /// Shim for the fleet-side render tests: draw the cockpit around a bare
    /// FleetApp with a scratch (empty) models pane, nodes focus.
    fn draw_fleet(f: &mut Frame, app: &FleetApp) {
        let mut na = node_app();
        if let Some(nd) = app.cfg.nodes.get(app.sel) {
            na.node = nd.clone();
        }
        draw_cockpit(
            f,
            &CockpitView {
                fleet: app,
                node: &na,
                focus: Focus::Nodes,
                headline: &app.status,
                netboot: None,
            },
        );
    }

    /// Shim for the model-pane render tests: draw the cockpit around a bare
    /// NodeApp with a scratch single-node fleet, models focus.
    fn draw_node(f: &mut Frame, app: &NodeApp) {
        let fl = fleet_app_n(1);
        draw_cockpit(
            f,
            &CockpitView {
                fleet: &fl,
                node: app,
                focus: Focus::Models,
                headline: &app.status,
                netboot: None,
            },
        );
    }

    #[test]
    fn node_renders_small_terminals() {
        for (w, h) in [(80, 24), (40, 12), (20, 8)] {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            let app = node_app();
            term.draw(|f| draw_node(f, &app)).unwrap();
        }
    }

    fn sample_netboot_view() -> NetbootView {
        NetbootView {
            server_ip: "192.168.1.91".into(),
            http_base: "http://192.168.1.91:8090".into(),
            interface: "enp6s0".into(),
            leases_path: "/var/lib/misc/dnsmasq.leases".into(),
            console_port: 6666,
            services: vec![
                ("dnsmasq", "active".into()),
                ("nfs-server", "active".into()),
                ("llmtune-netboot.service", "inactive".into()),
            ],
            boards: vec![
                NetbootBoard {
                    name: "bc250-01".into(),
                    ip: "192.168.1.120".into(),
                    mac: "58:11:22:aa:bb:cc".into(),
                    registered: true,
                },
                NetbootBoard {
                    name: "bc250-02".into(),
                    ip: "192.168.1.121".into(),
                    mac: "58:11:22:aa:bb:c1".into(),
                    registered: false,
                },
            ],
        }
    }

    #[test]
    fn netboot_view_counts_only_unregistered() {
        assert_eq!(sample_netboot_view().unregistered(), 1);
    }

    #[test]
    fn netboot_dash_renders_server_nodes_and_boards() {
        let mut na = node_app();
        na.mode = Mode::Netboot;
        let fl = fleet_app_n(2);
        let dash = netboot::NetbootDash::new(Some(sample_netboot_view()));
        let mut term = Terminal::new(TestBackend::new(110, 34)).unwrap();
        term.draw(|f| {
            draw_cockpit(
                f,
                &CockpitView {
                    fleet: &fl,
                    node: &na,
                    focus: Focus::Nodes,
                    headline: &na.status,
                    netboot: Some(&dash),
                },
            )
        })
        .unwrap();
        let t = buf_text(&term);
        assert!(t.contains("BOOT SERVER  192.168.1.91"), "{t}");
        assert!(t.contains("dnsmasq") && t.contains("nfs-server"));
        assert!(t.contains("NODES (2)"), "{t}");
        assert!(t.contains("node-00") && t.contains("node-01"), "{t}");
        assert!(t.contains("llama up"), "node state column: {t}");
        assert!(t.contains("41.2"), "last-bench tok/s column: {t}");
        assert!(t.contains("bc250-01") && t.contains("bc250-02"), "{t}");
        assert!(t.contains("1 new"), "unregistered count in header: {t}");
        assert!(t.to_lowercase().contains("register"), "action hint: {t}");
        assert!(t.contains("arm"), "arm/boot key hints: {t}");
    }

    #[test]
    fn netboot_dash_renders_unconfigured_and_console_pane() {
        let mut na = node_app();
        na.mode = Mode::Netboot;
        let fl = fleet_app_n(1);
        let mut dash = netboot::NetbootDash::new(None);
        dash.console = Some(netboot::ConsolePane::fake("bc250-x"));
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| {
            draw_cockpit(
                f,
                &CockpitView {
                    fleet: &fl,
                    node: &na,
                    focus: Focus::Nodes,
                    headline: &na.status,
                    netboot: Some(&dash),
                },
            )
        })
        .unwrap();
        let t = buf_text(&term);
        assert!(t.contains("netboot not configured"), "{t}");
        assert!(t.contains("NODES (1)"), "nodes table still renders: {t}");
        assert!(t.contains("CONSOLE bc250-x"), "console pane header: {t}");
    }

    #[test]
    fn node_renders_overlays() {
        for mode in [Mode::Help, Mode::ConfirmSwap(0)] {
            let mut term = Terminal::new(TestBackend::new(60, 20)).unwrap();
            let mut app = node_app();
            app.mode = mode;
            term.draw(|f| draw_node(f, &app)).unwrap();
        }
    }

    #[test]
    fn node_renders_new_overlays() {
        // Each read-only / confirm overlay must render without panicking, both
        // empty and with a little data.
        let mut a = node_app();
        a.models = vec![mi("Qwen3-30B-A3B-IQ2.gguf")];
        a.history = vec![Record {
            ts: 1_700_000_000,
            node: "localhost".into(),
            model: "Qwen3-30B-A3B-IQ2.gguf".into(),
            arch: "llama".into(),
            quant: Some("IQ2_XXS".into()),
            ctx: 32768,
            profile: "_default".into(),
            perf: crate::bench::PerfBench {
                prompt_tok_s: 120.0,
                gen_tok_s: 42.0,
                ttft_ms: 200.0,
                ..Default::default()
            },
            notes: String::new(),
            build: None,
            cluster: None,
            cluster_members: vec![],
        }];
        a.doctor_view = Some(vec![]);
        a.profiles_view = Some(profile::load().unwrap_or_default());
        a.builds_view = Some(vec![]);
        for mode in [
            Mode::History,
            Mode::Leaderboard,
            Mode::Compare,
            Mode::Doctor,
            Mode::Profiles,
            Mode::Builds,
            Mode::Clusters,
            Mode::ConfirmSwapAll,
        ] {
            a.mode = mode;
            let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
            term.draw(|f| draw_node(f, &a)).unwrap();
        }
    }

    #[test]
    fn fleet_renders() {
        // the dashboard must render for any node count on small terminals too
        for (w, h) in [(80, 24), (40, 12), (20, 8)] {
            for n in [1, 3, 12] {
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                let app = fleet_app_n(n);
                term.draw(|f| draw_fleet(f, &app)).unwrap();
            }
        }
    }

    #[test]
    fn fleet_rail_card_shows_telemetry_and_model() {
        // The SELECTED node's rail card is expanded: full telemetry + detail.
        let app = fleet_app_n(2);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("2230 MHz"), "card shows the GPU clock");
        assert!(txt.contains("38.1 W"), "card shows watts");
        assert!(txt.contains("61°C"), "card shows the temperature");
        assert!(txt.contains("41.2 t/s"), "card shows gen throughput");
        assert!(txt.contains("Qwen3-30B"), "card shows the served model");
        assert!(txt.contains("llama up"), "card shows llama health");
        assert!(txt.contains("node-00") && txt.contains("node-01"));
        // selecting the remote node: its expanded card carries the IP + transport
        let mut app = fleet_app_n(2);
        app.sel = 1;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("192.0.2.11"), "card shows the node IP: {txt}");
        assert!(txt.contains("(ssh)"), "card shows the transport");
    }

    #[test]
    fn fleet_rail_scrolls_to_keep_selection_visible() {
        // 12 nodes (a full 4U chassis) on a 24-row terminal: not all cards fit,
        // so the rail must scroll with the selection instead of panicking or
        // pinning to the top.
        let render = |sel: usize| -> String {
            let mut app = fleet_app_n(12);
            app.sel = sel;
            let mut term = Terminal::new(TestBackend::new(90, 24)).unwrap();
            term.draw(|f| draw_fleet(f, &app)).unwrap();
            buf_text(&term)
        };
        let top = render(0);
        assert!(top.contains("node-00"));
        assert!(
            !top.contains("node-11"),
            "last card must be off-rail when the first is selected"
        );
        let bottom = render(11);
        assert!(
            bottom.contains("node-11"),
            "selecting the last card must scroll it into view"
        );
        // on a tall terminal all 12 cards fit at once (12 x 4 rows + chrome)
        let mut app = fleet_app_n(12);
        app.sel = 0;
        let mut term = Terminal::new(TestBackend::new(90, 60)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let tall = buf_text(&term);
        assert!(tall.contains("node-00") && tall.contains("node-11"));
    }

    #[test]
    fn fleet_offline_card_shows_offline_without_stale_numbers() {
        let mut app = fleet_app_n(1);
        app.cards[0].status = NodeStatus::unreachable("node-00");
        app.cards[0].telem = None;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("OFFLINE"), "offline card is labelled");
        assert!(
            !txt.contains("MHz") && !txt.contains("t/s"),
            "an offline card must not show stale clock/throughput numbers"
        );
    }

    #[test]
    fn fleet_rollup_strip_sums_reachable_only() {
        // 3 nodes: two live with distinct watts/toks/temps, one unreachable
        // whose (stale) numbers must be excluded from every rollup figure.
        let mut app = fleet_app_n(3);
        app.cards[1].telem = Some(Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: 75.0,
            power_w: Some(40.0),
            ..Default::default()
        });
        app.cards[1].status.last_gen_tok_s = Some(20.0);
        app.cards[2].status = NodeStatus::unreachable("node-02");
        app.cards[2].status.last_gen_tok_s = Some(999.0);
        app.cards[2].telem = Some(Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: 99.0,
            power_w: Some(999.0),
            ..Default::default()
        });
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("78.1 W"), "total watts = 38.1 + 40.0 only");
        assert!(txt.contains("61.2 t/s"), "summed tok/s = 41.2 + 20.0 only");
        assert!(
            txt.contains("hot node-01 75°C"),
            "hottest REACHABLE node, not the dead 99C one"
        );
        assert!(txt.contains("2/3"), "reachable count");
        // the strip must render (with the right N/M) for any MULTI-node rack
        // (a single node suppresses it - see rollup_strip_suppressed_for_
        // single_node)
        for n in [2usize, 3, 12] {
            let app = fleet_app_n(n);
            let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
            term.draw(|f| draw_fleet(f, &app)).unwrap();
            let txt = buf_text(&term);
            assert!(txt.contains(&format!("{n}/{n}")), "strip N/M at n={n}");
        }
    }

    #[test]
    fn rollup_strip_suppressed_for_single_node() {
        // One node: the fleet "aggregate" is exactly the numbers the expanded
        // rail card already shows, so the strip row is dropped entirely. The
        // health/alert banner stays.
        let app = fleet_app_n(1);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(!txt.contains("1/1 up"), "no rollup strip at n=1: {txt}");
        assert!(txt.contains("all nodes healthy"), "banner stays: {txt}");
        // ... including the red alert path
        let mut app = fleet_app_n(1);
        app.cards[0].telem.as_mut().unwrap().temp_c = 81.0;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("ALERT:"), "alert banner stays at n=1: {txt}");
        assert!(!txt.contains("1/1 up"), "still no strip while alerting");
        // Two nodes: the totals are real information again - strip present.
        let app = fleet_app_n(2);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("2/2 up"), "strip present at n=2: {txt}");
        assert!(txt.contains("82.4 t/s"), "summed tok/s on the strip: {txt}");
        assert!(txt.contains("76.2 W"), "summed watts on the strip: {txt}");
    }

    #[test]
    fn fleet_banner_alerts_and_clears() {
        // all healthy -> quiet green, no ALERT
        let app = fleet_app_n(3);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("all nodes healthy"));
        assert!(!txt.contains("ALERT"));
        // one hot, one unreachable, one llama-down-with-a-model -> red banner
        // naming every offender and its reason
        let mut app = fleet_app_n(3);
        app.cards[0].telem.as_mut().unwrap().temp_c = 81.0;
        app.cards[1].status = NodeStatus::unreachable("node-01");
        app.cards[1].telem = None;
        app.cards[2].status.healthy = false; // served is still set
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("ALERT:"));
        assert!(txt.contains("node-00 81°C"), "hot node named with temp");
        assert!(txt.contains("node-01 unreachable"));
        assert!(txt.contains("node-02 llama down"));
        assert!(!txt.contains("all nodes healthy"));
    }

    #[test]
    fn fleet_card_and_rollup_show_tok_per_watt() {
        // 41.2 t/s / 38.1 W = 1.08 -> "1.1 t/s/W" on the expanded rail card
        // and the rollup strip
        let app = fleet_app_n(2);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(buf_text(&term).contains("1.1 t/s/W"));
        // no power reading anywhere -> the metric disappears (never a NaN/inf)
        let mut app = fleet_app_n(2);
        for c in &mut app.cards {
            c.telem.as_mut().unwrap().power_w = None;
        }
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(!buf_text(&term).contains("t/s/W"));
    }

    #[test]
    fn card_eff_guards_missing_and_zero_operands() {
        let mut app = fleet_app_n(1);
        assert_eq!(fleet::card_eff(&app.cards[0]), Some(41.2 / 38.1));
        app.cards[0].telem.as_mut().unwrap().power_w = Some(0.0);
        assert_eq!(fleet::card_eff(&app.cards[0]), None, "zero watts");
        app.cards[0].telem = None;
        assert_eq!(fleet::card_eff(&app.cards[0]), None, "no telemetry");
        let mut app = fleet_app_n(1);
        app.cards[0].status.last_gen_tok_s = None;
        assert_eq!(fleet::card_eff(&app.cards[0]), None, "no bench figure");
    }

    #[test]
    fn fleet_jump_keys_select_cards() {
        let mut app = fleet_app_n(12);
        app.jump('3');
        assert_eq!(app.sel, 2);
        app.jump('9');
        assert_eq!(app.sel, 8);
        app.jump('0');
        assert_eq!(app.sel, 9, "0 jumps to the 10th card");
        app.jump('1');
        assert_eq!(app.sel, 0);
        // out-of-range digits are a no-op, on any rack size
        let mut app = fleet_app_n(3);
        app.sel = 1;
        app.jump('9');
        assert_eq!(app.sel, 1);
        app.jump('0');
        assert_eq!(app.sel, 1);
        let mut app = fleet_app_n(1);
        app.jump('1');
        assert_eq!(app.sel, 0);
        app.jump('2');
        assert_eq!(app.sel, 0);
        // the footer documents the keys
        let app = fleet_app_n(3);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(buf_text(&term).contains("[1-9,0] jump"));
    }

    #[test]
    fn fleet_poll_dispatches_workers_without_blocking() {
        // poll() must return immediately (drain + dispatch only) even when the
        // node is an unreachable SSH box; the probe runs on a worker thread.
        let mut app = fleet_app_n(2);
        // Both nodes are unreachable SSH boxes (TEST-NET, never answer) so neither
        // probe can complete during the test: the in-flight workers stay pending
        // across the second poll, and node 0 isn't a fast local probe that folds
        // between the two polls (which would spuriously clear its inflight slot).
        for nd in app.cfg.nodes.iter_mut() {
            nd.host = Some("192.0.2.1".into());
            nd.transport = crate::config::Transport::Ssh;
        }
        app.polled_at = vec![None; 2]; // due now
        let t0 = Instant::now();
        app.poll();
        // A truly blocking poll would wait on the worker's ssh probe, which is
        // the ~8s ConnectTimeout; anything well under that proves it dispatched
        // and returned. Use a wide bound (4s) so scheduler jitter under heavy
        // parallel test load can't flake a correctly non-blocking poll.
        assert!(
            t0.elapsed() < Duration::from_secs(4),
            "poll() blocked the event thread ({}ms)",
            t0.elapsed().as_millis()
        );
        assert!(
            app.inflight.iter().all(|f| f.is_some()),
            "a worker must be in flight per node"
        );
        // a second call while in flight must not block or double-dispatch
        let before: Vec<_> = app.inflight.clone();
        app.poll();
        assert_eq!(before, app.inflight);
    }

    #[test]
    fn fleet_poll_folds_results_and_drops_stale_telemetry() {
        let mut app = fleet_app_n(1);
        app.cards[0].telem = None;
        let up = NodeStatus {
            name: "node-00".into(),
            reachable: true,
            healthy: true,
            benchmarking: false,
            served: Some("m.gguf".into()),
            models: 1,
            last_model: None,
            last_gen_tok_s: Some(40.0),
        };
        let reading = Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: 61.5,
            power_w: Some(38.1),
            ..Default::default()
        };
        app.poll_tx.send((0, up.clone(), Some(reading))).unwrap();
        app.poll();
        assert!(app.cards[0].telem.is_some());
        assert!(app.cards[0].status.healthy);
        // failed telemetry polls below the threshold keep the last reading...
        for _ in 0..TELEM_STALE_AFTER - 1 {
            app.poll_tx.send((0, up.clone(), None)).unwrap();
        }
        app.poll();
        assert!(app.cards[0].telem.is_some(), "kept under the threshold");
        // ...and the Nth consecutive miss drops it (no eternal stale card)
        app.poll_tx.send((0, up.clone(), None)).unwrap();
        app.poll();
        assert!(app.cards[0].telem.is_none(), "stale reading must drop");
        // an out-of-range index (impossible in practice) must not panic
        app.poll_tx.send((99, up, None)).unwrap();
        app.poll();
    }

    // -- Tier 2: ring buffer ------------------------------------------------

    #[test]
    fn ring_is_bounded_and_appends_in_order() {
        let mut r = Ring::new(60);
        assert!(r.is_empty());
        assert_eq!(r.min_max(), None);
        assert_eq!(r.last(), None);
        for i in 0..100 {
            r.push(i as f64);
        }
        assert_eq!(r.len(), 60, "capped at capacity");
        assert_eq!(r.last(), Some(99.0));
        // oldest surviving sample is 100 - 60 = 40
        assert_eq!(r.tail(60).first().copied(), Some(40.0));
        assert_eq!(r.min_max(), Some((40.0, 99.0)));
        // tail(n) keeps the LAST n, oldest first
        assert_eq!(r.tail(3), vec![97.0, 98.0, 99.0]);
    }

    #[test]
    fn ring_drops_non_finite_samples() {
        let mut r = Ring::new(8);
        r.push(f64::NAN);
        r.push(f64::INFINITY);
        r.push(f64::NEG_INFINITY);
        assert!(r.is_empty(), "non-finite samples never enter the ring");
        r.push(61.0);
        assert_eq!(r.min_max(), Some((61.0, 61.0)));
    }

    #[test]
    fn spark_str_normalizes_and_bounds() {
        use widgets::spark_str;
        assert_eq!(spark_str(&[], 10), "", "empty ring renders empty");
        assert_eq!(spark_str(&[1.0], 0), "");
        // flat series: mid bars, not a panic or divide-by-zero
        let flat = spark_str(&[5.0, 5.0, 5.0], 10);
        assert_eq!(flat.chars().count(), 3);
        assert!(flat.chars().all(|c| c == '▄'), "{flat}");
        // rising series: min maps low, max maps high
        let rise = spark_str(&[1.0, 2.0, 3.0, 4.0], 10);
        assert!(rise.starts_with('▁') && rise.ends_with('█'), "{rise}");
        // width-limited: keeps the LAST samples
        let tail = spark_str(&[0.0, 0.0, 0.0, 1.0, 2.0], 2);
        assert_eq!(tail.chars().count(), 2);
        assert!(tail.ends_with('█'), "{tail}");
    }

    // -- Tier 2: sparklines in the expanded rail card ------------------------

    #[test]
    fn fleet_expanded_card_renders_trend_sparklines() {
        // empty rings: the trend rows are absent, and nothing panics
        let app = fleet_app_n(1);
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(
            !buf_text(&term).contains("temp~"),
            "no trend rows when empty"
        );
        // partial ring (3 samples): rows render with a real range
        let mut app = fleet_app_n(1);
        for v in [50.0, 60.0, 70.0] {
            app.cards[0].temp_hist.push(v);
        }
        app.cards[0].power_hist.push(38.0);
        app.cards[0].tok_hist.push(41.2);
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("temp~"), "temp trend row");
        assert!(txt.contains("power~"), "power trend row");
        assert!(txt.contains("tok/s~"), "tok/s trend row");
        assert!(txt.contains("50-70"), "temp min-max range");
        // [50,60,70] normalizes to low/mid/high bars on the temp row
        assert!(txt.contains("▁▅█"), "bars rendered");
        // a single-sample power ring shows a flat value, not a 38-38 range
        assert!(txt.contains("38"), "power range");
    }

    #[test]
    fn fleet_expanded_card_renders_uma_mem() {
        // No mem fields on the wire (older node / non-BC250): dim n/a, no crash.
        let app = fleet_app_n(1);
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(
            txt.contains("mem     n/a"),
            "mem shows n/a when absent: {txt}"
        );
        // Populated: used/total in GiB with the pressure percentage.
        // 10340/1024 = 10.1, 15680/1024 = 15.3, 10340/15680 = 66%.
        let mut app = fleet_app_n(1);
        if let Some(t) = app.cards[0].telem.as_mut() {
            t.mem_used_mib = Some(10340);
            t.mem_total_mib = Some(15680);
        }
        // ...and a mem~ trend row once samples land
        for v in [40.0, 55.0, 66.0] {
            app.cards[0].mem_hist.push(v);
        }
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("10.1 / 15.3 GiB"), "mem used/total: {txt}");
        assert!(txt.contains("(66%)"), "mem pressure percent: {txt}");
        assert!(txt.contains("mem~"), "mem trend row: {txt}");
        assert!(txt.contains("40-66"), "mem trend min-max range: {txt}");
    }

    #[test]
    fn fleet_poll_fold_appends_mem_history() {
        let mut app = fleet_app_n(1);
        let up = NodeStatus {
            name: "node-00".into(),
            reachable: true,
            healthy: true,
            benchmarking: false,
            served: Some("m.gguf".into()),
            models: 1,
            last_model: Some("m.gguf".into()),
            last_gen_tok_s: None,
        };
        let t = |mem: Option<(u32, u32)>| Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: 60.0,
            mem_used_mib: mem.map(|(u, _)| u),
            mem_total_mib: mem.map(|(_, tot)| tot),
            ..Default::default()
        };
        // 8000/16000 = 50%; a reading without mem appends nothing; a zero
        // total (garbage) must not divide.
        app.poll_tx
            .send((0, up.clone(), Some(t(Some((8000, 16000))))))
            .unwrap();
        app.poll_tx.send((0, up.clone(), Some(t(None)))).unwrap();
        app.poll_tx
            .send((0, up.clone(), Some(t(Some((1, 0))))))
            .unwrap();
        app.poll();
        assert_eq!(app.cards[0].mem_hist.tail(10), vec![50.0]);
    }

    #[test]
    fn fleet_poll_fold_appends_history_rings() {
        let mut app = fleet_app_n(1);
        let up = NodeStatus {
            name: "node-00".into(),
            reachable: true,
            healthy: true,
            benchmarking: false,
            served: Some("m.gguf".into()),
            models: 1,
            last_model: Some("m.gguf".into()),
            last_gen_tok_s: Some(40.0),
        };
        let t = |c: f64, w: Option<f64>| Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: c,
            power_w: w,
            ..Default::default()
        };
        app.poll_tx
            .send((0, up.clone(), Some(t(60.0, Some(38.0)))))
            .unwrap();
        app.poll_tx
            .send((0, up.clone(), Some(t(62.0, None))))
            .unwrap();
        // a failed telemetry poll appends nothing (no fake samples)
        app.poll_tx.send((0, up.clone(), None)).unwrap();
        app.poll();
        assert_eq!(app.cards[0].temp_hist.tail(10), vec![60.0, 62.0]);
        assert_eq!(app.cards[0].power_hist.tail(10), vec![38.0]);
        // tok/s appended once: the same bench figure repeated every poll must
        // not fill the ring with duplicates...
        assert_eq!(app.cards[0].tok_hist.tail(10), vec![40.0]);
        // ...but a NEW bench figure lands
        let mut up2 = up.clone();
        up2.last_gen_tok_s = Some(43.5);
        app.poll_tx
            .send((0, up2, Some(t(63.0, Some(39.0)))))
            .unwrap();
        app.poll();
        assert_eq!(app.cards[0].tok_hist.tail(10), vec![40.0, 43.5]);
    }

    // -- Tier 2: compact / expanded rail ------------------------------------

    #[test]
    fn fleet_rail_renders_compact_and_expanded() {
        // both modes, every rack size, small terminals: no layout panic
        for compact in [false, true] {
            for (w, h) in [(100, 30), (80, 24), (40, 12), (20, 8)] {
                for n in [1, 3, 12] {
                    let mut app = fleet_app_n(n);
                    app.compact = compact;
                    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                    term.draw(|f| draw_fleet(f, &app)).unwrap();
                }
            }
        }
        // compact: a full 12-node chassis fits one 24-row screen, one line each
        let mut app = fleet_app_n(12);
        app.compact = true;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(
            txt.contains("node-00") && txt.contains("node-11"),
            "all 12 visible"
        );
        assert!(
            txt.contains("61°C") && txt.contains("41.2 t/s"),
            "temp + tok/s on the row"
        );
        // expanded on the same screen: the last cards scroll off instead
        app.compact = false;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(!buf_text(&term).contains("node-11"));
    }

    #[test]
    fn fleet_compact_rail_scrolls_and_jumps() {
        // selection + scroll work in compact mode too: an 11-row terminal only
        // fits a few rows, so selecting the last node must scroll it into view
        let mut app = fleet_app_n(12);
        app.compact = true;
        app.sel = 11;
        let mut term = Terminal::new(TestBackend::new(100, 11)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("node-11"), "selected row scrolled into view");
        assert!(txt.contains("node-09"), "window ends at the selection");
        // node-01 is neither in the 3-row window nor the rollup strip
        // (node-00 IS: it's the fixture's hottest node)
        assert!(!txt.contains("node-01"), "off-window row scrolled off");
        // jump keys are mode-independent
        app.jump('5');
        assert_eq!(app.sel, 4);
        // OFFLINE marking carries into the compact row
        let mut app = fleet_app_n(2);
        app.compact = true;
        app.cards[1].status = NodeStatus::unreachable("node-01");
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(buf_text(&term).contains("OFFLINE"));
    }

    // -- Tier 2: dashboard-driven load (now the always-on models pane) ------

    #[test]
    fn models_pane_lists_selects_and_marks_served() {
        // The middle pane replaces the old load-picker overlay: the focused
        // node's model list is always on screen, cursor + served marker.
        let mut c = cockpit_n(3);
        c.focus = Focus::Models;
        c.node.models = vec![mi("alpha.gguf"), mi("bravo.gguf"), mi("charlie.gguf")];
        c.node.models[2].served = true;
        c.node.sel = 1;
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("models (3)"), "pane title carries the count");
        assert!(txt.contains("▸ models"), "focused pane carries the marker");
        assert!(txt.contains("alpha.gguf") && txt.contains("charlie.gguf"));
        assert!(
            txt.contains("▸   bravo.gguf"),
            "cursor on the selection: {txt}"
        );
        assert!(txt.contains("● charlie.gguf"), "served model marked");
        assert!(
            txt.contains("[Enter] load"),
            "footer documents the load key"
        );
        // an empty list renders the card's hint, not a panic
        c.node.models.clear();
        c.node.sel = 0;
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        assert!(buf_text(&term).contains("no models found"));
        // and it survives a tiny terminal
        let mut term = Terminal::new(TestBackend::new(20, 8)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
    }

    #[test]
    fn fleet_loading_card_shows_spinner() {
        let mut app = fleet_app_n(2);
        app.cards[1].loading = Some("m.gguf".into());
        // the expanded (selected) card carries a labelled loading row
        app.sel = 1;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("loading"), "loading row label");
        assert!(txt.contains("m.gguf..."), "loading row names the model");
        // compact row too
        app.compact = true;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(buf_text(&term).contains("loading m.gguf"));
    }

    #[test]
    fn dispatch_load_guards_against_double_dispatch() {
        // node 1 is SSH-to-TEST-NET in the fixture: its worker can never
        // complete inside the test, so the guard state is stable to observe.
        let mut app = fleet_app_n(2);
        app.cards[1].loading = Some("already.gguf".into());
        assert_eq!(
            app.dispatch_load(&[1], "m.gguf"),
            0,
            "a mid-load node must never get a second load"
        );
        assert_eq!(
            app.cards[1].loading.as_deref(),
            Some("already.gguf"),
            "the in-flight load is untouched"
        );
        // a free node dispatches exactly once, then guards
        app.cards[1].loading = None;
        assert_eq!(app.dispatch_load(&[1], "m.gguf"), 1);
        assert_eq!(app.cards[1].loading.as_deref(), Some("m.gguf"));
        assert_eq!(
            app.dispatch_load(&[1], "n.gguf"),
            0,
            "second dispatch refused"
        );
        // out-of-range targets are skipped, not a panic
        assert_eq!(app.dispatch_load(&[99], "m.gguf"), 0);
    }

    #[test]
    fn enter_confirms_and_dispatches_load_on_focused_node() {
        // Models-pane Enter -> confirm dialog -> y dispatches the load on the
        // FOCUSED node's worker (node 1 is SSH-to-TEST-NET: stable to observe).
        let mut c = cockpit_n(2);
        c.fleet.sel = 1;
        c.node.node = c.fleet.cfg.nodes[1].clone();
        c.node.models = vec![mi("m.gguf")];
        c.focus = Focus::Models;
        assert_eq!(c.on_key(KeyCode::Enter), KeyOutcome::Continue);
        assert!(
            matches!(c.node.mode, Mode::ConfirmSwap(0)),
            "confirm opened"
        );
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("confirm swap") && txt.contains("m.gguf"));
        c.on_key(KeyCode::Char('y'));
        assert!(matches!(c.node.mode, Mode::Normal), "confirm closed");
        assert_eq!(c.fleet.cards[1].loading.as_deref(), Some("m.gguf"));
        assert!(
            c.fleet.status.contains("loading `m.gguf` on 1 node(s)"),
            "{}",
            c.fleet.status
        );
        // a second confirm on the same node is refused (already loading)
        c.on_key(KeyCode::Enter);
        c.on_key(KeyCode::Char('y'));
        assert_eq!(c.fleet.cards[1].loading.as_deref(), Some("m.gguf"));
        assert!(
            c.fleet.status.contains("no load dispatched"),
            "{}",
            c.fleet.status
        );
        // cancelling the confirm dispatches nothing
        c.fleet.cards[1].loading = None;
        c.on_key(KeyCode::Enter);
        c.on_key(KeyCode::Char('n'));
        assert!(c.fleet.cards[1].loading.is_none());
        assert_eq!(c.node.status, "cancelled");
    }

    #[test]
    fn swap_all_confirms_and_dispatches_every_node() {
        // A (load-all) -> confirm -> y dispatches one load worker per node.
        // All nodes SSH-to-TEST-NET so no real local load can run in a test.
        let mut c = cockpit_n(3);
        c.fleet.cfg.nodes[0].transport = Transport::Ssh;
        c.fleet.cfg.nodes[0].host = Some("192.0.2.9".into());
        c.node.models = vec![mi("m.gguf")];
        c.focus = Focus::Models;
        c.on_key(KeyCode::Char('A'));
        assert!(matches!(c.node.mode, Mode::ConfirmSwapAll));
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        assert!(buf_text(&term).contains("confirm swap-all"));
        c.on_key(KeyCode::Char('y'));
        assert!(c
            .fleet
            .cards
            .iter()
            .all(|card| card.loading.as_deref() == Some("m.gguf")));
        assert!(
            c.fleet.status.contains("loading `m.gguf` on 3 node(s)"),
            "{}",
            c.fleet.status
        );
    }

    // -- Cockpit: unified layout + focus ------------------------------------

    #[test]
    fn cockpit_renders_all_sizes_and_never_hides_the_card() {
        // Every size / rack / focus combination renders without panicking,
        // and the model card (the selected model's name) is ALWAYS present -
        // narrow terminals drop the rail, then the list, never the card.
        for (w, h) in [(120, 34), (100, 30), (80, 24), (60, 18), (40, 12), (20, 8)] {
            for n in [1usize, 3, 12] {
                for focus in [Focus::Nodes, Focus::Models] {
                    let mut c = cockpit_n(n);
                    c.node.models = vec![mi("m.gguf")];
                    c.focus = focus;
                    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                    term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
                    let txt = buf_text(&term);
                    assert!(
                        txt.contains("m.gguf"),
                        "model card missing at {w}x{h} n={n}"
                    );
                }
            }
        }
        // wide terminal: all three panes visible at once (nodes, models, card)
        let mut c = cockpit_n(3);
        c.node.models = vec![mi("alpha.gguf")];
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("nodes (3)"), "node rail present");
        assert!(txt.contains("models (1)"), "model list present");
        assert!(txt.contains("size"), "model card present (spec rows)");
        assert!(txt.contains("2230"), "rail telemetry present");
    }

    #[test]
    fn cockpit_focus_moves_with_tab_and_arrows() {
        let mut c = cockpit_n(2);
        assert_eq!(c.focus, Focus::Nodes);
        c.on_key(KeyCode::Tab);
        assert_eq!(c.focus, Focus::Models);
        c.on_key(KeyCode::Tab);
        assert_eq!(c.focus, Focus::Nodes);
        c.on_key(KeyCode::Right);
        assert_eq!(c.focus, Focus::Models);
        c.on_key(KeyCode::BackTab);
        assert_eq!(c.focus, Focus::Nodes);
        c.on_key(KeyCode::Left);
        assert_eq!(c.focus, Focus::Nodes, "Left from nodes stays");
        // Enter on the nodes pane hands focus to the models pane
        c.on_key(KeyCode::Enter);
        assert_eq!(c.focus, Focus::Models);
        // the ▸ marker follows the focused pane
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("▸ models") && !txt.contains("▸ nodes"));
        c.on_key(KeyCode::Tab);
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("▸ nodes") && !txt.contains("▸ models"));
    }

    #[test]
    fn cockpit_up_down_moves_within_the_focused_pane() {
        let mut c = cockpit_n(3);
        c.node.models = vec![mi("a.gguf"), mi("b.gguf"), mi("c.gguf")];
        // Models focus: Up/Down move the MODEL cursor, not the node selection.
        c.focus = Focus::Models;
        c.on_key(KeyCode::Down);
        assert_eq!(c.node.sel, 1);
        assert_eq!(c.fleet.sel, 0, "node selection untouched");
        c.on_key(KeyCode::Up);
        assert_eq!(c.node.sel, 0);
        // Nodes focus: Up/Down move the NODE selection and re-point the
        // models pane (stale list cleared, worker fetch dispatched).
        c.focus = Focus::Nodes;
        c.on_key(KeyCode::Down);
        assert_eq!(c.fleet.sel, 1);
        assert_eq!(c.node.node.name, "node-01", "models pane re-pointed");
        assert!(c.node.models.is_empty(), "stale model list cleared");
        assert!(
            matches!(c.fetch, ModelsFetch::Running { .. }),
            "fetch dispatched on a worker"
        );
        // jump keys move the node selection from EITHER focus
        c.focus = Focus::Models;
        c.on_key(KeyCode::Char('3'));
        assert_eq!(c.fleet.sel, 2);
        assert_eq!(c.node.node.name, "node-02");
    }

    #[test]
    fn cockpit_node_moves_refused_while_node_op_runs() {
        // A bench/expose worker is bound to the focused node; moving the
        // selection out from under it is refused with a status explaining why.
        let mut c = cockpit_n(2);
        c.node.bench = BenchState::Running(std::thread::spawn(|| anyhow::bail!("x")));
        c.on_key(KeyCode::Down);
        assert_eq!(c.fleet.sel, 0, "selection pinned while busy");
        assert!(c.node.status.contains("busy"), "{}", c.node.status);
        c.on_key(KeyCode::Char('2'));
        assert_eq!(c.fleet.sel, 0, "jump refused too");
    }

    #[test]
    fn cockpit_fetch_folds_and_discards_stale() {
        // A landed fetch for the CURRENT selection installs models + history.
        let mut c = cockpit_n(2);
        let models = vec![mi("fetched.gguf")];
        c.fetch = ModelsFetch::Running {
            node_idx: 0,
            handle: std::thread::spawn(move || (models, vec![])),
        };
        for _ in 0..200 {
            c.poll_fetch();
            if !matches!(c.fetch, ModelsFetch::Running { .. }) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(c.node.models.len(), 1);
        assert_eq!(c.node.models[0].name, "fetched.gguf");
        assert!(c.node.status.contains("1 model(s)"), "{}", c.node.status);
        // A landed fetch for a node the selection has MOVED OFF is discarded.
        let mut c = cockpit_n(2);
        let models = vec![mi("stale.gguf")];
        c.fetch = ModelsFetch::Running {
            node_idx: 1,
            handle: std::thread::spawn(move || (models, vec![])),
        };
        for _ in 0..200 {
            c.poll_fetch();
            if !matches!(c.fetch, ModelsFetch::Running { .. }) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(c.node.models.is_empty(), "stale result discarded");
    }

    #[test]
    fn cockpit_refetches_when_a_dispatched_load_finishes() {
        // A fleet-dispatched load finishing on the focused node (spinner
        // Some -> None) queues a models refetch so the served marker moves.
        let mut c = cockpit_n(1);
        c.fleet.cards[0].loading = Some("m.gguf".into());
        c.poll_fetch(); // observes the in-flight load
        assert!(matches!(c.fetch, ModelsFetch::Idle));
        c.fleet.cards[0].loading = None; // load worker reported
        c.poll_fetch();
        assert!(
            matches!(c.fetch, ModelsFetch::Running { .. }),
            "refetch dispatched on the load-finished edge"
        );
    }

    #[test]
    fn cockpit_keys_route_migrated_actions() {
        let mut c = cockpit_n(2);
        c.node.models = vec![mi("m.gguf")];
        // events overlay ([E]) opens from anywhere and closes on any key
        c.on_key(KeyCode::Char('E'));
        assert!(matches!(c.fleet.mode, FleetMode::Events));
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        assert!(buf_text(&term).contains("events (0)"));
        c.on_key(KeyCode::Char('x'));
        assert!(matches!(c.fleet.mode, FleetMode::Normal));
        // compact toggle
        c.on_key(KeyCode::Char('z'));
        assert!(c.fleet.compact);
        // filter routes focus to the models pane
        c.on_key(KeyCode::Char('/'));
        assert!(matches!(c.node.mode, Mode::Filter));
        assert_eq!(c.focus, Focus::Models);
        c.on_key(KeyCode::Char('m'));
        assert_eq!(c.node.filter, "m");
        c.on_key(KeyCode::Enter);
        assert!(matches!(c.node.mode, Mode::Normal));
        // Esc clears the filter first, then quits
        assert_eq!(c.on_key(KeyCode::Esc), KeyOutcome::Continue);
        assert!(c.node.filter.is_empty());
        assert_eq!(c.on_key(KeyCode::Esc), KeyOutcome::Quit);
        assert_eq!(c.on_key(KeyCode::Char('q')), KeyOutcome::Quit);
        // read-only node overlays open from the global map
        c.on_key(KeyCode::Char('h'));
        assert!(matches!(c.node.mode, Mode::History));
        c.on_key(KeyCode::Char('x'));
        c.on_key(KeyCode::Char('L'));
        assert!(matches!(c.node.mode, Mode::Leaderboard));
        c.on_key(KeyCode::Char('x'));
        c.on_key(KeyCode::Char('k'));
        assert!(matches!(c.node.mode, Mode::Compare));
        c.on_key(KeyCode::Char('x'));
        c.on_key(KeyCode::Char('K'));
        assert!(matches!(c.node.mode, Mode::Clusters));
        c.on_key(KeyCode::Char('x'));
        c.on_key(KeyCode::Char('?'));
        assert!(matches!(c.node.mode, Mode::Help));
        c.on_key(KeyCode::Char('x'));
        // the flag editor opens on the selected model and renders in the card pane
        c.on_key(KeyCode::Char('e'));
        assert!(matches!(c.node.mode, Mode::EditFlags));
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        assert!(buf_text(&term).contains("edit flags"));
        c.on_key(KeyCode::Esc);
        assert!(matches!(c.node.mode, Mode::Normal));
        // unload confirm
        c.on_key(KeyCode::Char('u'));
        assert!(matches!(c.node.mode, Mode::ConfirmUnload));
        c.on_key(KeyCode::Char('n'));
        assert!(matches!(c.node.mode, Mode::Normal));
    }

    #[test]
    fn cockpit_two_step_confirms_arm_and_cancel() {
        // B (bench-all): first press arms, second returns the run request,
        // any other key cancels. All-SSH nodes so nothing can actually bench.
        let mut c = cockpit_n(2);
        c.fleet.cfg.nodes[0].transport = Transport::Ssh;
        c.fleet.cfg.nodes[0].host = Some("192.0.2.9".into());
        assert_eq!(c.on_key(KeyCode::Char('B')), KeyOutcome::Continue);
        assert!(c.fleet.bench_all_armed);
        assert_eq!(c.on_key(KeyCode::Char('B')), KeyOutcome::BenchAll);
        assert!(!c.fleet.bench_all_armed);
        c.on_key(KeyCode::Char('B'));
        c.on_key(KeyCode::Char('x'));
        assert!(!c.fleet.bench_all_armed, "any other key cancels");
        assert!(c.fleet.status.contains("cancelled"));
        // F (efficiency bench): same two-step; firing dispatches the workers
        c.on_key(KeyCode::Char('F'));
        assert!(c.fleet.eff_armed);
        c.on_key(KeyCode::Char('F'));
        assert!(!c.fleet.eff_armed);
        assert!(matches!(c.fleet.eff, EffBench::Running { .. }));
    }

    #[test]
    fn cockpit_selected_rail_card_expands_with_detail() {
        // The SELECTED node's rail card carries the full per-node detail
        // (host, health, served, clocks, power/temp, eff, trends); the other
        // nodes stay compact. The old separate node-detail pane is gone.
        let mut c = cockpit_n(3);
        c.fleet.sel = 1;
        c.node.node = c.fleet.cfg.nodes[1].clone();
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("192.0.2.11"), "host row: {txt}");
        assert!(txt.contains("llama up"), "health row");
        assert!(txt.contains("served"), "served row");
        assert!(txt.contains("2230 MHz"), "gfx row");
        assert!(txt.contains("38.1 W"), "power row");
        assert!(txt.contains("61°C"), "temp on the power row");
        assert!(txt.contains("1.1 t/s/W"), "eff row");
        // only the selected card is expanded: one gfx row on the whole screen
        assert_eq!(txt.matches("MHz").count(), 1, "one expanded card: {txt}");
        // the detail pane is GONE: the node name appears only on its rail
        // card (the old middle pane repeated it as a panel title)
        assert_eq!(txt.matches("node-01").count(), 1, "no detail pane: {txt}");
        // selecting a different node moves the expansion with it
        c.fleet.sel = 2;
        c.node.node = c.fleet.cfg.nodes[2].clone();
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(
            txt.find("node-01").unwrap() < txt.find("gfx").unwrap(),
            "expanded card moved below node-01: {txt}"
        );
        assert!(txt.contains("192.0.2.12"), "node-02's host shown");
        // [z] forces the selected card compact too: no expanded rows at all
        c.fleet.compact = true;
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(!txt.contains("MHz"), "z compacts the selected card: {txt}");
        assert!(!txt.contains("192.0.2.12"), "no host row in compact mode");
    }

    #[test]
    fn cockpit_model_card_is_the_elastic_pane() {
        // Growing the terminal grows ONLY the model card: the rail and the
        // model list keep their natural widths; the card absorbs all extra
        // width and height.
        let (rail_pref, maxname) = (36u16, 20u16);
        let small = fleet::main_rects(Rect::new(0, 0, 100, 24), rail_pref, maxname);
        let big = fleet::main_rects(Rect::new(0, 0, 160, 50), rail_pref, maxname);
        let (rail_s, list_s, card_s) = (small.0.unwrap(), small.1.unwrap(), small.2);
        let (rail_b, list_b, card_b) = (big.0.unwrap(), big.1.unwrap(), big.2);
        assert_eq!(rail_s.width, rail_b.width, "rail width is natural/fixed");
        assert_eq!(rail_s.width, 36, "rail sits at its content width");
        assert_eq!(list_s.width, list_b.width, "list width is natural/fixed");
        assert_eq!(list_s.width, 28, "list sits at its content width");
        assert_eq!(
            card_b.width - card_s.width,
            60,
            "the card absorbs ALL extra width"
        );
        assert_eq!(card_b.height, 50, "the card spans the full column height");
        assert!(
            card_b.height > card_s.height,
            "extra height goes to the card"
        );
        // narrow terminals: the rail collapses first, then the list; the card
        // rect always exists
        let (rail, list, _card) = fleet::main_rects(Rect::new(0, 0, 60, 20), rail_pref, maxname);
        assert!(rail.is_none(), "rail collapsed on a narrow terminal");
        assert!(list.is_some());
        let (rail, list, card) = fleet::main_rects(Rect::new(0, 0, 30, 10), rail_pref, maxname);
        assert!(rail.is_none() && list.is_none(), "list collapsed too");
        assert_eq!(card.width, 30, "the card takes the whole width");
    }

    #[test]
    fn cockpit_rail_and_list_size_to_content() {
        // A long served-model name widens the rail so the `served` row shows
        // it in FULL, and the models list widens to the longest name - the
        // operator saw both truncated at the old fixed 34-col widths. The
        // card takes whatever width remains.
        let long = "LFM2.5-8B-A1B-UD-IQ4_NL.gguf"; // 28 cols
        let mut c = cockpit_n(1);
        c.fleet.cards[0].status.served = Some(long.into());
        c.node.models = vec![mi(long), mi("tiny.gguf")];
        c.node.models[0].served = true;
        // rail: lead(4) + label(8) + name(28) + chrome(4) = 44 (grew past 34)
        let pref = fleet::rail_natural_width(&c.fleet);
        assert_eq!(pref, 44, "rail width derived from the served row");
        let (rail, list, card) = fleet::main_rects(Rect::new(0, 0, 140, 30), pref, 28);
        assert_eq!(rail.unwrap().width, 44, "rail grew to fit the served row");
        assert_eq!(list.unwrap().width, 36, "list fits the longest name");
        assert_eq!(card.width, 140 - 44 - 36, "the card gets the rest");
        // and on the actual buffer: the full name renders untruncated in the
        // rail (served row), the list, AND the card title - 3 occurrences,
        // zero ellipsized copies
        let mut term = Terminal::new(TestBackend::new(140, 30)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert_eq!(
            txt.matches(long).count(),
            3,
            "full name in rail + list + card: {txt}"
        );
        assert!(
            !txt.contains("LFM2.5-8B-A1B-UD-…"),
            "no ellipsized copy: {txt}"
        );
        // wider terminal: rail/list hold their content width, the card grows
        let (rail_b, list_b, card_b) = fleet::main_rects(Rect::new(0, 0, 190, 30), pref, 28);
        assert_eq!(rail_b.unwrap().width, 44);
        assert_eq!(list_b.unwrap().width, 36);
        assert_eq!(card_b.width, card.width + 50, "extra width -> the card");
        // a pathological name hits the caps instead of eating the screen
        let mut c2 = cockpit_n(1);
        c2.fleet.cards[0].status.served = Some("x".repeat(120));
        assert_eq!(fleet::rail_natural_width(&c2.fleet), 56, "rail capped");
        let (rail_c, list_c, _) = fleet::main_rects(Rect::new(0, 0, 190, 30), 56, 120);
        assert_eq!(rail_c.unwrap().width, 56);
        assert_eq!(list_c.unwrap().width, 56, "list capped");
        // shorter content sizes down toward the RAIL_W floor
        let c3 = cockpit_n(1);
        assert_eq!(
            fleet::rail_natural_width(&c3.fleet),
            38,
            "fixture's 22-col served name -> 4+8+22+4"
        );
    }

    #[test]
    fn fleet_load_result_folds_and_clears_loading() {
        let mut app = fleet_app_n(2);
        app.cards[1].loading = Some("m.gguf".into());
        app.load_tx
            .send((
                1,
                Ok(SwapReport {
                    from: None,
                    to: "m.gguf".into(),
                    ok: true,
                    reverted: false,
                    elapsed_secs: 12.0,
                    used_default: true,
                    detail: "ok".into(),
                }),
            ))
            .unwrap();
        app.poll();
        assert!(app.cards[1].loading.is_none(), "spinner cleared");
        assert!(
            app.status.contains("node-01: loaded `m.gguf`"),
            "{}",
            app.status
        );
        // a failure clears the spinner and headlines the error
        let mut app = fleet_app_n(2);
        app.cards[1].loading = Some("m.gguf".into());
        app.load_tx
            .send((1, Err(anyhow::anyhow!("llama-server did not come back"))))
            .unwrap();
        app.poll();
        assert!(app.cards[1].loading.is_none());
        assert!(app.status.contains("load failed"), "{}", app.status);
        // an out-of-range node index must not panic
        app.load_tx
            .send((99, Err(anyhow::anyhow!("ghost node"))))
            .unwrap();
        app.poll();
    }

    #[test]
    fn footer_follows_focus_and_documents_keys() {
        // Nodes focus: node navigation + compact toggle on the focus line.
        let mut c = cockpit_n(2);
        let mut term = Terminal::new(TestBackend::new(140, 24)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("[1-9,0] jump"));
        assert!(txt.contains("[z] compact"));
        assert!(txt.contains("[Enter] models"));
        // toggled: the hint flips to expand
        c.fleet.compact = true;
        let mut term = Terminal::new(TestBackend::new(140, 24)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        assert!(buf_text(&term).contains("[z] expand"));
        // Models focus: the load keys take over the focus line.
        c.focus = Focus::Models;
        let mut term = Terminal::new(TestBackend::new(140, 24)).unwrap();
        term.draw(|f| draw_cockpit(f, &c.view())).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("[Enter] load"));
        assert!(txt.contains("[A] load all"));
        assert!(txt.contains("[/] filter"));
    }

    // -- Tier 3: cluster grouping --------------------------------------------

    fn cluster(name: &str, head: &str, workers: &[&str]) -> ClusterCfg {
        ClusterCfg {
            name: name.into(),
            head: head.into(),
            workers: workers.iter().map(|s| s.to_string()).collect(),
            rpc_port: 50052,
            rpc_bin: None,
            rpc_bind: None,
        }
    }

    /// 5 nodes with one [[cluster]]: `big` = head node-03 + worker node-01, so
    /// the rail order (3,1,0,2,4) differs from the config order - the
    /// interesting case for grouping/navigation.
    fn fleet_app_clustered() -> FleetApp {
        let mut app = fleet_app_n(5);
        app.cfg.clusters = vec![cluster("big", "node-03", &["node-01"])];
        app
    }

    #[test]
    fn rail_sections_group_and_degrade() {
        // no clusters: one anonymous section, all nodes, no chrome
        let app = fleet_app_n(3);
        let s = fleet::rail_sections(&app.cfg);
        assert_eq!(s.len(), 1);
        assert!(s[0].cluster.is_none());
        assert_eq!(s[0].members, vec![0, 1, 2]);
        // clustered: head first, then workers; the rest under standalone
        let app = fleet_app_clustered();
        let s = fleet::rail_sections(&app.cfg);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].cluster, Some(0));
        assert_eq!(s[0].members, vec![3, 1], "head first, then workers");
        assert!(s[1].cluster.is_none());
        assert_eq!(s[1].members, vec![0, 2, 4]);
        assert_eq!(fleet::rail_order(&app.cfg), vec![3, 1, 0, 2, 4]);
        // members missing from the fleet are skipped, a cluster with no fleet
        // nodes contributes nothing, and a doubly-claimed node stays with the
        // first cluster that named it
        let mut app = fleet_app_n(2);
        app.cfg.clusters = vec![
            cluster("ghost", "nope", &["nada"]),
            cluster("a", "node-00", &["missing", "node-01"]),
            cluster("b", "node-01", &[]),
        ];
        let s = fleet::rail_sections(&app.cfg);
        assert_eq!(s.len(), 1, "ghost renders nothing; b's node already taken");
        assert_eq!(s[0].cluster, Some(1));
        assert_eq!(s[0].members, vec![0, 1]);
        assert_eq!(fleet::rail_order(&app.cfg), vec![0, 1]);
    }

    #[test]
    fn fleet_rail_groups_by_cluster() {
        let app = fleet_app_clustered();
        let mut term = Terminal::new(TestBackend::new(100, 40)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("big"), "cluster header shows the name");
        assert!(txt.contains("cluster"), "labelled as a cluster");
        assert!(txt.contains("2/2 up"), "member up-count: {txt}");
        // aggregate line: summed tok/s + watts over the 2 members
        assert!(txt.contains("82.4 t/s"), "pooled tok/s = 41.2 x 2: {txt}");
        assert!(txt.contains("76.2 W"), "total watts = 38.1 x 2: {txt}");
        assert!(txt.contains("standalone"), "ungrouped nodes get a section");
        // the common cluster-less case stays chrome-free (unchanged look)
        let plain = fleet_app_n(3);
        let mut term = Terminal::new(TestBackend::new(100, 40)).unwrap();
        term.draw(|f| draw_fleet(f, &plain)).unwrap();
        let txt = buf_text(&term);
        assert!(!txt.contains("standalone"));
        assert!(!txt.contains("cluster"));
    }

    #[test]
    fn fleet_rail_groups_render_all_sizes() {
        // clustered rail renders for any rack size / terminal, both modes
        for (w, h) in [(100, 30), (80, 24), (40, 12), (20, 8)] {
            for n in [1usize, 3, 12] {
                for compact in [false, true] {
                    let mut app = fleet_app_n(n);
                    let names: Vec<String> = app.cfg.nodes.iter().map(|x| x.name.clone()).collect();
                    let workers: Vec<&str> =
                        names.iter().skip(1).take(2).map(|s| s.as_str()).collect();
                    app.cfg.clusters = vec![cluster("big", &names[0], &workers)];
                    app.compact = compact;
                    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                    term.draw(|f| draw_fleet(f, &app)).unwrap();
                }
            }
        }
        // compact grouping keeps the header (one combined line)
        let mut app = fleet_app_clustered();
        app.compact = true;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("big"));
        assert!(txt.contains("2/2 up"));
    }

    #[test]
    fn fleet_grouped_rail_scrolls_to_selection() {
        // 12 nodes all in one cluster on a short terminal: selecting the last
        // node must scroll it into view past the header lines.
        let mut app = fleet_app_n(12);
        let workers: Vec<String> = (1..12).map(|i| format!("node-{i:02}")).collect();
        app.cfg.clusters = vec![ClusterCfg {
            name: "big".into(),
            head: "node-00".into(),
            workers,
            rpc_port: 50052,
            rpc_bin: None,
            rpc_bind: None,
        }];
        app.sel = 11;
        let mut term = Terminal::new(TestBackend::new(90, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(buf_text(&term).contains("node-11"));
        // and selecting the first shows the header + first card
        app.sel = 0;
        let mut term = Terminal::new(TestBackend::new(90, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("big"));
        assert!(txt.contains("node-00"));
        assert!(!txt.contains("node-11"));
    }

    #[test]
    fn fleet_nav_follows_rail_order() {
        let mut app = fleet_app_clustered(); // rail order 3,1,0,2,4
        app.sel = 3;
        app.move_sel(1);
        assert_eq!(app.sel, 1, "down from the head goes to its worker");
        app.move_sel(1);
        assert_eq!(app.sel, 0, "then into the standalone section");
        app.move_sel(-1);
        assert_eq!(app.sel, 1);
        app.move_sel(-1);
        assert_eq!(app.sel, 3);
        app.move_sel(-1);
        assert_eq!(app.sel, 3, "clamped at the top of the rail");
        // jump keys are rail positions, not config indices
        app.jump('1');
        assert_eq!(app.sel, 3);
        app.jump('2');
        assert_eq!(app.sel, 1);
        app.jump('5');
        assert_eq!(app.sel, 4);
        app.jump('9');
        assert_eq!(app.sel, 4, "out-of-range jump is a no-op");
    }

    #[test]
    fn fleet_cluster_agg_excludes_down_member() {
        let mut app = fleet_app_clustered();
        app.cards[3].status.last_gen_tok_s = Some(20.0);
        app.cards[1].status.last_gen_tok_s = Some(13.3);
        let mut term = Terminal::new(TestBackend::new(100, 40)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(
            buf_text(&term).contains("33.3 t/s"),
            "pooled tok/s = 20.0 + 13.3"
        );
        // the worker dies: the aggregate drops its numbers, the up-count shows it
        app.cards[1].status = NodeStatus::unreachable("node-01");
        app.cards[1].telem = None;
        let mut term = Terminal::new(TestBackend::new(100, 40)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("1/2 up"), "down worker counted: {txt}");
        assert!(
            !txt.contains("33.3 t/s"),
            "a dead member must leave the pool sum"
        );
    }

    #[test]
    fn fleet_expanded_card_shows_cluster_membership() {
        let mut app = fleet_app_clustered();
        app.sel = 3;
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(buf_text(&term).contains("(head)"));
        app.sel = 1;
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(buf_text(&term).contains("(worker)"));
        app.sel = 0;
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(!txt.contains("(head)") && !txt.contains("(worker)"));
    }

    // -- Tier 3: configurable thresholds --------------------------------------

    #[test]
    fn configured_thresholds_drive_banner_and_colors() {
        use crate::settings::AlertsCfg;
        // temp_color follows the configured bands; defaults preserve 70/80
        let d = AlertsCfg::default();
        assert_eq!(fleet::temp_color(69.9, &d), widgets::GOOD);
        assert_eq!(fleet::temp_color(70.0, &d), widgets::WARN);
        assert_eq!(fleet::temp_color(80.0, &d), widgets::BAD);
        let c = AlertsCfg {
            temp_warn: 55.0,
            temp_crit: 65.0,
            ..Default::default()
        };
        assert_eq!(fleet::temp_color(50.0, &c), widgets::GOOD);
        assert_eq!(fleet::temp_color(60.0, &c), widgets::WARN);
        assert_eq!(fleet::temp_color(65.0, &c), widgets::BAD);

        // banner: a 75C node is quiet at the defaults (behavior unchanged)...
        let mut app = fleet_app_n(1);
        app.cards[0].telem.as_mut().unwrap().temp_c = 75.0;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(!buf_text(&term).contains("ALERT"));
        // ...and alerts once temp_crit is configured at/below it
        app.alerts_cfg.temp_crit = 75.0;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(
            txt.contains("ALERT:") && txt.contains("node-00 75°C"),
            "{txt}"
        );
    }

    #[test]
    fn min_tok_s_floor_alerts_when_configured() {
        let mut app = fleet_app_n(1); // fixture's last bench: 41.2 t/s
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(!buf_text(&term).contains("slow"), "no floor by default");
        app.alerts_cfg.min_tok_s = Some(50.0);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("node-00 slow 41.2 t/s"), "{txt}");
    }

    #[test]
    fn unreachable_debounce_gates_the_alert() {
        let mut app = fleet_app_n(1);
        app.cards[0].status = NodeStatus::unreachable("node-00");
        app.cards[0].unreachable_since = Some(Instant::now());
        // immediate at the default config (behavior unchanged)
        assert!(fleet::alert_flags(&app.cards[0], &app.alerts_cfg).unreachable);
        // debounced: a fresh outage does not alert yet
        app.alerts_cfg.unreachable_after_secs = Some(3600);
        assert!(!fleet::alert_flags(&app.cards[0], &app.alerts_cfg).unreachable);
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        assert!(!buf_text(&term).contains("ALERT"));
        // an outage with no recorded start (defensive) stays quiet too
        app.cards[0].unreachable_since = None;
        assert!(!fleet::alert_flags(&app.cards[0], &app.alerts_cfg).unreachable);
    }

    #[test]
    fn bench_pause_never_raises_a_down_alert() {
        let mut app = fleet_app_n(1); // fixture: served + healthy
        app.cards[0].status.healthy = false;
        // served + unhealthy + the bench-pause marker: intentional, no alert
        app.cards[0].status.benchmarking = true;
        assert!(!fleet::alert_flags(&app.cards[0], &app.alerts_cfg).down);
        // the real bench-window shape (server down => /props gone => no served)
        app.cards[0].status.served = None;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(!txt.contains("ALERT"), "{txt}");
        // The rail ellipsizes at its natural width - assert the visible stems.
        assert!(txt.contains("benchmarking"), "{txt}");
        assert!(txt.contains("(paused for bench"), "{txt}");
        assert!(!txt.contains("llama down"), "{txt}");
        // the same card without the marker is a real down
        app.cards[0].status.served = Some("m.gguf".into());
        app.cards[0].status.benchmarking = false;
        assert!(fleet::alert_flags(&app.cards[0], &app.alerts_cfg).down);
    }

    // -- Tier 3: event log -----------------------------------------------------

    #[test]
    fn event_log_is_bounded_fifo() {
        let mut log = EventLog::new(50);
        for i in 0..60u64 {
            log.push(FleetEvent {
                ts: i,
                node: "n".into(),
                msg: format!("e{i}"),
                sev: EventSev::Info,
            });
        }
        assert_eq!(log.len(), 50, "bounded at capacity");
        let tail: Vec<u64> = log.tail(50).map(|e| e.ts).collect();
        assert_eq!(tail.first(), Some(&10), "oldest dropped first");
        assert_eq!(tail.last(), Some(&59));
        // tail(n) keeps the LAST n, oldest first
        let t3: Vec<u64> = log.tail(3).map(|e| e.ts).collect();
        assert_eq!(t3, vec![57, 58, 59]);
    }

    #[test]
    fn fold_alert_events_is_edge_triggered() {
        let mut log = EventLog::new(10);
        let card = NodeCard::new("n");
        let clear = AlertFlags::default();
        let hot = AlertFlags {
            hot: true,
            ..Default::default()
        };
        // no change -> nothing
        fold_alert_events(&mut log, 1, "n", clear, clear, &card);
        assert_eq!(log.len(), 0);
        // a persisting condition (level, not edge) -> nothing
        fold_alert_events(&mut log, 2, "n", hot, hot, &card);
        assert_eq!(log.len(), 0);
        // two conditions rising together -> one event each
        let both = AlertFlags {
            hot: true,
            down: true,
            ..Default::default()
        };
        fold_alert_events(&mut log, 3, "n", clear, both, &card);
        assert_eq!(log.len(), 2);
        // partial clear (hot persists) -> no recovery yet
        fold_alert_events(&mut log, 4, "n", both, hot, &card);
        assert_eq!(log.len(), 2);
        // full clear -> exactly one recovery
        fold_alert_events(&mut log, 5, "n", hot, clear, &card);
        assert_eq!(log.len(), 3);
        assert_eq!(log.tail(1).next().unwrap().msg, "recovered");
    }

    #[test]
    fn poll_fold_edge_triggers_events() {
        let mut app = fleet_app_n(1);
        let up = NodeStatus {
            name: "node-00".into(),
            reachable: true,
            healthy: true,
            benchmarking: false,
            served: Some("m.gguf".into()),
            models: 1,
            last_model: None,
            last_gen_tok_s: Some(41.2),
        };
        let t = |c: f64| Telemetry {
            gfxclk_mhz: 2230,
            uclk_mhz: 450,
            temp_c: c,
            power_w: Some(38.1),
            ..Default::default()
        };
        // healthy poll: no transition, no event
        app.poll_tx.send((0, up.clone(), Some(t(61.0)))).unwrap();
        app.poll();
        assert_eq!(app.events.len(), 0);
        // crossing temp_crit logs ONCE...
        app.poll_tx.send((0, up.clone(), Some(t(85.0)))).unwrap();
        app.poll();
        assert_eq!(app.events.len(), 1);
        // ...and a persisting condition never repeats (edge, not level)
        app.poll_tx.send((0, up.clone(), Some(t(86.0)))).unwrap();
        app.poll();
        assert_eq!(app.events.len(), 1);
        let ev: Vec<String> = app.events.tail(10).map(|e| e.msg.clone()).collect();
        assert_eq!(ev, vec!["85°C (crit)"]);
        // cooling back down logs a recovery
        app.poll_tx.send((0, up.clone(), Some(t(61.0)))).unwrap();
        app.poll();
        let ev: Vec<String> = app.events.tail(10).map(|e| e.msg.clone()).collect();
        assert_eq!(ev, vec!["85°C (crit)", "recovered"]);
        // going unreachable logs (immediately at the default config)
        app.poll_tx
            .send((0, NodeStatus::unreachable("node-00"), None))
            .unwrap();
        app.poll();
        let ev: Vec<String> = app.events.tail(10).map(|e| e.msg.clone()).collect();
        assert_eq!(ev.last().unwrap(), "unreachable");
        // coming back with llama dead while a model is loaded: recovery of the
        // outage + the new degraded condition, in one fold
        let mut down = up.clone();
        down.healthy = false;
        app.poll_tx.send((0, down, Some(t(61.0)))).unwrap();
        app.poll();
        let ev: Vec<String> = app.events.tail(10).map(|e| e.msg.clone()).collect();
        assert_eq!(ev.last().unwrap(), "llama down (model loaded)");
    }

    #[test]
    fn events_overlay_renders_empty_and_populated() {
        // empty: the hint, no panic
        let mut app = fleet_app_n(2);
        app.mode = FleetMode::Events;
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("events (0)"), "{txt}");
        assert!(txt.contains("no events yet"));
        // populated: `HH:MM  node  msg` rows (ts 1_782_909_240 = 12:34 UTC)
        app.events.push(FleetEvent {
            ts: 1_782_909_240,
            node: "node-07".into(),
            msg: "81°C (crit)".into(),
            sev: EventSev::Crit,
        });
        app.events.push(FleetEvent {
            ts: 1_782_909_300,
            node: "node-07".into(),
            msg: "recovered".into(),
            sev: EventSev::Info,
        });
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("events (2)"));
        assert!(txt.contains("12:34"), "HH:MM timestamp: {txt}");
        assert!(txt.contains("node-07"));
        assert!(txt.contains("81°C (crit)"));
        assert!(txt.contains("recovered"));
        // survives a tiny terminal
        let mut term = Terminal::new(TestBackend::new(20, 8)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
    }

    // -- Tier 3: efficiency bench ----------------------------------------------

    fn eff_r(tok: f64, w: f64) -> Option<EffResult> {
        Some(EffResult {
            tok_s: Some(tok),
            avg_w: Some(w),
            err: None,
        })
    }

    fn bench_rec(tok: f64, avg_w: Option<f64>) -> Record {
        Record {
            ts: 1_700_000_000,
            node: "n".into(),
            model: "m.gguf".into(),
            arch: "llama".into(),
            quant: None,
            ctx: 32768,
            profile: "_default".into(),
            perf: crate::bench::PerfBench {
                gen_tok_s: tok,
                n_gen: 128,
                telemetry: crate::telemetry::TelemetrySummary {
                    power_avg_w: avg_w,
                    ..Default::default()
                },
                ..Default::default()
            },
            notes: String::new(),
            build: None,
            cluster: None,
            cluster_members: vec![],
        }
    }

    #[test]
    fn eff_median_and_eff_of_guard_operands() {
        assert_eq!(fleet::eff_median(&[]), None);
        assert_eq!(
            fleet::eff_median(&[Some(EffResult {
                tok_s: Some(40.0),
                avg_w: None,
                err: None
            })]),
            None,
            "no power = unmeasured"
        );
        assert_eq!(fleet::eff_median(&[eff_r(40.0, 40.0)]), Some(1.0));
        // even count -> mean of the middle two
        let r = [
            eff_r(40.0, 40.0),
            eff_r(44.0, 40.0),
            eff_r(48.0, 40.0),
            eff_r(20.0, 40.0),
        ];
        assert_eq!(fleet::eff_median(&r), Some((1.0 + 1.1) / 2.0));
        // zero watts never divides
        assert_eq!(
            fleet::eff_of(&EffResult {
                tok_s: Some(40.0),
                avg_w: Some(0.0),
                err: None
            }),
            None
        );
    }

    #[test]
    fn rank_eff_ranks_and_flags_outliers() {
        // effs 1.0 / 1.05 / 1.1 / 0.5 -> median 1.025; 0.5 < 0.85 x median = LOW
        let r = [
            eff_r(40.0, 40.0),
            eff_r(42.0, 40.0),
            eff_r(44.0, 40.0),
            eff_r(20.0, 40.0),
        ];
        let ranked = fleet::rank_eff(&r);
        let order: Vec<usize> = ranked.iter().map(|x| x.node_idx).collect();
        assert_eq!(order, vec![2, 1, 0, 3], "best t/s-per-W first");
        assert!(ranked[3].low, "the weak node is flagged");
        assert!(!ranked[0].low && !ranked[1].low && !ranked[2].low);
        // unmeasured nodes list after measured ones: power-less, then failed
        let r = [
            Some(EffResult {
                tok_s: None,
                avg_w: None,
                err: Some("ssh".into()),
            }),
            eff_r(40.0, 40.0),
            Some(EffResult {
                tok_s: Some(40.0),
                avg_w: None,
                err: None,
            }),
        ];
        let ranked = fleet::rank_eff(&r);
        let order: Vec<usize> = ranked.iter().map(|x| x.node_idx).collect();
        assert_eq!(order, vec![1, 2, 0]);
        // fewer than 3 measured: no outlier calls (a 2-node median flags nothing)
        let r = [eff_r(40.0, 40.0), eff_r(10.0, 40.0)];
        assert!(fleet::rank_eff(&r).iter().all(|x| !x.low));
    }

    #[test]
    fn eff_bench_folds_results_and_opens_overlay() {
        let mut app = fleet_app_n(3);
        app.eff = EffBench::Running {
            results: vec![None, None, None],
        };
        app.eff_tx
            .send((0, Ok(bench_rec(41.2, Some(38.1)))))
            .unwrap();
        app.eff_tx.send((1, Ok(bench_rec(40.0, None)))).unwrap(); // older node: no power on the wire
        app.poll();
        assert!(
            matches!(app.eff, EffBench::Running { .. }),
            "still waiting on node 2"
        );
        assert!(app.status.contains("2/3"), "{}", app.status);
        app.eff_tx
            .send((2, Err(anyhow::anyhow!("ssh timeout"))))
            .unwrap();
        app.poll();
        assert!(matches!(app.eff, EffBench::Done(_)));
        assert!(matches!(app.mode, FleetMode::EffResults));
        // the overlay renders ranked rows, the no-power row, and the failure
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("efficiency"), "{txt}");
        assert!(txt.contains("1.08 t/s/W"), "41.2 / 38.1 ranked: {txt}");
        assert!(txt.contains("no power data"));
        assert!(txt.contains("bench failed: ssh timeout"));
        assert!(txt.contains("rack median"));
        // a stale/duplicate report must not panic or reopen anything
        app.eff_tx.send((0, Err(anyhow::anyhow!("ghost")))).unwrap();
        app.poll();
        assert!(matches!(app.eff, EffBench::Done(_)));
    }

    #[test]
    fn eff_overlay_flags_outlier_and_survives_small_terminals() {
        let mut app = fleet_app_n(4);
        app.eff = EffBench::Done(vec![
            eff_r(40.0, 40.0),
            eff_r(42.0, 40.0),
            eff_r(44.0, 40.0),
            eff_r(20.0, 40.0),
        ]);
        app.mode = FleetMode::EffResults;
        let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("LOW"), "outlier flagged: {txt}");
        assert!(txt.contains("node-03"));
        assert!(txt.contains("51% below median"), "{txt}");
        for (w, h) in [(60, 12), (20, 8)] {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            term.draw(|f| draw_fleet(f, &app)).unwrap();
        }
    }

    #[test]
    fn fleet_footer_documents_tier3_keys() {
        let app = fleet_app_n(2);
        let mut term = Terminal::new(TestBackend::new(160, 24)).unwrap();
        term.draw(|f| draw_fleet(f, &app)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("[F] eff"));
        assert!(txt.contains("[E] events"));
        assert!(txt.contains("[B] bench-all"));
        assert!(txt.contains("[?] help"));
    }

    #[test]
    fn wrap_display_western_and_cjk() {
        // western word wrap
        let w = wrap_display("the quick brown fox", 9);
        assert!(w.iter().all(|l| dwidth(l) <= 9));
        assert!(w.len() >= 2);
        // CJK has no spaces -> must hard-break by display width
        let cjk = wrap_display("回去工作回去工作回去工作", 6); // each char width 2
        assert!(cjk.iter().all(|l| dwidth(l) <= 6), "{cjk:?}");
        // blank lines preserved
        assert_eq!(wrap_display("a\n\nb", 10).len(), 3);
    }

    #[test]
    fn flag_editor_renders() {
        let mut app = node_app();
        app.models = vec![mi("m.gguf")];
        app.flag_edit = Some(FlagEdit {
            model_name: "Q.gguf".into(),
            flags: vec!["-c 32768".into(), "-ngl 99".into()],
            sel: 0,
            buf: None,
        });
        app.mode = Mode::EditFlags;
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw_node(f, &app)).unwrap();
        app.flag_edit.as_mut().unwrap().buf = Some("-c 4096".into());
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw_node(f, &app)).unwrap();
    }

    #[test]
    fn split_flags_groups_values() {
        let f = split_flags("-c 32768 --parallel 1 --no-mmap -ngl 99");
        assert_eq!(f, vec!["-c 32768", "--parallel 1", "--no-mmap", "-ngl 99"]);
        assert!(split_flags("").is_empty());
    }

    #[test]
    fn dwidth_counts_wide_chars() {
        assert_eq!(dwidth("ab"), 2);
        assert_eq!(dwidth("回去"), 4); // two width-2 chars
    }

    fn mi(name: &str) -> ModelInfo {
        ModelInfo {
            name: name.into(),
            arch: "llama".into(),
            params: None,
            quant: None,
            size_gib: 5.0,
            ctx_max: None,
            profile: "_default".into(),
            used_default: true,
            served: false,
            bin: "/x/llama-server".into(),
            flags: "-c 4096 -ngl 99".into(),
            overridden: false,
            mem: Some(crate::mem::estimate(
                5 * (1u64 << 30),
                Some(crate::mem::Dims {
                    n_layers: 28,
                    n_head: 16,
                    n_head_kv: 4,
                    head_dim: 128,
                    ctx_train: Some(40960),
                }),
                4096,
                "q4_0",
                Some(crate::mem::UmaBudget {
                    vram_total: 8u64 << 30,
                    gtt_total: 5u64 << 30,
                    vram_used: 0,
                    gtt_used: 0,
                    sys_total: 16u64 << 30,
                    sys_available: 13u64 << 30,
                }),
            )),
            has_mtp: false,
        }
    }

    fn buf_text(term: &Terminal<TestBackend>) -> String {
        term.backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn filter_narrows_visible_and_maps_selection() {
        let mut a = node_app();
        a.models = vec![mi("Qwen3.5-9B"), mi("gemma-12b"), mi("Qwen3.6-27B")];
        assert_eq!(a.visible().len(), 3);
        a.filter = "qwen".into(); // case-insensitive
        let v = a.visible();
        assert_eq!(v.len(), 2);
        a.sel = 1;
        assert_eq!(a.selected_model().unwrap().name, "Qwen3.6-27B");
        a.filter = "zzz".into();
        assert!(a.visible().is_empty());
        assert!(a.selected_model().is_none());
    }

    #[test]
    fn model_table_scrolls_to_selection() {
        let models: Vec<ModelInfo> = (0..12).map(|i| mi(&format!("zmodel-{i:02}"))).collect();
        let render = |sel: usize| -> String {
            let mut a = node_app();
            a.models = models.clone();
            a.sel = sel;
            // short screen: not all 12 model rows fit at once
            let mut term = Terminal::new(TestBackend::new(100, 14)).unwrap();
            term.draw(|f| draw_node(f, &a)).unwrap();
            buf_text(&term)
        };
        let top = render(0);
        let bottom = render(11);
        assert!(top.contains("zmodel-00"));
        assert!(
            !top.contains("zmodel-11"),
            "last model must be off-screen when the top is selected"
        );
        assert!(
            bottom.contains("zmodel-11"),
            "selecting the last model must scroll it into view"
        );
    }

    #[test]
    fn move_sel_clamps_empty() {
        let mut app = node_app();
        move_sel(&mut app, 1);
        assert_eq!(app.sel, 0);
        move_sel(&mut app, -1);
        assert_eq!(app.sel, 0);
    }

    #[test]
    fn card_renders_memory_fit() {
        let mut a = node_app();
        a.models = vec![mi("Qwen3-30B-A3B-IQ2.gguf")];
        a.sel = 0;
        let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
        term.draw(|f| draw_node(f, &a)).unwrap();
        let txt = buf_text(&term);
        // the memory-fit section (ctx + headroom) is drawn in the card
        assert!(
            txt.contains("memory @4096"),
            "card must show the serving ctx"
        );
        assert!(txt.contains("headroom"), "card must show UMA headroom");
    }

    #[test]
    fn endpoint_overlay_renders_url_and_snippets() {
        let mut a = node_app();
        a.endpoint_view = Some(Endpoint {
            base_url: "http://127.0.0.1:8080".into(),
            openai_base: "http://127.0.0.1:8080/v1".into(),
            healthy: true,
            model: Some("Qwen3-30B".into()),
            api_key: None,
            exposed: false,
            local_base: "http://127.0.0.1:8080/v1".into(),
            lan_base: Some("http://192.0.2.10:8080/v1".into()),
            identity_branded: false,
        });
        a.mode = Mode::Endpoint;
        let mut term = Terminal::new(TestBackend::new(120, 28)).unwrap();
        term.draw(|f| draw_node(f, &a)).unwrap();
        let txt = buf_text(&term);
        assert!(
            txt.contains("http://127.0.0.1:8080/v1"),
            "overlay shows the base"
        );
        assert!(
            txt.contains("chat/completions"),
            "overlay shows the curl snippet"
        );
        assert!(txt.contains("Qwen3-30B"), "overlay shows the served model");
    }

    #[test]
    fn header_uses_the_sibling_theme() {
        // The shared BC-250-tool identity: the app name in the magenta KEY color
        // with a dim subtitle, and the focused panel carrying a `▸` marker.
        let mut a = node_app();
        a.models = vec![mi("m.gguf")];
        let mut term = Terminal::new(TestBackend::new(90, 12)).unwrap();
        term.draw(|f| draw_node(f, &a)).unwrap();
        let txt = buf_text(&term);
        assert!(txt.contains("llmtune"), "header shows the app name");
        assert!(
            txt.contains("BC-250 inference engine"),
            "header carries the dim subtitle"
        );
        assert!(
            txt.contains("▸ models"),
            "focused panel carries the ▸ marker"
        );
    }
}
