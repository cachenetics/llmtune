// SPDX-License-Identifier: GPL-2.0-only
//! The unified cockpit - one screen for a BC-250 rack. A health/alert banner
//! and (from two nodes up) a rack-rollup strip (total watts, summed tok/s,
//! tok/s per watt, hottest node, reachable count) over three panes: the node
//! telemetry RAIL (left, the selected node's card expanded with its full
//! detail/trends), the focused node's model LIST (middle), and the selected
//! model's CARD (right). The rail and the list size to their CONTENT - wide
//! enough that the served-model row and the longest model name render in
//! full (capped so one pathological name can't eat the screen); the model
//! card is the single ELASTIC pane - all extra terminal width and height
//! goes to it. Focus moves between the nodes and models panes; on narrow
//! terminals the rail collapses first, then the list; the model card is
//! never hidden. Everything here is a fold over the already-polled state -
//! no IO on the draw path.

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use super::widgets::*;
use super::{
    AlertFlags, CockpitView, EffBench, EffResult, EventSev, FleetApp, Focus, Mode, NodeCard, Ring,
};
use crate::config::{ClusterCfg, Config, Node};
use crate::fmt::fit;
use crate::settings::AlertsCfg;

/// Node-rail width FLOOR - the detail/sparkline rows want at least this much
/// even when every name is short. The rail grows past it to fit its content
/// (see `rail_natural_width`), up to RAIL_MAX.
const RAIL_W: u16 = 34;
/// Content-sized rail cap: one pathologically long served-model name widens
/// the rail only this far, then ellipsizes - the card must keep its share.
const RAIL_MAX: u16 = 56;
/// Content-sized model-list cap (the same guard as RAIL_MAX). 56 fits a ~44-char
/// GGUF name + chrome in full; only a genuinely pathological name ellipsizes.
const MODELS_MAX: u16 = 56;
/// The model card never gets fewer columns than this.
const CARD_MIN: u16 = 30;
/// Minimum usable model-list width before it collapses into the card.
const MODELS_MIN: u16 = 18;
/// Minimum usable rail width before it collapses.
const RAIL_MIN: u16 = 20;

pub(super) fn draw_cockpit(f: &mut Frame, app: &CockpitView) {
    // With ONE node the fleet rollup strip is pure duplication - the
    // "aggregate" is exactly the numbers the expanded rail card already
    // shows - so its row is dropped; from two nodes up the totals (rack
    // watts, summed tok/s, hottest node) are real information. The
    // health/alert banner stays in every layout.
    let single = app.fleet.cards.len() < 2;
    let mut constraints = vec![
        Constraint::Length(1), // header
        Constraint::Length(1), // health/alert banner
    ];
    if !single {
        constraints.push(Constraint::Length(1)); // fleet rollup strip
    }
    constraints.push(Constraint::Min(4)); // nodes | models | card
    constraints.push(Constraint::Length(3)); // status + key hints
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(f.area());
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " llmtune ",
                Style::default().fg(KEY).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "BC-250 inference engine  ·  cockpit",
                Style::default().fg(DIM),
            ),
        ])),
        rows[0],
    );

    draw_banner(f, rows[1], app.fleet);
    let mut next = 2;
    if !single {
        draw_rollup(f, rows[next], app.fleet);
        next += 1;
    }
    draw_main(f, rows[next], app);
    draw_footer(f, rows[next + 1], app);

    // Modal overlays last, above every pane.
    let area = f.area();
    match app.node.mode {
        Mode::ConfirmSwap(idx) => super::overlays::draw_confirm(f, area, app.node, idx),
        Mode::ConfirmUnload => super::overlays::draw_confirm_unload(f, area, app.node),
        Mode::ConfirmServer => super::overlays::draw_confirm_server(f, area, app.node),
        Mode::ConfirmExpose => super::overlays::draw_confirm_expose(f, area, app.node),
        Mode::ConfirmSwapAll => super::overlays::draw_confirm_swap_all(f, area, app.node),
        Mode::Help => super::overlays::draw_help(f, area),
        Mode::Endpoint => super::overlays::draw_endpoint(f, area, app.node),
        Mode::History => super::overlays::draw_history(f, area, app.node),
        Mode::Leaderboard => super::overlays::draw_leaderboard(f, area, app.node),
        Mode::Compare => super::overlays::draw_compare(f, area, app.node),
        Mode::Doctor => super::overlays::draw_doctor(f, area, app.node),
        Mode::Profiles => super::overlays::draw_profiles(f, area, app.node),
        Mode::Builds => super::overlays::draw_builds(f, area, app.node),
        Mode::Clusters => super::overlays::draw_clusters(f, area, app.node),
        Mode::Netboot => super::netboot::draw_netboot(f, area, app),
        Mode::Normal | Mode::Filter | Mode::EditFlags => {}
    }
    // The busy modal (bench / load / expose-auth reload) centers on the WHOLE
    // terminal rather than cramping into the card pane.
    if app.node.busy() {
        super::overlays::draw_busy(f, area, app.node);
    }
    draw_events(f, app.fleet);
    draw_eff(f, app.fleet);
}

/// The rail's natural (content-derived) width: wide enough that the widest
/// line of any node's EXPANDED card renders in full - most importantly the
/// `served  <model>` row, plus the node name and the host row - floored at
/// RAIL_W (the detail/sparkline rows want that much anyway) and capped at
/// RAIL_MAX so one pathological name can't eat the card's share. A fold over
/// every polled card (not just the selection), so the rail width is stable
/// as the selection moves. No IO.
pub(super) fn rail_natural_width(app: &FleetApp) -> u16 {
    const LEAD: usize = 4; // marker (2) + health dot (1) + space (1)
    const LABEL: usize = 8; // detail-row label column ("served  ")
    let mut content = 0usize;
    for (card, node) in app.cards.iter().zip(&app.cfg.nodes) {
        content = content.max(LEAD + dwidth(&node.name));
        if let Some(served) = &card.status.served {
            content = content.max(LEAD + LABEL + dwidth(served));
        }
        if let Some(loading) = &card.loading {
            // "<spinner> <model>..." on the loading row
            content = content.max(LEAD + LABEL + 2 + dwidth(loading) + 3);
        }
        let host = node.host.as_deref().unwrap_or("local");
        // host value + the " (transport)" suffix (longest tag = "agent")
        content = content.max(LEAD + LABEL + dwidth(host) + 8);
    }
    // +4 for the panel chrome (borders + horizontal padding) around the
    // rail's inner width.
    ((content + 4) as u16).clamp(RAIL_W, RAIL_MAX)
}

/// The three-pane column split. The rail and the model list keep their
/// NATURAL widths (they never grow with the terminal): `rail_pref` is the
/// rail's content-derived width (`rail_natural_width`), and the list sizes
/// to the longest model name (marker + borders included, capped at
/// MODELS_MAX) so no name is cut off. The model card is the single elastic
/// pane - all extra width (and, columns being full height, all extra height)
/// goes to it. Degrades on narrow terminals: the rail squeezes then
/// collapses first, then the model list - the card rect is always returned.
/// Pure geometry, so the elasticity is directly testable.
pub(super) fn main_rects(
    area: Rect,
    rail_pref: u16,
    maxname: u16,
) -> (Option<Rect>, Option<Rect>, Rect) {
    let w = area.width;
    let show_rail = w >= RAIL_MIN + 22 + CARD_MIN;
    let show_models = w >= MODELS_MIN + CARD_MIN;
    if show_rail {
        let models_w = (maxname + 8)
            .clamp(22, MODELS_MAX)
            .min(w - RAIL_MIN - CARD_MIN);
        let rail_w = rail_pref
            .clamp(RAIL_MIN, RAIL_MAX)
            .min(w - models_w - CARD_MIN)
            .max(RAIL_MIN);
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(rail_w),
                Constraint::Length(models_w),
                Constraint::Min(CARD_MIN),
            ])
            .split(area);
        (Some(cols[0]), Some(cols[1]), cols[2])
    } else if show_models {
        let models_w = (maxname + 8)
            .clamp(MODELS_MIN, MODELS_MAX)
            .min(w - CARD_MIN);
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(models_w), Constraint::Min(CARD_MIN)])
            .split(area);
        (None, Some(cols[0]), cols[1])
    } else {
        (None, None, area)
    }
}

/// The three-pane body: rail | model list | model card over `main_rects`.
fn draw_main(f: &mut Frame, area: Rect, app: &CockpitView) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let maxname = app
        .node
        .models
        .iter()
        .map(|m| dwidth(&m.name) as u16)
        .max()
        .unwrap_or(18);
    let (rail, models, card) = main_rects(area, rail_natural_width(app.fleet), maxname);
    if let Some(r) = rail {
        draw_rail(f, r, app.fleet, app.focus == Focus::Nodes);
    }
    if let Some(m) = models {
        draw_models(f, m, app);
    }
    draw_card_pane(f, card, app);
}

fn draw_models(f: &mut Frame, area: Rect, app: &CockpitView) {
    super::node::draw_models(f, area, app.node, app.focus == Focus::Models);
}

/// The right pane: the selected model's card (or the in-card flag editor),
/// with the busy modal over it while a node op runs on a worker.
fn draw_card_pane(f: &mut Frame, area: Rect, app: &CockpitView) {
    if matches!(app.node.mode, Mode::EditFlags) {
        super::node::draw_flag_editor(f, area, app.node);
    } else {
        super::node::draw_card(f, area, app.node);
    }
    // The busy modal is drawn last in draw_cockpit, centered on the whole frame.
}

/// Status headline + two hint lines: the first follows the focused pane, the
/// second carries the global actions ([?] lists everything).
fn draw_footer(f: &mut Frame, area: Rect, app: &CockpitView) {
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    let status_col = if app.headline.starts_with("[fail]")
        || app.headline.contains("failed")
        || app.headline.contains("did not take")
    {
        BAD
    } else {
        GOOD
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(" {}", app.headline),
            Style::default().fg(status_col).add_modifier(Modifier::BOLD),
        ))),
        parts[0],
    );
    if matches!(app.node.mode, Mode::Filter) {
        f.render_widget(
            Paragraph::new(key_line(&[
                ("type", "filter"),
                ("Enter", "apply"),
                ("Esc", "clear"),
            ])),
            parts[1],
        );
        return;
    }
    let zlabel = if app.fleet.compact {
        "expand"
    } else {
        "compact"
    };
    let focus_line = match app.focus {
        Focus::Nodes => key_line(&[
            ("↑↓", "node"),
            ("1-9,0", "jump"),
            ("Tab/→", "models"),
            ("Enter", "models"),
            ("z", zlabel),
            ("r", "refresh"),
            ("q", "quit"),
        ]),
        Focus::Models => key_line(&[
            ("↑↓", "model"),
            ("Tab/←", "nodes"),
            ("Enter", "load"),
            ("A", "load all"),
            ("/", "filter"),
            ("r", "refresh"),
            ("q", "quit"),
        ]),
    };
    f.render_widget(Paragraph::new(focus_line), parts[1]);
    f.render_widget(
        Paragraph::new(key_line(&[
            ("b", "bench"),
            ("e", "flags"),
            ("o", "endpoint"),
            ("s", "server"),
            ("n", "expose"),
            ("u", "unload"),
            ("E", "events"),
            ("F", "eff"),
            ("B", "bench-all"),
            ("N", "netboot"),
            ("?", "help"),
        ])),
        parts[2],
    );
}

// ---------------------------------------------------------------------------
// Rollup + health banner (pure folds over the already-polled cards; no IO)
// ---------------------------------------------------------------------------

/// Fleet-wide fold over the rail cards. Sums skip unreachable/unprobed nodes so
/// a dead node never inflates (or zeroes) the rack totals.
pub(super) struct Rollup {
    pub total: usize,
    pub reachable: usize,
    /// Summed socket watts over reachable nodes reporting power.
    pub watts: Option<f64>,
    /// Summed last-bench gen tok/s over reachable nodes that have one.
    pub tok_s: Option<f64>,
    /// Hottest reachable node: (card index, temp C).
    pub hottest: Option<(usize, f64)>,
}

pub(super) fn rollup(cards: &[NodeCard]) -> Rollup {
    rollup_over(cards.iter().enumerate())
}

/// The same fold over one cluster's member cards (a subset of the rail), for
/// the per-cluster aggregate line.
pub(super) fn rollup_members(cards: &[NodeCard], members: &[usize]) -> Rollup {
    rollup_over(members.iter().filter_map(|&i| cards.get(i).map(|c| (i, c))))
}

fn rollup_over<'a>(cards: impl Iterator<Item = (usize, &'a NodeCard)>) -> Rollup {
    let mut r = Rollup {
        total: 0,
        reachable: 0,
        watts: None,
        tok_s: None,
        hottest: None,
    };
    for (i, c) in cards {
        r.total += 1;
        if !c.probed || !c.status.reachable {
            continue;
        }
        r.reachable += 1;
        if let Some(w) = c.telem.as_ref().and_then(|t| t.power_w) {
            *r.watts.get_or_insert(0.0) += w;
        }
        if let Some(t) = c.status.last_gen_tok_s {
            *r.tok_s.get_or_insert(0.0) += t;
        }
        if let Some(t) = &c.telem {
            if r.hottest.is_none_or(|(_, h)| t.temp_c > h) {
                r.hottest = Some((i, t.temp_c));
            }
        }
    }
    r
}

/// tok/s per watt: the rack metric that matters on ~40 W boxes. None unless the
/// node has BOTH a bench figure and a nonzero power reading (never divides by 0).
pub(super) fn card_eff(card: &NodeCard) -> Option<f64> {
    let tok = card.status.last_gen_tok_s?;
    let w = card.telem.as_ref().and_then(|t| t.power_w)?;
    (w > 0.0).then_some(tok / w)
}

/// One card's alert state vs the CONFIGURED thresholds (settings.toml
/// `[alerts]`, defaults warn 70 / crit 80). Pure fold over the card; the banner
/// derives its text from this and the poll fold edge-triggers the event log off
/// it, so the two can never disagree. Unprobed nodes are still probing, not
/// alerts; an unreachable node only alerts once the outage is older than the
/// optional `unreachable_after_secs` debounce.
pub(super) fn alert_flags(card: &NodeCard, cfg: &AlertsCfg) -> AlertFlags {
    let mut f = AlertFlags::default();
    if !card.probed {
        return f;
    }
    if !card.status.reachable {
        f.unreachable = match cfg.unreachable_after_secs {
            None => true,
            Some(s) => card
                .unreachable_since
                .is_some_and(|t| t.elapsed().as_secs() >= s),
        };
        return f;
    }
    f.hot = card
        .telem
        .as_ref()
        .is_some_and(|t| t.temp_c >= cfg.temp_crit);
    // A bench-paused server is intentional and self-clearing - never a down
    // alert (served is None during the pause anyway; the guard is explicit).
    f.down = card.status.served.is_some() && !card.status.healthy && !card.status.benchmarking;
    f.slow = cfg
        .min_tok_s
        .is_some_and(|m| card.status.last_gen_tok_s.is_some_and(|t| t < m));
    f
}

/// Per-node alert reasons for the banner: temp >= temp_crit, unreachable (past
/// the debounce), llama-server down while a model is meant to be served, and
/// throughput under the optional floor.
pub(super) fn alerts(app: &FleetApp) -> Vec<String> {
    let mut out = Vec::new();
    for (card, node) in app.cards.iter().zip(&app.cfg.nodes) {
        let f = alert_flags(card, &app.alerts_cfg);
        if f.unreachable {
            out.push(format!("{} unreachable", node.name));
            continue;
        }
        if f.hot {
            if let Some(t) = &card.telem {
                out.push(format!("{} {:.0}°C", node.name, t.temp_c));
            }
        }
        if f.down {
            out.push(format!("{} llama down", node.name));
        }
        if f.slow {
            if let Some(t) = card.status.last_gen_tok_s {
                out.push(format!("{} slow {t:.1} t/s", node.name));
            }
        }
    }
    out
}

/// One prominent row: red + the offending node(s) while any threshold is
/// crossed, quiet green when the whole rack is clear.
fn draw_banner(f: &mut Frame, area: Rect, app: &FleetApp) {
    if area.height == 0 {
        return;
    }
    let alerts = alerts(app);
    let line = if !alerts.is_empty() {
        Line::from(vec![
            Span::styled(
                " ALERT: ".to_string(),
                Style::default().fg(BAD).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                alerts.join(", "),
                Style::default().fg(BAD).add_modifier(Modifier::BOLD),
            ),
        ])
    } else if app.cards.iter().all(|c| c.probed) {
        Line::from(Span::styled(
            " all nodes healthy".to_string(),
            Style::default().fg(GOOD),
        ))
    } else {
        Line::from(Span::styled(
            " probing...".to_string(),
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ))
    };
    f.render_widget(Paragraph::new(line), area);
}

/// The rack totals strip: total watts, summed gen tok/s, fleet tok/s per watt,
/// the hottest node (temp-colored), and the reachable count. Only drawn from
/// two nodes up - with one node it would just repeat the expanded rail card
/// (see draw_cockpit).
fn draw_rollup(f: &mut Frame, area: Rect, app: &FleetApp) {
    if area.height == 0 {
        return;
    }
    let r = rollup(&app.cards);
    let dim = Style::default().fg(DIM);
    // Absent figures are skipped entirely (an all-offline rack rolls up to just
    // "0/N up") - no stale or placeholder numbers on the strip.
    let mut spans = vec![Span::raw(" ")];
    if let Some(w) = r.watts {
        spans.push(Span::styled(
            format!("{w:.1}"),
            Style::default().add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(" W  ".to_string(), dim));
    }
    if let Some(t) = r.tok_s {
        spans.push(Span::styled(
            format!("{t:.1}"),
            Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(" t/s  ".to_string(), dim));
    }
    if let (Some(t), Some(w)) = (r.tok_s, r.watts) {
        if w > 0.0 {
            spans.push(Span::styled(
                format!("{:.1} t/s/W  ", t / w),
                Style::default().fg(ACCENT),
            ));
        }
    }
    if let Some((i, temp)) = r.hottest {
        if let Some(node) = app.cfg.nodes.get(i) {
            spans.push(Span::styled("hot ".to_string(), dim));
            spans.push(Span::styled(
                format!("{} {temp:.0}°C  ", node.name),
                Style::default().fg(temp_color(temp, &app.alerts_cfg)),
            ));
        }
    }
    let up_col = if r.total > 0 && r.reachable == r.total {
        GOOD
    } else {
        WARN
    };
    spans.push(Span::styled(
        format!("{}/{}", r.reachable, r.total),
        Style::default().fg(up_col).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(" up".to_string(), dim));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ---------------------------------------------------------------------------
// Rail (cluster-grouped)
// ---------------------------------------------------------------------------

/// One rail section: a cluster (with its fleet-member node indices, head
/// first) or the ungrouped remainder (`cluster: None`).
pub(super) struct RailSection {
    /// Index into `cfg.clusters`; None = the standalone section.
    pub cluster: Option<usize>,
    /// Node indices into `cfg.nodes`/`cards`, head first for a cluster.
    pub members: Vec<usize>,
}

/// Group the fleet nodes by cluster membership for the rail. No clusters
/// configured (the common single-box case) = one anonymous section, zero
/// chrome. Degrades gracefully: cluster members missing from the fleet are
/// skipped, a node claimed twice stays with the first cluster, and a cluster
/// with no fleet nodes renders nothing.
pub(super) fn rail_sections(cfg: &Config) -> Vec<RailSection> {
    if cfg.clusters.is_empty() {
        return vec![RailSection {
            cluster: None,
            members: (0..cfg.nodes.len()).collect(),
        }];
    }
    let mut used = vec![false; cfg.nodes.len()];
    let mut out = Vec::new();
    for (ci, c) in cfg.clusters.iter().enumerate() {
        let mut members = Vec::new();
        for name in std::iter::once(&c.head).chain(c.workers.iter()) {
            if let Some(i) = cfg.nodes.iter().position(|n| &n.name == name) {
                if !std::mem::replace(&mut used[i], true) {
                    members.push(i);
                }
            }
        }
        if !members.is_empty() {
            out.push(RailSection {
                cluster: Some(ci),
                members,
            });
        }
    }
    let rest: Vec<usize> = (0..cfg.nodes.len()).filter(|&i| !used[i]).collect();
    if !rest.is_empty() {
        out.push(RailSection {
            cluster: None,
            members: rest,
        });
    }
    out
}

/// The rail's display order of node indices (sections flattened) - also the
/// selection order for Up/Down and the 1-9,0 jump keys, so visual neighbors
/// are selection neighbors even when cluster grouping reorders the config.
pub(super) fn rail_order(cfg: &Config) -> Vec<usize> {
    rail_sections(cfg)
        .into_iter()
        .flat_map(|s| s.members)
        .collect()
}

fn draw_rail(f: &mut Frame, area: Rect, app: &FleetApp, focused: bool) {
    let title = format!("nodes ({})", app.cards.len());
    let block = panel(&title, focused);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let w = inner.width as usize;
    let ms = app.started.elapsed().as_millis();
    let sections = rail_sections(&app.cfg);
    let grouped = sections.iter().any(|s| s.cluster.is_some());

    // Build the whole rail (cluster headers + cards) tracking the line where
    // the selected card ends, then scroll so the selection is fully visible.
    let mut lines: Vec<Line> = Vec::new();
    let mut sel_end = 0usize;
    for s in &sections {
        match s.cluster.and_then(|ci| app.cfg.clusters.get(ci)) {
            Some(c) => lines.extend(cluster_header(app, c, &s.members)),
            // The standalone header only exists when clusters are configured -
            // never chrome on a plain (cluster-less) rack.
            None if grouped => lines.push(Line::from(Span::styled(
                "standalone".to_string(),
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            ))),
            None => {}
        }
        for &i in &s.members {
            let (Some(card), Some(node)) = (app.cards.get(i), app.cfg.nodes.get(i)) else {
                continue;
            };
            let selected = i == app.sel;
            // The SELECTED node's card expands with its full detail; the rest
            // stay one line each so a full chassis still fits and scrolls.
            // [z] forces the selected card compact too (the all-scan view).
            if selected && !app.compact {
                lines.extend(expanded_lines(app, i, w, ms));
            } else {
                lines.push(compact_line(card, node, selected, w, ms, &app.alerts_cfg));
            }
            if selected {
                sel_end = lines.len();
            }
        }
    }
    let first = sel_end.saturating_sub(inner.height as usize);
    f.render_widget(Paragraph::new(lines).scroll((first as u16, 0)), inner);
}

/// A cluster's section header: name + members-up count, then an aggregate line
/// (pooled tok/s, total watts, hottest member - the weakest-link tell, since an
/// RPC pool degrades to its slowest/hottest node). One combined row in compact
/// mode so a grouped chassis still scans tight.
fn cluster_header(app: &FleetApp, c: &ClusterCfg, members: &[usize]) -> Vec<Line<'static>> {
    let r = rollup_members(&app.cards, members);
    let dim = Style::default().fg(DIM);
    let up_col = if r.total > 0 && r.reachable == r.total {
        GOOD
    } else {
        WARN
    };
    let mut head = vec![
        Span::styled(
            c.name.clone(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" cluster  ".to_string(), dim),
        Span::styled(
            format!("{}/{} up", r.reachable, r.total),
            Style::default().fg(up_col),
        ),
    ];
    let mut agg: Vec<Span> = Vec::new();
    if let Some(t) = r.tok_s {
        agg.push(Span::styled(
            format!("{t:.1}"),
            Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
        ));
        agg.push(Span::styled(" t/s  ".to_string(), dim));
    }
    if let Some(wt) = r.watts {
        agg.push(Span::styled(
            format!("{wt:.1}"),
            Style::default().add_modifier(Modifier::BOLD),
        ));
        agg.push(Span::styled(" W  ".to_string(), dim));
    }
    if let Some((i, temp)) = r.hottest {
        if let Some(node) = app.cfg.nodes.get(i) {
            agg.push(Span::styled("hot ".to_string(), dim));
            agg.push(Span::styled(
                format!("{} {temp:.0}°C", node.name),
                Style::default().fg(temp_color(temp, &app.alerts_cfg)),
            ));
        }
    }
    if app.compact {
        // one row: name + up-count + whatever aggregates fit
        head.push(Span::raw("  "));
        head.extend(agg);
        vec![Line::from(head)]
    } else {
        let mut agg_line = vec![Span::raw("  ")];
        if agg.is_empty() {
            agg_line.push(Span::styled("-".to_string(), dim));
        } else {
            agg_line.extend(agg);
        }
        vec![Line::from(head), Line::from(agg_line)]
    }
}

/// The health dot for a card: green = serving, cyan = paused for a benchmark
/// (expected, self-clearing - not a fault), yellow = up but llama-server
/// down, red = unreachable, dim = not probed yet.
fn health_dot(card: &NodeCard) -> (&'static str, Color) {
    if !card.probed {
        ("·", DIM)
    } else if !card.status.reachable {
        ("○", BAD)
    } else if card.status.healthy {
        ("●", GOOD)
    } else if card.status.benchmarking {
        ("○", ACCENT)
    } else {
        ("○", WARN)
    }
}

/// Temp color vs the CONFIGURED thresholds (settings.toml `[alerts]`; defaults
/// preserve the historical 70/80 behavior).
pub(super) fn temp_color(c: f64, cfg: &AlertsCfg) -> Color {
    if c >= cfg.temp_crit {
        BAD
    } else if c >= cfg.temp_warn {
        WARN
    } else {
        GOOD
    }
}

/// UMA memory-pressure color (percent of the pool used). The BC-250 shares
/// one 16 GiB pool between system RAM and GPU GTT: a model that doesn't fit
/// OOMs llama-server mid-load with no graceful error, so the escalation here
/// is the early warning the instantaneous clocks can't give.
fn mem_color(pct: f64) -> Color {
    if pct >= 88.0 {
        BAD
    } else if pct >= 70.0 {
        WARN
    } else {
        GOOD
    }
}

/// Socket power color: the BC-250 firmware field saturates at 65.535 W, and
/// sustained >55 W socket draw means the box is working hard in a hot chassis.
fn power_color(w: f64) -> Color {
    if w >= 65.0 {
        BAD
    } else if w >= 55.0 {
        WARN
    } else {
        Color::Reset
    }
}

/// The SELECTED node's expanded rail card: the compact header row plus the
/// full per-node detail that used to live in a separate middle pane - host,
/// cluster role, health, in-flight load, served model, model count, last
/// bench, GPU clocks, power/temp, tok/s per watt, and the trend sparklines
/// over the poll history (the slow climb into thermal-wedge territory that
/// the instantaneous numbers hide). Pure fold over the polled card; `w`
/// (the rail's inner width) only budgets the value/sparkline columns.
fn expanded_lines(app: &FleetApp, idx: usize, w: usize, ms: u128) -> Vec<Line<'static>> {
    let (Some(card), Some(node)) = (app.cards.get(idx), app.cfg.nodes.get(idx)) else {
        return Vec::new();
    };
    let offline = card.probed && !card.status.reachable;
    let th = &app.alerts_cfg;

    // header: marker + health dot + name (the compact row's lead, selected)
    let (dot, dot_col) = health_dot(card);
    let name_style = if offline {
        Style::default().fg(DIM)
    } else {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    };
    let lead = 4usize; // marker (2) + dot (1) + space (1)
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "▸ ".to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(dot.to_string(), Style::default().fg(dot_col)),
        Span::raw(" "),
        Span::styled(fit(&node.name, w.saturating_sub(lead).max(1)), name_style),
    ])];

    // detail rows: 4-col indent + 8-col label + value
    let label = |s: &str| Span::styled(format!("    {s:<8}"), Style::default().fg(DIM));
    let vw = w.saturating_sub(lead + 8).max(4); // value budget

    // Host line only when it adds information: a remote transport, or an explicit
    // host. For the plain local node it would render "local (local)" under the
    // "localhost" name - the word repeated three times, pure noise - so skip it.
    if !matches!(node.transport, crate::config::Transport::Local) || node.host.is_some() {
        let host = node.host.clone().unwrap_or_else(|| "local".into());
        let transport = match node.transport {
            crate::config::Transport::Local => "local",
            crate::config::Transport::Ssh => "ssh",
            crate::config::Transport::Agent => "agent",
        };
        lines.push(Line::from(vec![
            label("host"),
            Span::raw(fit(&host, vw.saturating_sub(transport.len() + 3).max(4))),
            Span::styled(format!(" ({transport})"), Style::default().fg(DIM)),
        ]));
    }

    // RPC-cluster membership (from fleet.toml [[cluster]] blocks), if any.
    let role = app.cfg.clusters.iter().find_map(|c| {
        if c.head == node.name {
            Some((c, "head"))
        } else if c.workers.contains(&node.name) {
            Some((c, "worker"))
        } else {
            None
        }
    });
    if let Some((c, role)) = role {
        lines.push(Line::from(vec![
            label("cluster"),
            Span::styled(c.name.clone(), Style::default().fg(ACCENT)),
            Span::styled(format!(" ({role})"), Style::default().fg(DIM)),
        ]));
    }

    let (htxt, hcol) = if !card.probed {
        ("probing...", DIM)
    } else if !card.status.reachable {
        ("OFFLINE - unreachable", BAD)
    } else if card.status.healthy {
        ("● llama up", GOOD)
    } else if card.status.benchmarking {
        // The bench stopped the server on purpose; calm, not a fault.
        ("○ benchmarking (paused)", ACCENT)
    } else {
        ("○ llama down", WARN)
    };
    lines.push(Line::from(vec![
        label("health"),
        Span::styled(htxt.to_string(), Style::default().fg(hcol)),
    ]));

    // A dashboard-dispatched load in flight on this node's worker.
    if let Some(m) = &card.loading {
        let m = fit(m, vw.saturating_sub(5));
        lines.push(Line::from(vec![
            label("loading"),
            Span::styled(
                format!("{} {}...", spinner_ch(ms), m.trim_end()),
                Style::default().fg(WARN).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    // While a benchmark holds the server, "served (none)" would contradict the
    // health line above - say what is actually happening instead.
    let paused = card.status.benchmarking && !card.status.healthy;
    let served = card.status.served.clone().unwrap_or_else(|| {
        if paused {
            "(paused for benchmark)".into()
        } else {
            "(none)".into()
        }
    });
    let has_served = card.status.served.is_some();
    lines.push(Line::from(vec![
        label("served"),
        Span::styled(
            fit(&served, vw),
            if has_served {
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD)
            } else if paused {
                Style::default().fg(ACCENT)
            } else {
                Style::default().fg(DIM)
            },
        ),
    ]));
    lines.push(Line::from(vec![
        label("models"),
        Span::raw(format!("{}", card.status.models)),
    ]));
    match card.status.last_gen_tok_s {
        Some(t) => lines.push(Line::from(vec![
            label("last"),
            Span::styled(
                format!("{t:.1} t/s"),
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
        ])),
        None => lines.push(Line::from(vec![
            label("last"),
            Span::styled("not benchmarked".to_string(), Style::default().fg(DIM)),
        ])),
    }

    match &card.telem {
        Some(t) => {
            lines.push(Line::from(vec![
                label("gfx"),
                Span::raw(format!("{} MHz", t.gfxclk_mhz)),
                Span::styled("  uclk ".to_string(), Style::default().fg(DIM)),
                Span::raw(format!("{}", t.uclk_mhz)),
            ]));
            // UMA memory: used/total with the pressure percentage. The used
            // figure escalates green -> yellow -> red as the pool fills (a
            // model that doesn't fit OOMs llama-server with no graceful
            // error, so this line is the remote early warning).
            let mut mem_line = vec![label("mem")];
            match (t.mem_used_mib, t.mem_total_mib) {
                (Some(used), Some(total)) if total > 0 => {
                    let pct = used as f64 / total as f64 * 100.0;
                    mem_line.push(Span::styled(
                        format!("{:.1}", used as f64 / 1024.0),
                        Style::default()
                            .fg(mem_color(pct))
                            .add_modifier(Modifier::BOLD),
                    ));
                    mem_line.push(Span::raw(format!(" / {:.1} GiB", total as f64 / 1024.0)));
                    mem_line.push(Span::styled(
                        format!("  ({pct:.0}%)"),
                        Style::default().fg(DIM),
                    ));
                }
                // Older node / non-BC250: the wire omits the fields.
                _ => mem_line.push(Span::styled("n/a".to_string(), Style::default().fg(DIM))),
            }
            lines.push(Line::from(mem_line));
            let power = match t.power_w {
                Some(pw) => Span::styled(
                    format!("{pw:.1} W"),
                    Style::default()
                        .fg(power_color(pw))
                        .add_modifier(Modifier::BOLD),
                ),
                None => Span::styled("n/a".to_string(), Style::default().fg(DIM)),
            };
            lines.push(Line::from(vec![
                label("power"),
                power,
                Span::styled("  temp ".to_string(), Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.0}°C", t.temp_c),
                    Style::default()
                        .fg(temp_color(t.temp_c, th))
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        }
        None => lines.push(Line::from(vec![
            label("gpu"),
            Span::styled(
                if offline {
                    "offline".to_string()
                } else {
                    "no telemetry".to_string()
                },
                Style::default().fg(DIM),
            ),
        ])),
    }
    // gen tok/s per socket watt: the per-node efficiency figure
    lines.push(Line::from(vec![
        label("eff"),
        match card_eff(card) {
            Some(e) => Span::styled(
                format!("{e:.1} t/s/W"),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            None => Span::styled("-".to_string(), Style::default().fg(DIM)),
        },
    ]));

    // Trend sparklines over the poll history (~60 samples at the 5 s cadence,
    // about 5 minutes). Rows with no samples yet are skipped.
    let spark_w = w.saturating_sub(lead + 8 + 8).clamp(4, 24);
    let trends: [(&str, &Ring, Color, usize); 4] = [
        (
            "temp~",
            &card.temp_hist,
            card.temp_hist
                .last()
                .map(|t| temp_color(t, th))
                .unwrap_or(DIM),
            0,
        ),
        // UMA pressure as percent-of-pool: the slow climb toward the OOM
        // ceiling, colored like the mem line above.
        (
            "mem~",
            &card.mem_hist,
            card.mem_hist.last().map(mem_color).unwrap_or(DIM),
            0,
        ),
        ("power~", &card.power_hist, Color::Reset, 0),
        ("tok/s~", &card.tok_hist, GOOD, 1),
    ];
    for (name, ring, col, prec) in trends {
        let Some((mn, mx)) = ring.min_max() else {
            continue;
        };
        let range = if (mx - mn).abs() < f64::EPSILON {
            format!("{mx:.prec$}")
        } else {
            format!("{mn:.prec$}-{mx:.prec$}")
        };
        lines.push(Line::from(vec![
            label(name),
            Span::styled(
                spark_str(&ring.tail(spark_w), spark_w),
                Style::default().fg(col),
            ),
            Span::styled(format!(" {range}"), Style::default().fg(DIM)),
        ]));
    }

    lines
}

/// One compact rail row: `▸ ● name  61°C  41.2 t/s` - health, heat, and
/// throughput at one line per node. The default for every UNSELECTED node
/// (so a full chassis fits and scrolls); [z] forces the selected node's
/// card compact too, the whole-rack scan view.
fn compact_line(
    card: &NodeCard,
    node: &Node,
    selected: bool,
    w: usize,
    ms: u128,
    th: &AlertsCfg,
) -> Line<'static> {
    let offline = card.probed && !card.status.reachable;
    let marker = if selected { "▸ " } else { "  " };
    let (dot, dot_col) = health_dot(card);
    let name_style = if offline {
        Style::default().fg(DIM)
    } else if selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().add_modifier(Modifier::BOLD)
    };

    // Metrics first (fixed-ish width), then the name gets the leftover budget.
    let mut metrics: Vec<Span> = Vec::new();
    let mut mw = 0usize;
    if let Some(loading) = &card.loading {
        let txt = format!("{} loading {}", spinner_ch(ms), fit(loading, 12));
        mw = crate::fmt::dwidth(&txt);
        metrics.push(Span::styled(
            txt,
            Style::default().fg(WARN).add_modifier(Modifier::BOLD),
        ));
    } else if offline {
        mw = 7;
        metrics.push(Span::styled(
            "OFFLINE".to_string(),
            Style::default().fg(BAD).add_modifier(Modifier::DIM),
        ));
    } else if !card.probed {
        mw = 10;
        metrics.push(Span::styled(
            "probing...".to_string(),
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ));
    } else {
        match &card.telem {
            Some(t) => {
                let txt = format!("{:.0}°C", t.temp_c);
                mw += crate::fmt::dwidth(&txt) + 2;
                metrics.push(Span::styled(
                    txt,
                    Style::default().fg(temp_color(t.temp_c, th)),
                ));
                metrics.push(Span::raw("  "));
            }
            None => {
                mw += 6;
                metrics.push(Span::styled("--°C  ".to_string(), Style::default().fg(DIM)));
            }
        }
        match card.status.last_gen_tok_s {
            Some(tok) => {
                let txt = format!("{tok:.1} t/s");
                mw += crate::fmt::dwidth(&txt);
                metrics.push(Span::styled(
                    txt,
                    Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
                ));
            }
            None => {
                mw += 7;
                metrics.push(Span::styled(
                    "-.- t/s".to_string(),
                    Style::default().fg(DIM),
                ));
            }
        }
    }

    let lead = 4usize; // marker (2) + dot (1) + space (1)
    let name_budget = w.saturating_sub(lead + mw + 2).max(4);
    let mut spans = vec![
        Span::styled(
            marker.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(dot.to_string(), Style::default().fg(dot_col)),
        Span::raw(" "),
        Span::styled(fit(&node.name, name_budget), name_style),
        Span::raw("  "),
    ];
    spans.extend(metrics);
    Line::from(spans)
}

// ---------------------------------------------------------------------------
// Event-log overlay ([E])
// ---------------------------------------------------------------------------

/// The [E]vents overlay: the bounded fleet event log (state TRANSITIONS only,
/// edge-triggered from the poll fold), oldest first so it reads chronologically;
/// the newest tail that fits is shown.
fn draw_events(f: &mut Frame, app: &FleetApp) {
    if !matches!(app.mode, super::FleetMode::Events) {
        return;
    }
    let area = f.area();
    let n = app.events.len();
    let h = (n as u16 + 4).clamp(7, 20);
    let inner = super::overlays::open_overlay(f, area, &format!("events ({n})"), 64, h);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let list_h = (inner.height as usize).saturating_sub(2).max(1);

    let mut lines: Vec<Line> = Vec::new();
    if n == 0 {
        lines.push(Line::from(Span::styled(
            "no events yet - node state transitions land here".to_string(),
            Style::default().fg(DIM),
        )));
    }
    for ev in app.events.tail(list_h) {
        let col = match ev.sev {
            EventSev::Crit => BAD,
            EventSev::Warn => WARN,
            EventSev::Info => GOOD,
        };
        // "YYYY-MM-DD HH:MM" -> keep the HH:MM tail for a tight row
        let ts = crate::fmt::fmt_ts(ev.ts);
        let hm = ts.split(' ').nth(1).unwrap_or("--:--").to_string();
        lines.push(Line::from(vec![
            Span::styled(hm, Style::default().fg(DIM)),
            Span::raw("  "),
            Span::styled(
                fit(&ev.node, 12),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(ev.msg.clone(), Style::default().fg(col)),
        ]));
    }
    while lines.len() < list_h + 1 {
        lines.push(Line::from(""));
    }
    lines.push(super::overlays::close_hint());
    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Efficiency bench ([F]): ranking + results overlay
// ---------------------------------------------------------------------------

/// Outlier gate: a node whose tok/s-per-watt is below this fraction of the rack
/// median gets flagged. Needs >= 3 measured nodes (a 2-node "median" would flag
/// whichever box is second).
pub(super) const EFF_OUTLIER_FRAC: f64 = 0.85;

/// One ranked efficiency-bench row (per node).
pub(super) struct EffRank {
    pub node_idx: usize,
    pub tok_s: Option<f64>,
    pub avg_w: Option<f64>,
    /// gen tok/s per average socket watt; None without both operands.
    pub eff: Option<f64>,
    pub err: Option<String>,
    /// Well below the rack median: bad paste / weak silicon / mis-tuned governor.
    pub low: bool,
}

/// tok/s per watt for one result; None unless BOTH operands landed (a node on
/// an older llmtune reports no avg power - it lists, unranked, never divides
/// by zero).
pub(super) fn eff_of(r: &EffResult) -> Option<f64> {
    let t = r.tok_s?;
    let w = r.avg_w?;
    (w > 0.0).then(|| t / w)
}

/// Median tok/s-per-watt over the measured nodes; None when none measured.
pub(super) fn eff_median(results: &[Option<EffResult>]) -> Option<f64> {
    let mut effs: Vec<f64> = results.iter().flatten().filter_map(eff_of).collect();
    if effs.is_empty() {
        return None;
    }
    effs.sort_by(|a, b| a.total_cmp(b));
    let n = effs.len();
    Some(if n % 2 == 1 {
        effs[n / 2]
    } else {
        (effs[n / 2 - 1] + effs[n / 2]) / 2.0
    })
}

/// Rank the efficiency-bench results best-first: measured nodes by tok/s-per-W
/// descending, then power-less nodes, then failed benches. Flags outliers vs
/// the rack median (see EFF_OUTLIER_FRAC).
pub(super) fn rank_eff(results: &[Option<EffResult>]) -> Vec<EffRank> {
    let med = eff_median(results);
    let measured = results
        .iter()
        .flatten()
        .filter(|r| eff_of(r).is_some())
        .count();
    let mut out: Vec<EffRank> = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let r = r.as_ref()?;
            let eff = eff_of(r);
            let low = measured >= 3
                && matches!((eff, med), (Some(e), Some(m)) if m > 0.0 && e < m * EFF_OUTLIER_FRAC);
            Some(EffRank {
                node_idx: i,
                tok_s: r.tok_s,
                avg_w: r.avg_w,
                eff,
                err: r.err.clone(),
                low,
            })
        })
        .collect();
    out.sort_by(|a, b| match (a.eff, b.eff) {
        (Some(x), Some(y)) => y.total_cmp(&x),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        // among the unmeasured: power-less rows before failed ones
        (None, None) => (a.err.is_some()).cmp(&b.err.is_some()),
    });
    out
}

/// The efficiency-bench results overlay: nodes ranked by gen tok/s per socket
/// watt, outliers well below the rack median flagged in red. On a homogeneous
/// rack the flag is a hardware/tuning tell, found automatically.
fn draw_eff(f: &mut Frame, app: &FleetApp) {
    if !matches!(app.mode, super::FleetMode::EffResults) {
        return;
    }
    let EffBench::Done(results) = &app.eff else {
        return;
    };
    let ranked = rank_eff(results);
    let area = f.area();
    let h = (ranked.len() as u16 + 6).clamp(9, 22);
    let inner = super::overlays::open_overlay(f, area, "efficiency - tok/s per watt", 84, h);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let w = inner.width as usize;
    let dim = Style::default().fg(DIM);
    let list_h = (inner.height as usize).saturating_sub(4).max(1);

    let mut lines: Vec<Line> = Vec::new();
    match eff_median(results) {
        Some(med) => lines.push(Line::from(vec![
            Span::styled("rack median ".to_string(), dim),
            Span::styled(
                format!("{med:.2} t/s/W"),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
        ])),
        None => lines.push(Line::from(Span::styled(
            "no node reported both tok/s and power".to_string(),
            dim,
        ))),
    }
    lines.push(Line::from(""));
    for (pos, r) in ranked.iter().enumerate().take(list_h) {
        let name = app
            .cfg
            .nodes
            .get(r.node_idx)
            .map(|n| n.name.clone())
            .unwrap_or_else(|| "?".into());
        if let Some(e) = &r.err {
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(fit(&name, 12), dim),
                Span::raw("  "),
                Span::styled(
                    fit(&format!("bench failed: {e}"), w.saturating_sub(20).max(8)),
                    Style::default().fg(BAD),
                ),
            ]));
            continue;
        }
        let tok = r
            .tok_s
            .map(|t| format!("{t:>6.1}"))
            .unwrap_or_else(|| "   -.-".into());
        let mut spans = vec![
            Span::styled(format!("{:>2}  ", pos + 1), dim),
            Span::styled(
                fit(&name, 12),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(tok, Style::default().fg(GOOD)),
            Span::styled(" t/s  ".to_string(), dim),
        ];
        match (r.avg_w, r.eff) {
            (Some(wt), Some(e)) => {
                spans.push(Span::styled(format!("{wt:>5.1}"), Style::default()));
                spans.push(Span::styled(" W  ".to_string(), dim));
                spans.push(Span::styled(
                    format!("{e:.2} t/s/W"),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ));
            }
            _ => spans.push(Span::styled("no power data".to_string(), dim)),
        }
        if r.low {
            spans.push(Span::styled(
                "  LOW".to_string(),
                Style::default().fg(BAD).add_modifier(Modifier::BOLD),
            ));
            if let (Some(e), Some(m)) = (r.eff, eff_median(results)) {
                if m > 0.0 {
                    spans.push(Span::styled(
                        format!(" ({:.0}% below median)", (1.0 - e / m) * 100.0),
                        Style::default().fg(BAD),
                    ));
                }
            }
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(""));
    lines.push(super::overlays::close_hint());
    f.render_widget(Paragraph::new(lines), inner);
}
