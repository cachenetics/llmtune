// SPDX-License-Identifier: GPL-2.0-only
//! The model panes of the cockpit: the focused node's model LIST, the selected
//! model's CARD, and the in-card flag editor. Layout/orchestration lives in
//! `fleet::draw_cockpit`; these are pure pane draws over `NodeApp` state.

use ratatui::prelude::*;
use ratatui::widgets::{Cell, Paragraph, Row, Table};

use super::widgets::*;
use super::NodeApp;
use crate::mem;

/// Render the in-card flag editor (one flag per line; selected highlighted; the
/// line being edited shows its text + a cursor).
pub(super) fn draw_flag_editor(f: &mut Frame, area: Rect, app: &NodeApp) {
    let Some(fe) = &app.flag_edit else {
        return;
    };
    let title = format!("edit flags · {}", fe.model_name);
    let block = panel(&title, true);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    let mut lines: Vec<Line> = Vec::new();
    if fe.flags.is_empty() {
        lines.push(Line::from(Span::styled(
            "no flags - press a to add one",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )));
    }
    for (i, flag) in fe.flags.iter().enumerate() {
        let selected = i == fe.sel;
        if selected && fe.buf.is_some() {
            let buf = fe.buf.as_deref().unwrap_or("");
            lines.push(Line::from(vec![
                Span::styled(
                    "› ",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    buf.to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled("▏", Style::default().fg(ACCENT)),
            ]));
        } else {
            let (marker, st) = if selected {
                ("▸ ", Style::default().fg(Color::Black).bg(ACCENT))
            } else {
                ("  ", Style::default().fg(Color::Gray))
            };
            lines.push(Line::from(vec![
                Span::styled(marker, Style::default().fg(ACCENT)),
                Span::styled(flag.clone(), st),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), rows[0]);

    let hint = if fe.buf.is_some() {
        key_line(&[("Enter", "ok"), ("Esc", "cancel line")])
    } else {
        key_line(&[
            ("↑↓", "move"),
            ("Enter", "edit"),
            ("a", "add"),
            ("d", "del"),
            ("s", "save"),
            ("x", "reset"),
            ("Esc", "cancel"),
        ])
    };
    f.render_widget(Paragraph::new(hint), rows[1]);
}

pub(super) fn draw_models(f: &mut Frame, area: Rect, app: &NodeApp, focused: bool) {
    let vis = app.visible();
    let rows: Vec<Row> = vis
        .iter()
        .map(|&i| {
            let m = &app.models[i];
            let mark = if m.served { "●" } else { " " };
            let base = if m.served {
                Style::default().fg(GOOD)
            } else {
                Style::default()
            };
            // Just the model name; everything else lives in the card.
            Row::new(vec![Cell::from(format!("{mark} {}", m.name))]).style(base)
        })
        .collect();
    // title shows filtered/total + the active filter
    let title = if app.filter.is_empty() {
        format!("models ({})", app.models.len())
    } else {
        format!(
            "models ({}/{})  /{}",
            vis.len(),
            app.models.len(),
            app.filter
        )
    };
    let table = Table::new(rows, [Constraint::Min(10)])
        .header(
            Row::new(vec!["model"]).style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        )
        .block(panel(&title, focused));
    let sel = (!vis.is_empty()).then_some(app.sel);
    render_table(f, area, table, sel, focused);
}

pub(super) fn draw_card(f: &mut Frame, area: Rect, app: &NodeApp) {
    let block = panel("model", false);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(m) = app.selected_model() else {
        let msg = if app.models.is_empty() {
            "no models found - check models_dir"
        } else {
            "no model matches the filter"
        };
        f.render_widget(
            Paragraph::new(Span::styled(msg, Style::default().fg(DIM))),
            inner,
        );
        return;
    };

    let fw = (inner.width as usize).saturating_sub(2).max(1);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        m.name.clone(),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    if m.served {
        lines.push(Line::from(Span::styled(
            " ● serving ",
            Style::default()
                .bg(GOOD)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )));
    }

    // benchmark numbers for this model, right under the name (b refreshes them).
    match app.history.iter().find(|r| r.model == m.name) {
        Some(r) => lines.push(Line::from(vec![
            Span::styled("prefill ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:.0} t/s", r.perf.prompt_tok_s),
                Style::default().fg(GOOD),
            ),
            Span::styled("   gen ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:.1} t/s", r.perf.gen_tok_s),
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
            Span::styled("   ttft ", Style::default().fg(DIM)),
            Span::raw(format!("{:.0}ms", r.perf.ttft_ms)),
        ])),
        None => lines.push(Line::from(Span::styled(
            "not benchmarked - press b",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ))),
    }
    lines.push(Line::from(""));

    // specs
    lines.push(Line::from(vec![
        badge_span(&m.arch, ACCENT),
        Span::raw(" "),
        badge_span(m.params.as_deref().unwrap_or("?"), DIM),
        Span::raw(" "),
        badge_span(m.quant.as_deref().unwrap_or("?"), DIM),
    ]));
    lines.push(Line::from(vec![
        Span::styled("size ", Style::default().fg(DIM)),
        Span::raw(format!("{:.1} GiB", m.size_gib)),
        Span::styled("   ctx ", Style::default().fg(DIM)),
        Span::raw(
            m.ctx_max
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".into()),
        ),
        Span::styled("   quant ", Style::default().fg(DIM)),
        Span::raw(m.quant.clone().unwrap_or_else(|| "-".into())),
    ]));

    // Memory-fit: weights + KV at the serving ctx/quant vs the UMA budget. The
    // honest "will it run" math - no verdict, headroom colored (green fits / red
    // over), still informational either way.
    if let Some(me) = &m.mem {
        lines.push(Line::from(""));
        if me.dims.is_some() {
            lines.push(Line::from(Span::styled(
                format!("memory @{} {}", me.ctx, me.kv_quant),
                Style::default().fg(DIM),
            )));
            lines.push(Line::from(vec![
                Span::styled("  wt ", Style::default().fg(DIM)),
                Span::raw(format!("{:.1}", mem::gib(me.weights_bytes))),
                Span::styled("  + KV ", Style::default().fg(DIM)),
                Span::raw(format!("{:.1}", mem::gib(me.kv_bytes))),
                Span::styled("  = ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.1} GiB", mem::gib(me.working_bytes)),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("working set ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.1} GiB", mem::gib(me.working_bytes)),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled("  (KV n/a - no header dims)", Style::default().fg(DIM)),
            ]));
        }
        match (&me.budget, me.headroom_bytes) {
            (Some(b), Some(h)) => {
                let (hcol, sign) = if h >= 0 { (GOOD, "+") } else { (BAD, "") };
                // physical UMA (real 16 GiB, not the 16.5 the GPU counters imply),
                // what's usable right now, and the headroom against it.
                lines.push(Line::from(vec![
                    Span::styled("  UMA ", Style::default().fg(DIM)),
                    Span::raw(format!("{:.1}", mem::gib(b.physical_total()))),
                    Span::styled("  free ", Style::default().fg(DIM)),
                    Span::raw(format!("{:.1}", mem::gib(b.usable()))),
                    Span::styled("  headroom ", Style::default().fg(DIM)),
                    Span::styled(
                        format!("{sign}{:.1} GiB", mem::gib_i(h)),
                        Style::default().fg(hcol).add_modifier(Modifier::BOLD),
                    ),
                ]));
                // Context each KV quant buys in the free budget - the KV-size vs
                // context trade-off at a glance. The serving quant is highlighted.
                if let Some(dims) = &me.dims {
                    let kv_budget = b.usable() as i64 - me.weights_bytes as i64;
                    let mut spans = vec![Span::styled("  ctx by KV ", Style::default().fg(DIM))];
                    for q in ["f16", "q8_0", "q4_0"] {
                        let c = mem::max_ctx(dims, kv_budget, q);
                        let serving = q == me.kv_quant;
                        let val = if c == 0 { "-".to_string() } else { fmt_ctx(c) };
                        let mut vs = Style::default().fg(if c == 0 {
                            BAD
                        } else if serving {
                            GOOD
                        } else {
                            Color::Reset
                        });
                        if serving {
                            vs = vs.add_modifier(Modifier::BOLD);
                        }
                        spans.push(Span::styled(format!("{q} "), Style::default().fg(DIM)));
                        spans.push(Span::styled(format!("{val}  "), vs));
                    }
                    lines.push(Line::from(spans));
                }
            }
            _ => lines.push(Line::from(Span::styled(
                "  UMA budget unknown (run on the BC-250)",
                Style::default().fg(DIM),
            ))),
        }
    }

    lines.push(Line::from(""));

    // profile + the llama.cpp launch variables (where you'll apply/adjust/store)
    if m.used_default {
        lines.push(Line::from(vec![
            Span::styled("profile ", Style::default().fg(DIM)),
            Span::styled(
                format!("{} (default - no arch profile)", m.profile),
                Style::default().fg(WARN),
            ),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("profile ", Style::default().fg(DIM)),
            Span::styled(m.profile.clone(), Style::default().fg(ACCENT)),
        ]));
    }
    if !m.flags.is_empty() {
        let mut hdr = vec![Span::styled("llama.cpp flags", Style::default().fg(DIM))];
        if m.overridden {
            hdr.push(Span::styled(
                "  (overridden - x in editor resets)",
                Style::default().fg(WARN),
            ));
        }
        lines.push(Line::from(hdr));
        for flag in split_flags(&m.flags) {
            // one flag per line; wrap only if a single flag is very long
            for seg in wrap_display(&flag, fw) {
                lines.push(Line::from(Span::styled(
                    format!("  {seg}"),
                    Style::default().fg(Color::Gray),
                )));
            }
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}
