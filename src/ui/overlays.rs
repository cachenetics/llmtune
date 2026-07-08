// SPDX-License-Identifier: GPL-2.0-only
//! Per-mode overlays: read-only stats views, confirm dialogs, busy modal, help.

use ratatui::prelude::*;
use ratatui::widgets::{Cell, Clear, Paragraph, Row, Table};

use super::widgets::*;
use super::{BgState, NodeApp};
use crate::fmt::fmt_ts;
use crate::{compare, doctor, endpoint};

/// Center a read-only overlay, clear it, draw its titled border, and hand back the
/// padded inner rect - the shared scaffold for the stats/config overlays below.
pub(super) fn open_overlay(f: &mut Frame, area: Rect, title: &str, w: u16, h: u16) -> Rect {
    let r = centered(w, h, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay(title), r);
    r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    })
}

/// The dim italic "press any key to close" footer line the read-only overlays share.
pub(super) fn close_hint() -> Line<'static> {
    Line::from(Span::styled(
        "press any key to close",
        Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
    ))
}

/// Read-only overlay: the selected model's benchmark history (newest first).
pub(super) fn draw_history(f: &mut Frame, area: Rect, app: &NodeApp) {
    let name = app
        .selected_model()
        .map(|m| m.name.clone())
        .unwrap_or_default();
    let inner = open_overlay(f, area, &format!("history - {name}"), 84, 22);
    let rows: Vec<Row> = app
        .history
        .iter()
        .filter(|r| r.model == name)
        .take(30)
        .map(|r| {
            Row::new(vec![
                Cell::from(fmt_ts(r.ts)),
                Cell::from(format!("{:.0}", r.perf.prompt_tok_s)),
                Cell::from(format!("{:.1}", r.perf.gen_tok_s)),
                Cell::from(format!("{:.0}", r.perf.ttft_ms)),
                Cell::from(r.ctx.to_string()),
                Cell::from(r.quant.clone().unwrap_or_else(|| "-".into())),
            ])
        })
        .collect();
    let rows_h = inner.height.saturating_sub(2);
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(rows_h), Constraint::Length(1)])
        .split(inner);
    if rows.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "no benchmark history for this model - press b to bench it",
                Style::default().fg(DIM),
            )),
            split[0],
        );
    } else {
        let table = Table::new(
            rows,
            [
                Constraint::Length(17),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Min(8),
            ],
        )
        .header(
            Row::new(vec!["when", "prefill", "gen", "ttft", "ctx", "quant"])
                .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        );
        f.render_widget(table, split[0]);
    }
    f.render_widget(Paragraph::new(close_hint()), split[1]);
}

/// Read-only overlay: best gen tok/s per model across the node's history.
pub(super) fn draw_leaderboard(f: &mut Frame, area: Rect, app: &NodeApp) {
    let inner = open_overlay(f, area, "leaderboard", 74, 22);
    // model -> its best (gen, prefill), keeping only plausible records.
    let mut best: std::collections::BTreeMap<String, (f64, f64)> =
        std::collections::BTreeMap::new();
    for r in &app.history {
        if !r.perf.is_plausible() {
            continue;
        }
        let e = best.entry(r.model.clone()).or_insert((f64::MIN, 0.0));
        if r.perf.gen_tok_s > e.0 {
            *e = (r.perf.gen_tok_s, r.perf.prompt_tok_s);
        }
    }
    let mut ranked: Vec<(String, f64, f64)> =
        best.into_iter().map(|(m, (g, p))| (m, p, g)).collect();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));
    let rows_h = inner.height.saturating_sub(2);
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(rows_h), Constraint::Length(1)])
        .split(inner);
    if ranked.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "no plausible benchmark records yet",
                Style::default().fg(DIM),
            )),
            split[0],
        );
    } else {
        let rows: Vec<Row> = ranked
            .into_iter()
            .map(|(m, pp, gen)| {
                Row::new(vec![
                    Cell::from(m),
                    Cell::from(format!("{pp:.0}")),
                    Cell::from(Span::styled(
                        format!("{gen:.1}"),
                        Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
                    )),
                ])
            })
            .collect();
        let table = Table::new(
            rows,
            [
                Constraint::Min(20),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec!["model", "pp t/s", "gen t/s"])
                .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        );
        f.render_widget(table, split[0]);
    }
    f.render_widget(Paragraph::new(close_hint()), split[1]);
}

/// Read-only overlay: quant families and their throughput trade-off.
pub(super) fn draw_compare(f: &mut Frame, area: Rect, app: &NodeApp) {
    let inner = open_overlay(f, area, "compare - quant trade-off", 74, 24);
    let families = compare::group(&app.history);
    let mut lines: Vec<Line> = Vec::new();
    if families.is_empty() {
        lines.push(Line::from(Span::styled(
            "no plausible benchmark records to compare yet",
            Style::default().fg(DIM),
        )));
    }
    for fam in &families {
        lines.push(Line::from(Span::styled(
            fam.key.clone(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )));
        for v in &fam.variants {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<10}", v.quant),
                    Style::default().fg(Color::Gray),
                ),
                Span::styled(
                    format!("ctx {:<7}", fmt_ctx(v.ctx as u64)),
                    Style::default().fg(DIM),
                ),
                Span::styled(
                    format!("{:.1} gen t/s", v.best_gen_tok_s),
                    Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
                ),
            ]));
        }
        lines.push(Line::from(""));
    }
    lines.push(close_hint());
    f.render_widget(Paragraph::new(lines), inner);
}

/// Read-only overlay: preflight checks, colored by status.
pub(super) fn draw_doctor(f: &mut Frame, area: Rect, app: &NodeApp) {
    let inner = open_overlay(f, area, "doctor", 84, 24);
    let checks = app.doctor_view.as_deref().unwrap_or(&[]);
    let mut lines: Vec<Line> = Vec::new();
    if checks.is_empty() {
        lines.push(Line::from(Span::styled(
            "no checks",
            Style::default().fg(DIM),
        )));
    }
    let fw = (inner.width as usize).saturating_sub(2).max(1);
    for c in checks {
        let (glyph, col) = match c.status {
            doctor::Status::Ok => ("●", GOOD),
            doctor::Status::Warn => ("●", WARN),
            doctor::Status::Fail => ("●", BAD),
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{glyph} "), Style::default().fg(col)),
            Span::styled(
                c.label.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]));
        for seg in wrap_display(&c.detail, fw) {
            lines.push(Line::from(Span::styled(
                format!("  {seg}"),
                Style::default().fg(DIM),
            )));
        }
    }
    lines.push(Line::from(""));
    lines.push(close_hint());
    f.render_widget(Paragraph::new(lines), inner);
}

/// Read-only overlay: configured launch profiles.
pub(super) fn draw_profiles(f: &mut Frame, area: Rect, app: &NodeApp) {
    let inner = open_overlay(f, area, "profiles", 84, 24);
    let profiles = app.profiles_view.as_deref().unwrap_or(&[]);
    let mut lines: Vec<Line> = Vec::new();
    if profiles.is_empty() {
        lines.push(Line::from(Span::styled(
            "no profiles configured",
            Style::default().fg(DIM),
        )));
    }
    let fw = (inner.width as usize).saturating_sub(2).max(1);
    for p in profiles {
        lines.push(Line::from(vec![
            Span::styled(
                p.id.clone(),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("   arch: {}", p.arch_match.join(", ")),
                Style::default().fg(DIM),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  build ", Style::default().fg(DIM)),
            Span::raw(p.build.clone().unwrap_or_else(|| "-".into())),
            Span::styled("  bin ", Style::default().fg(DIM)),
            Span::raw(p.bin.clone()),
        ]));
        if !p.flags.is_empty() {
            lines.push(Line::from(Span::styled(
                "  flags",
                Style::default().fg(DIM),
            )));
            for flag in split_flags(&p.flags) {
                for seg in wrap_display(&flag, fw) {
                    lines.push(Line::from(Span::styled(
                        format!("    {seg}"),
                        Style::default().fg(Color::Gray),
                    )));
                }
            }
        }
        lines.push(Line::from(""));
    }
    lines.push(close_hint());
    f.render_widget(Paragraph::new(lines), inner);
}

/// Read-only overlay: managed llama.cpp builds and their installed versions.
pub(super) fn draw_builds(f: &mut Frame, area: Rect, app: &NodeApp) {
    let inner = open_overlay(f, area, "builds", 78, 22);
    let statuses = app.builds_view.as_deref().unwrap_or(&[]);
    let mut lines: Vec<Line> = Vec::new();
    if statuses.is_empty() {
        lines.push(Line::from(Span::styled(
            "no builds configured (see builds.toml)",
            Style::default().fg(DIM),
        )));
    }
    for s in statuses {
        let tracked = s
            .spec
            .as_ref()
            .map(|sp| format!("  tracks {} @ {}", sp.git_url, sp.git_ref))
            .unwrap_or_else(|| "  (installed, not in builds.toml)".to_string());
        lines.push(Line::from(vec![
            Span::styled(
                s.name.clone(),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(tracked, Style::default().fg(DIM)),
        ]));
        if s.versions.is_empty() {
            lines.push(Line::from(Span::styled(
                "    (not installed)",
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            )));
        }
        for v in &s.versions {
            let (mark, st) = if v.current {
                (
                    "  * ",
                    Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
                )
            } else {
                ("    ", Style::default().fg(Color::Gray))
            };
            lines.push(Line::from(Span::styled(format!("{mark}{}", v.slug), st)));
        }
    }
    lines.push(Line::from(""));
    lines.push(close_hint());
    f.render_widget(Paragraph::new(lines), inner);
}

/// Read-only overlay: configured RPC clusters.
pub(super) fn draw_clusters(f: &mut Frame, area: Rect, app: &NodeApp) {
    let inner = open_overlay(f, area, "clusters", 78, 22);
    let mut lines: Vec<Line> = Vec::new();
    if app.clusters.is_empty() {
        lines.push(Line::from(Span::styled(
            "no clusters configured (see [[cluster]] in fleet.toml)",
            Style::default().fg(DIM),
        )));
    }
    for c in &app.clusters {
        lines.push(Line::from(vec![
            Span::styled(
                c.name.clone(),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("   rpc :{}", c.rpc_port), Style::default().fg(DIM)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  head ", Style::default().fg(DIM)),
            Span::raw(c.head.clone()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  workers ", Style::default().fg(DIM)),
            Span::raw(if c.workers.is_empty() {
                "(none)".to_string()
            } else {
                c.workers.join(", ")
            }),
        ]));
        lines.push(Line::from(""));
    }
    lines.push(close_hint());
    f.render_widget(Paragraph::new(lines), inner);
}

/// Confirm dialog for swap-all: load the selected model on every fleet node.
pub(super) fn draw_confirm_swap_all(f: &mut Frame, area: Rect, app: &NodeApp) {
    let name = app
        .selected_model()
        .map(|m| m.name.clone())
        .unwrap_or_default();
    let n = if app.fleet_nodes.is_empty() {
        1
    } else {
        app.fleet_nodes.len()
    };
    let inner = open_overlay(f, area, "confirm swap-all", 60, 7);
    let lines = vec![
        Line::from(format!("Load on all {n} fleet nodes:")),
        Line::from(Span::styled(
            name,
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "restarts each llama-server sequentially",
            Style::default().fg(DIM),
        )),
        Line::from(vec![
            Span::styled(
                "[y]",
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" load all    "),
            Span::styled("[n]", Style::default().fg(BAD).add_modifier(Modifier::BOLD)),
            Span::raw(" cancel"),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

/// Centered overlay: the OpenAI endpoint + copy-paste connection snippets, so a
/// user can point their own harness at the box without leaving the TUI.
pub(super) fn draw_endpoint(f: &mut Frame, area: Rect, app: &NodeApp) {
    let Some(ep) = &app.endpoint_view else {
        return;
    };
    // Fixed width, height sized to content: the snippet body now carries BOTH a
    // local and a network section, so a 24-row box would clip the LAN commands -
    // exactly what the owner opened this to read. fw derives from the fixed width.
    let width = 78u16;
    let fw = (width as usize).saturating_sub(4);

    let (state, scol) = if ep.healthy {
        ("up", GOOD)
    } else {
        ("down (no model loaded)", WARN)
    };
    // Exposure state: the one line that tells the owner whether the LAN command
    // below actually works yet.
    let expo_line = if ep.exposed {
        Line::from(vec![
            Span::styled("exposure     ", Style::default().fg(DIM)),
            Span::styled("exposed to LAN (0.0.0.0)", Style::default().fg(GOOD)),
        ])
    } else {
        Line::from(vec![
            Span::styled("exposure     ", Style::default().fg(DIM)),
            Span::styled("localhost-only", Style::default().fg(WARN)),
            Span::styled(
                "  - press [e] or `endpoint expose on`",
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            ),
        ])
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled("OpenAI base  ", Style::default().fg(DIM)),
            Span::styled(
                ep.openai_base.clone(),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  [{state}]"), Style::default().fg(scol)),
        ]),
        Line::from(vec![
            Span::styled("served model ", Style::default().fg(DIM)),
            Span::raw(ep.model.clone().unwrap_or_else(|| "(none)".into())),
        ]),
        expo_line,
        Line::from(match &ep.api_key {
            Some(k) => vec![
                Span::styled("auth         ", Style::default().fg(DIM)),
                Span::styled("ON", Style::default().fg(GOOD).add_modifier(Modifier::BOLD)),
                Span::styled(format!("  key required: {k}"), Style::default().fg(GOOD)),
            ],
            None => vec![
                Span::styled("auth         ", Style::default().fg(DIM)),
                Span::styled(
                    "OFF",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  keyless - any client connects", Style::default().fg(DIM)),
            ],
        }),
        // Identity: does the model's chat template override the harness's
        // system prompt (model-branded), or does the harness define identity?
        Line::from(if ep.identity_branded {
            vec![
                Span::styled("identity     ", Style::default().fg(DIM)),
                Span::styled(
                    "model-branded",
                    Style::default().fg(WARN).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "  - the template overrides your harness; press [i]",
                    Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
                ),
            ]
        } else {
            vec![
                Span::styled("identity     ", Style::default().fg(DIM)),
                Span::styled("harness-controlled", Style::default().fg(GOOD)),
                Span::styled(
                    "  - your system prompt defines identity",
                    Style::default().fg(DIM),
                ),
            ]
        }),
        // Action hint up here (not at the bottom) so the snippet body can't push
        // it off the box.
        Line::from(vec![
            Span::styled("[e]", Style::default().fg(KEY).add_modifier(Modifier::BOLD)),
            Span::raw(" toggle expose   "),
            Span::styled(
                "[a]",
                if app.auth_off_armed {
                    Style::default().fg(BAD).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(KEY).add_modifier(Modifier::BOLD)
                },
            ),
            if app.auth_off_armed {
                Span::styled(
                    " press a again to CONFIRM disabling auth   ",
                    Style::default().fg(BAD).add_modifier(Modifier::BOLD),
                )
            } else if ep.api_key.is_some() {
                Span::raw(" auth off   ")
            } else {
                Span::raw(" auth on   ")
            },
            Span::styled("[i]", Style::default().fg(KEY).add_modifier(Modifier::BOLD)),
            Span::raw(" identity   "),
            Span::styled(
                "any other closes",
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            ),
        ]),
        Line::from(""),
    ];
    // The snippet body in a code color; wrap long lines so nothing spills.
    for logical in endpoint::snippets(ep).split('\n') {
        let style = if logical.trim_start().starts_with('#') {
            Style::default().fg(DIM)
        } else {
            Style::default().fg(Color::Cyan)
        };
        for seg in wrap_display(logical, fw) {
            lines.push(Line::from(Span::styled(seg, style)));
        }
    }

    // Size the box to the content (2 for the border), capped to the screen.
    let height = (lines.len() as u16 + 2).min(area.height);
    let r = centered(width, height, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay("endpoint"), r);
    let inner = r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    });
    f.render_widget(Paragraph::new(lines), inner);
}

pub(super) fn draw_busy(f: &mut Frame, area: Rect, app: &NodeApp) {
    let bg = matches!(app.bg, BgState::Running(_));
    let what = if bg { "working" } else { "benchmarking" };
    let elapsed = app.started.elapsed();
    let title = format!("{what}  ·  {:.0}s", elapsed.as_secs_f64());
    // Benchmarking shells out to llama-bench, which blocks (warmup + repeats)
    // and reports only at the end - so we show a "measuring, result at end"
    // hint + the elapsed timer + pulse rather than a per-rep counter we can't
    // actually tick.
    let hint = if bg {
        app.status.as_str()
    } else {
        "llama-bench: warmup + reps - measuring, result at end (~½-1½ min)"
    };
    // The bench result line, once llama-bench has returned (completed == total).
    let bench_done = match &app.bench_progress {
        Some(p) if p.completed >= p.total && p.gen_tok_s > 0.0 => {
            Some(format!("result  {:.1} gen tok/s", p.gen_tok_s))
        }
        _ => None,
    };

    // Size the box to its content so long model names and the staged hint never
    // clip (they exceeded the old fixed width - "…(auto-reverts" got cut off).
    // Width is the widest line of text, clamped to what the area can hold; any
    // remaining overflow wraps rather than truncating.
    let want = dwidth(&app.status)
        .max(dwidth(hint))
        .max(bench_done.as_deref().map(dwidth).unwrap_or(0))
        .max(dwidth(&title) + 2);
    let max_inner = area.width.saturating_sub(6) as usize;
    let inner_w = want.clamp(24, max_inner.max(24));

    // Wrap the variable-length text to the inner width.
    let mut lines: Vec<Line> = wrap_display(&app.status, inner_w)
        .into_iter()
        .map(|l| Line::from(Span::styled(l, Style::default().fg(ACCENT))))
        .collect();
    match &bench_done {
        Some(t) => lines.push(Line::from(Span::styled(
            t.clone(),
            Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
        ))),
        None => lines.extend(
            wrap_display(hint, inner_w)
                .into_iter()
                .map(|l| Line::from(Span::styled(l, Style::default().fg(DIM)))),
        ),
    }
    lines.push(Line::from(""));

    // border (2) + one row per content line + the pulse (1), clamped to area.
    let box_h = (lines.len() as u16 + 3).min(area.height);
    let box_w = (inner_w as u16 + 4).min(area.width);
    let r = centered(box_w, box_h, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay(&title), r);
    let inner = r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    });
    lines.push(pulse_line(inner.width as usize, elapsed.as_millis()));
    f.render_widget(Paragraph::new(lines), inner);
}

pub(super) fn draw_confirm(f: &mut Frame, area: Rect, app: &NodeApp, idx: usize) {
    let name = app
        .visible()
        .get(idx)
        .and_then(|&i| app.models.get(i))
        .map(|m| m.name.clone())
        .unwrap_or_default();
    let r = centered(56, 7, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay("confirm swap"), r);
    let inner = r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    });
    let lines = vec![
        Line::from("Swap the served model to:"),
        Line::from(Span::styled(
            name,
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "restarts llama-server; auto-reverts if it fails to come up",
            Style::default().fg(DIM),
        )),
        Line::from(vec![
            Span::styled(
                "[y]",
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" load    "),
            Span::styled("[n]", Style::default().fg(BAD).add_modifier(Modifier::BOLD)),
            Span::raw(" cancel"),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

pub(super) fn draw_confirm_server(f: &mut Frame, area: Rect, app: &NodeApp) {
    let r = centered(56, 7, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay("confirm server"), r);
    let inner = r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    });
    // Active-state was probed once when the dialog opened (see the `s` handler);
    // rendering must not fork a subprocess on every frame.
    let active = app.server_active;
    let (verb, effect) = if active {
        ("Stop", "frees the GPU for manual testing (no inference)")
    } else {
        ("Start", "resumes inference on the served model")
    };
    let lines = vec![
        Line::from(format!("{verb} the llama-server?")),
        Line::from(Span::styled(
            app.node.llama_unit.clone(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(effect, Style::default().fg(DIM))),
        Line::from(vec![
            Span::styled(
                "[y]",
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" {}    ", verb.to_lowercase())),
            Span::styled(
                "[r]",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" restart    "),
            Span::styled("[n]", Style::default().fg(BAD).add_modifier(Modifier::BOLD)),
            Span::raw(" cancel"),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

pub(super) fn draw_confirm_expose(f: &mut Frame, area: Rect, app: &NodeApp) {
    let r = centered(64, 8, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay("confirm expose"), r);
    let inner = r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    });
    // Probed once when the dialog opened (see probe_expose_state) - no per-frame
    // settings.toml reads inside this render path.
    let (exposed, has_key) = app.expose_state;
    let (question, effect, ecol) = if exposed {
        (
            "Restrict llama-server to localhost?",
            "closes the firewall port, binds 127.0.0.1",
            DIM,
        )
    } else if has_key {
        (
            "Expose llama-server to the LAN?",
            "binds 0.0.0.0 + opens the firewall (api-key required)",
            WARN,
        )
    } else {
        (
            "Expose llama-server to the LAN?",
            "binds 0.0.0.0 + opens the firewall - UNAUTHENTICATED",
            WARN,
        )
    };
    let lines = vec![
        Line::from(question),
        Line::from(Span::styled(
            app.node.llama_unit.clone(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(effect, Style::default().fg(ecol))),
        Line::from(Span::styled(
            "reloads the served model to apply",
            Style::default().fg(DIM),
        )),
        Line::from(vec![
            Span::styled(
                "[y]",
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
            Span::raw(if exposed {
                " restrict    "
            } else {
                " expose    "
            }),
            Span::styled("[n]", Style::default().fg(BAD).add_modifier(Modifier::BOLD)),
            Span::raw(" cancel"),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

pub(super) fn draw_confirm_unload(f: &mut Frame, area: Rect, app: &NodeApp) {
    let r = centered(56, 7, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay("confirm unload"), r);
    let inner = r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    });
    let served = app
        .models
        .iter()
        .find(|m| m.served)
        .map(|m| m.name.clone())
        .unwrap_or_else(|| "the served model".to_string());
    let lines = vec![
        Line::from("Remove llmtune's drop-in for:"),
        Line::from(Span::styled(
            served,
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "reverts the unit to its base config on next restart",
            Style::default().fg(DIM),
        )),
        Line::from(vec![
            Span::styled(
                "[y]",
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" unload    "),
            Span::styled("[n]", Style::default().fg(BAD).add_modifier(Modifier::BOLD)),
            Span::raw(" cancel"),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

pub(super) fn draw_help(f: &mut Frame, area: Rect) {
    let r = centered(64, 32, area);
    f.render_widget(Clear, r);
    f.render_widget(overlay("help"), r);
    let inner = r.inner(Margin {
        horizontal: 2,
        vertical: 1,
    });
    let row = |k: &'static str, d: &'static str| {
        Line::from(vec![
            Span::styled(
                format!("{k:<10}"),
                Style::default().fg(KEY).add_modifier(Modifier::BOLD),
            ),
            Span::styled(d, Style::default()),
        ])
    };
    let lines = vec![
        row("Tab / <->", "move focus: nodes pane <-> models pane"),
        row("Up/Down", "move within the focused pane"),
        row("1-9,0", "jump to rail node 1-10"),
        row("Enter", "nodes: focus models   models: load selection"),
        row("A", "load selected model on ALL fleet nodes"),
        row("u", "unload (remove llmtune's drop-in)"),
        row("s", "server stop/start/restart (free the GPU)"),
        row("n", "expose to the LAN / restrict to localhost"),
        row("b", "benchmark the served model"),
        row("e", "edit the selected model's llama.cpp flags"),
        row("o", "show the OpenAI endpoint + connection snippets"),
        row("/", "filter models  (Esc clears)"),
        row("h", "bench history for the selected model"),
        row("L", "leaderboard (best gen t/s per model)"),
        row("k", "compare quant families / throughput"),
        row("y", "doctor (preflight checks)"),
        row("p", "launch profiles"),
        row("g", "managed llama.cpp builds"),
        row("K", "configured RPC clusters"),
        row("E", "fleet event log"),
        row("F", "efficiency bench all nodes (press twice)"),
        row("B", "bench all nodes (press twice)"),
        row("N", "fleet/netboot dashboard (arm/boot/console/register)"),
        row("z", "compact rail (selected card expands otherwise)"),
        row("r", "refresh nodes + models"),
        row("Esc", "clear filter / quit   (q quits)"),
        Line::from(""),
        Line::from(Span::styled(
            "a load restarts llama-server and auto-reverts on failure",
            Style::default().fg(DIM),
        )),
        Line::from(Span::styled(
            "press any key to close",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}
