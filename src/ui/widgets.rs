// SPDX-License-Identifier: GPL-2.0-only
//! Shared widget/helper library - the memtune/biostune/aputune sibling look
//! (colors, panels, overlays, key hints) plus the width-aware text helpers.

use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, Borders, HighlightSpacing, Padding, Table, TableState};

pub(crate) use crate::fmt::dwidth;

/// Selection highlight style for a table row: strong when its pane is focused,
/// muted when not (so the cursor is still locatable but clearly inactive).
pub(crate) fn highlight(active: bool) -> Style {
    if active {
        Style::default()
            .bg(ACCENT)
            .fg(Color::Black)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().add_modifier(Modifier::BOLD).fg(ACCENT)
    }
}

/// A small bracketed badge, e.g. `[llama]`, for the detail card.
pub(crate) fn badge_span(text: &str, col: Color) -> Span<'static> {
    Span::styled(
        format!("[{text}]"),
        Style::default().fg(col).add_modifier(Modifier::BOLD),
    )
}

/// Render a selectable table whose viewport auto-scrolls to keep `selected`
/// visible. `selected = None` (e.g. empty list) draws no cursor.
pub(crate) fn render_table(
    f: &mut Frame,
    area: Rect,
    table: Table,
    selected: Option<usize>,
    active: bool,
) {
    let table = table
        .row_highlight_style(highlight(active))
        .highlight_symbol(if active { "▸ " } else { "  " })
        .highlight_spacing(HighlightSpacing::Always);
    let mut ts = TableState::default();
    ts.select(selected);
    f.render_stateful_widget(table, area, &mut ts);
}

/// Hard-break a single space-less run (e.g. a CJK sentence) into chunks each at
/// most `width` display columns.
pub(crate) fn hard_break(word: &str, width: usize) -> Vec<String> {
    use unicode_width::UnicodeWidthChar;
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut w = 0usize;
    for ch in word.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw > width && w > 0 {
            out.push(std::mem::take(&mut cur));
            w = 0;
        }
        cur.push(ch);
        w += cw;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Width-aware word wrap: returns lines each at most `width` display columns,
/// breaking on spaces where possible and hard-breaking over-long runs (CJK has
/// no spaces). Pre-wrapping ourselves (vs ratatui's Wrap) keeps wide characters
/// from spilling past the box border.
pub(crate) fn wrap_display(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    for para in text.split('\n') {
        let mut cur = String::new();
        let mut cur_w = 0usize;
        for word in para.split(' ') {
            let ww = dwidth(word);
            if ww > width {
                if cur_w > 0 {
                    lines.push(std::mem::take(&mut cur));
                    cur_w = 0;
                }
                let mut chunks = hard_break(word, width);
                if let Some(last) = chunks.pop() {
                    for c in chunks {
                        lines.push(c);
                    }
                    cur_w = dwidth(&last);
                    cur = last;
                }
                continue;
            }
            if cur_w > 0 && cur_w + 1 + ww > width {
                lines.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            if cur_w > 0 {
                cur.push(' ');
                cur_w += 1;
            }
            cur.push_str(word);
            cur_w += ww;
        }
        lines.push(cur); // preserve blank lines too
    }
    lines
}

/// Split a llama-server flag string into individual flags, each `-flag [value…]`
/// on its own entry (a new flag starts at a `-`-prefixed token).
pub(crate) fn split_flags(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for tok in s.split_whitespace() {
        if tok.starts_with('-') && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(tok);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

pub(crate) const ACCENT: Color = Color::Cyan;

pub(crate) const GOOD: Color = Color::Green;

pub(crate) const WARN: Color = Color::Yellow;

pub(crate) const BAD: Color = Color::Red;

pub(crate) const DIM: Color = Color::DarkGray;

pub(crate) const KEY: Color = Color::Magenta;

/// A static (non-focusable) panel: dim border, accent bold title. The shared
/// look across memtune/biostune/aputune.
pub(crate) fn rounded(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .padding(Padding::horizontal(1))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))
}

/// An interactive panel whose border and title light up (and gain a `▸` marker)
/// when it holds input focus - the aputune `panel()` convention.
pub(crate) fn panel(title: &str, focused: bool) -> Block<'_> {
    let c = if focused { ACCENT } else { DIM };
    let t = if focused {
        format!(" ▸ {title} ")
    } else {
        format!(" {title} ")
    };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(c))
        .padding(Padding::horizontal(1))
        .title(Span::styled(
            t,
            Style::default().fg(c).add_modifier(Modifier::BOLD),
        ))
}

/// A modal overlay panel: accent border + accent title (siblings build modals as
/// a `rounded` block with an explicit accent border).
pub(crate) fn overlay(title: &str) -> Block<'_> {
    rounded(title).border_style(Style::default().fg(ACCENT))
}

pub(crate) fn centered(w: u16, h: u16, area: Rect) -> Rect {
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect {
        x,
        y,
        width: w.min(area.width),
        height: h.min(area.height),
    }
}

pub(crate) fn pulse_line(width: usize, ms: u128) -> Line<'static> {
    let win = 6usize.min(width.max(1));
    let travel = width.saturating_sub(win);
    let span = (2 * travel).max(1);
    let p = ((ms / 50) % span as u128) as usize;
    let pos = if p <= travel { p } else { span - p };
    Line::from(vec![
        Span::styled("\u{2591}".repeat(pos), Style::default().fg(DIM)),
        Span::styled(
            "\u{2588}".repeat(win),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "\u{2591}".repeat(width.saturating_sub(pos + win)),
            Style::default().fg(DIM),
        ),
    ])
}

/// Render samples as a unicode bar string (one char per sample, 8 levels),
/// normalized min..max so a trend is visible even inside a narrow band (a
/// 58->64 C climb must not render flat). Keeps the LAST `width` samples when
/// there are more than fit. Empty input renders empty; a flat series renders
/// mid-height bars. Pure text, so it drops into any Line/Span layout.
pub(crate) fn spark_str(vals: &[f64], width: usize) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if vals.is_empty() || width == 0 {
        return String::new();
    }
    let take = vals.len().min(width);
    let vals = &vals[vals.len() - take..];
    let (mn, mx) = vals
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let span = mx - mn;
    vals.iter()
        .map(|&v| {
            if span <= f64::EPSILON {
                BARS[3]
            } else {
                let i = ((v - mn) / span * 7.0).round() as usize;
                BARS[i.min(7)]
            }
        })
        .collect()
}

/// Braille spinner frame for a given elapsed-ms clock (10 fps). The fleet
/// "loading model" indicator; deterministic in tests (pass a fixed ms).
pub(crate) fn spinner_ch(ms: u128) -> char {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[((ms / 100) % FRAMES.len() as u128) as usize]
}

pub(crate) fn key_line(items: &[(&str, &str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (k, label) in items {
        spans.push(Span::styled(
            format!("[{k}]"),
            Style::default().fg(KEY).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" {label}  "),
            Style::default().fg(DIM),
        ));
    }
    Line::from(spans)
}

/// Compact context length: powers-of-2-k where exact (32768 -> "32k"), else a
/// rounded k, else the raw number.
pub(crate) fn fmt_ctx(n: u64) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{}k", n / 1024)
    } else if n >= 1000 {
        format!("{:.0}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}
