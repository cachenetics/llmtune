// SPDX-License-Identifier: GPL-2.0-only
//! Terminal-width and timestamp format helpers shared by the CLI renderers and
//! the TUI. ONE definition each for `fmt_ts` and `dwidth` (they were previously
//! duplicated in main.rs and ui.rs).

/// Minimal UTC timestamp (YYYY-MM-DD HH:MM) without pulling in a date crate.
pub(crate) fn fmt_ts(unix: u64) -> String {
    // days since epoch -> civil date (Howard Hinnant's algorithm).
    let secs = unix % 86_400;
    let days = (unix / 86_400) as i64;
    let (h, mi) = (secs / 3600, (secs % 3600) / 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}")
}

/// Display width of a string in terminal columns (wide/CJK chars count as 2).
pub(crate) fn dwidth(s: &str) -> usize {
    use unicode_width::UnicodeWidthStr;
    UnicodeWidthStr::width(s)
}

/// Right-pad `s` to exactly `cols` terminal columns (no truncation). Padding is
/// computed by display width, so a wide (CJK) name doesn't shove later columns -
/// the byte/char-counting `{:<N}` can't do this.
pub(crate) fn padw(s: &str, cols: usize) -> String {
    let w = dwidth(s);
    if w >= cols {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(cols - w))
    }
}

/// Fit `s` into exactly `cols` terminal columns for a fixed table column:
/// truncate by display width (char-safe, ellipsis if cut), then pad to `cols`.
pub(crate) fn fit(s: &str, cols: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if dwidth(s) <= cols {
        return padw(s, cols);
    }
    let budget = cols.saturating_sub(1); // leave a column for the ellipsis
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw > budget {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    w += 1;
    if w < cols {
        out.push_str(&" ".repeat(cols - w));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_pad_are_display_width_exact() {
        // ASCII shorter than the column -> padded to exactly N columns.
        assert_eq!(dwidth(&fit("short.gguf", 26)), 26);
        assert_eq!(dwidth(&padw("short.gguf", 40)), 40);
        // A wide (CJK) name must occupy EXACTLY the column width, not overflow it
        // (the bug: char-count padding let 2-col chars shove later columns).
        let name = "abcdefghijklmnopqrstuvwx\u{6a21}\u{578b}long.gguf";
        let f = fit(name, 26);
        assert_eq!(dwidth(&f), 26, "fit must be exactly the column width");
        assert!(f.contains('\u{2026}'));
        // pure-CJK over the limit: char-safe, exact width, no panic.
        let cjk = "\u{6a21}\u{578b}".repeat(10);
        assert_eq!(dwidth(&fit(&cjk, 10)), 10);
        // content already wider than the pad target is returned untouched (no panic).
        assert_eq!(padw("verylongname", 4), "verylongname");
    }

    #[test]
    fn fmt_ts_renders_civil_utc() {
        assert_eq!(fmt_ts(0), "1970-01-01 00:00");
        // 2026-07-01 12:34 UTC
        assert_eq!(fmt_ts(1_782_909_240), "2026-07-01 12:34");
    }
}
