// SPDX-License-Identifier: GPL-2.0-only
//! The agentic error/exit-code convention (fleet M7).
//!
//! llmtune is driven by scripts and LLM agents with no TTY, so failures must
//! be machine-parseable and exit codes must distinguish "the tool hit an
//! error" from "a guard refused the action on purpose":
//!
//! - exit 0: success. With `--json`, stdout carries exactly one JSON document
//!   (or one JSON object per line for streaming commands like
//!   `netboot console --json`).
//! - exit 1: error. Something failed (bad arguments, IO, an unreachable
//!   node, a failed build). Nothing may have completed.
//! - exit 2: refusal. A safety guard declined the action and state was left
//!   as it was: the `models rm` served-guard, the `netboot up` off-host
//!   reachability failure, `netboot boot` without a TTY or `--yes`, the
//!   cluster build-version-skew guard. Refusals name their override flag
//!   (`--force`, `--yes`, `--allow-version-skew`, `--skip-check`).
//!
//! With `--json`, an error/refusal prints ONE JSON object to STDERR:
//! `{"error": "<message chain>", "refused": <bool>}` - stdout stays
//! data-only, so `llmtune --json ... | jq` never sees a non-JSON byte.
//! Without `--json` the same failures print `error: <chain>` to stderr;
//! the exit codes are identical in both modes.

use std::fmt;

/// Exit code for plain errors.
pub const EXIT_ERROR: i32 = 1;
/// Exit code for guard refusals (state untouched; an override flag exists).
pub const EXIT_REFUSED: i32 = 2;

/// Marker error for a deliberate guard refusal. Wrap the human message in
/// this (via [`refusal`]) instead of a bare `bail!` so the top-level handler
/// can map it to [`EXIT_REFUSED`] even through `.context(...)` layers.
#[derive(Debug)]
pub struct Refusal(pub String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refusal {}

/// Build a refusal error (exit code [`EXIT_REFUSED`] at the top level).
pub fn refusal(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Refusal(msg.into()))
}

/// Whether an error (anywhere in its context chain) is a [`Refusal`].
pub fn is_refusal(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.downcast_ref::<Refusal>().is_some())
}

/// The exit code for a failed command under the convention above.
pub fn exit_code(e: &anyhow::Error) -> i32 {
    if is_refusal(e) {
        EXIT_REFUSED
    } else {
        EXIT_ERROR
    }
}

/// Ask a y/N question on the tty. Non-interactive callers (no tty on stdin)
/// are REFUSED (exit 2) with `refusal_msg`, which must name the
/// non-interactive override flag.
pub fn confirm_tty(prompt: &str, refusal_msg: &str) -> anyhow::Result<bool> {
    use std::io::{IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return Err(refusal(refusal_msg));
    }
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

/// The one-object stderr payload for a failure under `--json`.
pub fn json_error(e: &anyhow::Error) -> String {
    serde_json::json!({
        "error": format!("{e:#}"),
        "refused": is_refusal(e),
    })
    .to_string()
}

/// Top-level failure rendering: JSON object (stderr) under `--json`, the
/// classic `error:` line otherwise. Returns the exit code to use.
pub fn report_failure(e: &anyhow::Error, json: bool) -> i32 {
    if json {
        eprintln!("{}", json_error(e));
    } else {
        eprintln!("error: {e:#}");
    }
    exit_code(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn refusal_maps_to_exit_2_and_error_to_exit_1() {
        let r = refusal("`m.gguf` is currently SERVED on: bc250-a");
        assert_eq!(exit_code(&r), EXIT_REFUSED);
        let e = anyhow::anyhow!("no model matches `zzz`");
        assert_eq!(exit_code(&e), EXIT_ERROR);
    }

    #[test]
    fn refusal_survives_context_wrapping() {
        // `.context()` layers must not demote a refusal to a plain error.
        let e = Err::<(), _>(refusal("reachability self-check failed"))
            .context("netboot up")
            .unwrap_err();
        assert!(is_refusal(&e));
        assert_eq!(exit_code(&e), EXIT_REFUSED);
    }

    #[test]
    fn json_error_shape_is_stable_and_parses() {
        let e = Err::<(), _>(refusal("refusing to trigger power without a tty"))
            .context("netboot boot")
            .unwrap_err();
        let v: serde_json::Value = serde_json::from_str(&json_error(&e)).unwrap();
        assert_eq!(v["refused"], serde_json::Value::Bool(true));
        let msg = v["error"].as_str().unwrap();
        assert!(msg.contains("netboot boot"), "context chain kept: {msg}");
        assert!(msg.contains("without a tty"));

        let plain = anyhow::anyhow!("ssh `bc250-a` failed: timeout");
        let v: serde_json::Value = serde_json::from_str(&json_error(&plain)).unwrap();
        assert_eq!(v["refused"], serde_json::Value::Bool(false));
        assert!(v["error"].as_str().unwrap().contains("timeout"));
    }
}
