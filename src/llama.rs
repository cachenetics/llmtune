// SPDX-License-Identifier: GPL-2.0-only
//! Thin client for a running llama-server: served-model detection, health,
//! props (context window), pre-warm, and the /completion call the bench drives.

use serde_json::Value;
use std::path::Path;
use std::time::Duration;

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(4))
        .build()
}

/// Attach the configured API key (if any) as a Bearer token, so llmtune's own
/// probes keep working when llama-server is launched with `--api-key`. `/health`
/// is public and doesn't need it.
fn auth(req: ureq::Request) -> ureq::Request {
    match crate::settings::api_key() {
        Some(k) => req.set("Authorization", &format!("Bearer {k}")),
        None => req,
    }
}

/// Fetch GET `{url}/props` as JSON, or None if the server is down.
pub fn props(url: &str) -> Option<Value> {
    auth(agent().get(&format!("{url}/props")))
        .call()
        .ok()?
        .into_json()
        .ok()
}

/// Model basename from a `/props` value (`model_path`).
pub fn name_from_props(v: &Value) -> Option<String> {
    let p = v.get("model_path")?.as_str()?;
    if p.is_empty() {
        return None;
    }
    Path::new(p)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
}

/// Served context window (`n_ctx`) from a `/props` value.
pub fn ctx_from_props(v: &Value) -> Option<u32> {
    v.pointer("/default_generation_settings/n_ctx")
        .and_then(|x| x.as_u64())
        .or_else(|| v.get("n_ctx").and_then(|x| x.as_u64()))
        .map(|n| n as u32)
}

/// Basename of the currently-served model, via `/props`.
pub fn served_name(url: &str) -> Option<String> {
    name_from_props(&props(url)?)
}

/// The server's ACTIVE chat template from `/props` (`chat_template`) - the
/// embedded GGUF one, or the `--chat-template-file` override when one is
/// staged. None if the server is down or the field is absent.
pub fn chat_template(url: &str) -> Option<String> {
    props(url)?
        .get("chat_template")?
        .as_str()
        .map(str::to_string)
}

/// Whether the server reports healthy at GET `{url}/health`.
pub fn health_ok(url: &str) -> bool {
    match agent().get(&format!("{url}/health")).call() {
        Ok(resp) => resp
            .into_string()
            .map(|s| s.to_lowercase().contains("ok"))
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// Pre-warm the KV cache with a few short completions after a swap, so the first
/// real request isn't cold. Best-effort: failures are ignored.
pub fn warm(url: &str, token_counts: &[u32]) {
    let ag = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(70))
        .build();
    for &n in token_counts {
        let body = serde_json::json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": n,
        });
        let _ = auth(ag.post(&format!("{url}/v1/chat/completions"))).send_json(body);
    }
}
