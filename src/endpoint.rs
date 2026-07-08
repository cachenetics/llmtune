// SPDX-License-Identifier: GPL-2.0-only
//! Endpoint surface - "point your harness at the box."
//!
//! The product's real output is a fast model at a URL your tools can hit. Most
//! owners connect their own agent/harness rather than chatting in-TUI, so this
//! surfaces the OpenAI-compatible endpoint llama-server already exposes: the base
//! URL, the served model id, health, and copy-paste snippets (curl / OpenAI SDK /
//! a generic agent note). The snippet RENDERER is a pure function (tested); the
//! probe does the live health/served lookup.
//!
//! There are two ways to reach a served model, and owners need BOTH: from the
//! box itself (`127.0.0.1`, always works) and from another machine on the LAN
//! (the host's routable IP - only once the server is exposed with
//! `endpoint expose on`). So the probe carries the local base, the LAN base, and
//! the exposure flag, and the renderer emits a section for each - including the
//! command to expose when the LAN path isn't open yet.

use crate::config::Node;
use crate::llama;
use serde::{Deserialize, Serialize};

/// A node's live OpenAI-compatible endpoint. Also the `node endpoint --json`
/// wire shape the SSH transport parses (one wire contract, no drift).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    /// llama-server base, e.g. `http://127.0.0.1:8080`.
    pub base_url: String,
    /// OpenAI-compatible base, e.g. `http://127.0.0.1:8080/v1`.
    pub openai_base: String,
    pub healthy: bool,
    /// The currently-served model id (None if nothing is loaded / server down).
    #[serde(default)]
    pub model: Option<String>,
    /// The required API key, if the server is launched with one.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Is llama-server bound to all interfaces (reachable from the LAN)?
    #[serde(default)]
    pub exposed: bool,
    /// The always-valid localhost OpenAI base, e.g. `http://127.0.0.1:8080/v1`.
    /// (Older remote peers omit this; the renderer falls back to `openai_base`.)
    #[serde(default)]
    pub local_base: String,
    /// The LAN-routable OpenAI base, e.g. `http://192.0.2.10:8080/v1`, when a
    /// primary IPv4 is discoverable. Present regardless of `exposed` so the
    /// renderer can show the network command (and how to open it).
    #[serde(default)]
    pub lan_base: Option<String>,
    /// Does the served model's chat template embed an identity-branding
    /// instruction that overrides the harness's system prompt? `true` =
    /// model-branded (the template forces its own identity); `false` =
    /// harness-controlled (the harness's system prompt defines identity -
    /// also the safe/neutral read when the server is down).
    #[serde(default)]
    pub identity_branded: bool,
}

/// Probe a node's llama-server for health + the served model id.
pub fn probe(node: &Node) -> Endpoint {
    let local = node.llama_url.trim_end_matches('/').to_string();
    let port = crate::nodeops::parse_bind(&node.llama_url).1;
    let exposed = crate::settings::exposed();
    // The LAN address is derived whenever a routable IP exists - independent of
    // whether we're exposed yet - so the renderer can print the network command
    // and, if closed, the one line that opens it.
    let lan = crate::settings::lan_ip().map(|ip| format!("http://{ip}:{port}"));
    // Primary base (back-compat `base_url`/`openai_base`): LAN when exposed and
    // discoverable, else localhost. Health + served are always probed on the
    // node URL (localhost, which 0.0.0.0 still covers).
    let base = if exposed {
        lan.clone().unwrap_or_else(|| local.clone())
    } else {
        local.clone()
    };
    Endpoint {
        openai_base: format!("{base}/v1"),
        healthy: llama::health_ok(&node.llama_url),
        model: llama::served_name(&node.llama_url),
        base_url: base,
        api_key: crate::settings::api_key(),
        exposed,
        local_base: format!("{local}/v1"),
        lan_base: lan.map(|b| format!("{b}/v1")),
        identity_branded: llama::chat_template(&node.llama_url)
            .map(|t| crate::identity::template_is_branded(&t))
            .unwrap_or(false),
    }
}

/// The model id to put in example requests: the served model, or a placeholder
/// (llama-server ignores the field, but real OpenAI clients require one).
fn model_field(ep: &Endpoint) -> String {
    ep.model
        .clone()
        .unwrap_or_else(|| "local-model".to_string())
}

/// The curl + OpenAI-SDK block for a SINGLE base URL. Pure; the auth pieces
/// vary with whether an API key is required.
fn connect_block(base: &str, model: &str, api_key: &Option<String>) -> String {
    let (curl_auth, sdk_key) = match api_key {
        Some(k) => (
            format!("  -H 'Authorization: Bearer {k}' \\\n  "),
            k.clone(),
        ),
        None => (String::new(), "not-needed".to_string()),
    };
    format!(
        "curl {base}/chat/completions \\\n  \
         {curl_auth}\
           -H 'Content-Type: application/json' \\\n  \
           -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'\n\
         \n\
         from openai import OpenAI\n\
         client = OpenAI(base_url=\"{base}\", api_key=\"{sdk_key}\")\n\
         print(client.chat.completions.create(model=\"{model}\",\n  \
           messages=[{{\"role\": \"user\", \"content\": \"hello\"}}]).choices[0].message.content)"
    )
}

/// Copy-paste connection snippets, in two sections: from THIS machine
/// (localhost, always works) and from ANOTHER machine on the LAN (the routable
/// IP). The network section shows the address regardless of exposure, and - when
/// the server is still localhost-only - the one command that opens it. Pure so
/// the CLI and the TUI overlay render identically.
pub fn snippets(ep: &Endpoint) -> String {
    let model = model_field(ep);
    // Fall back to the primary base when an older remote peer didn't send the
    // split localhost base.
    let local = if ep.local_base.is_empty() {
        ep.openai_base.as_str()
    } else {
        ep.local_base.as_str()
    };

    let mut out = String::new();
    out.push_str("# --- From this machine (local) ---\n");
    out.push_str(&connect_block(local, &model, &ep.api_key));
    out.push_str("\n\n");

    match &ep.lan_base {
        Some(lan) if ep.exposed => {
            out.push_str("# --- From another machine on the network (exposed now) ---\n");
            out.push_str(&connect_block(lan, &model, &ep.api_key));
        }
        Some(lan) => {
            out.push_str("# --- From another machine on the network ---\n");
            out.push_str(
                "# localhost-only right now - open it with:  llmtune endpoint expose on\n",
            );
            if ep.api_key.is_none() {
                out.push_str("# (open server: set a key first - llmtune endpoint api-key set)\n");
            }
            out.push_str(&connect_block(lan, &model, &ep.api_key));
        }
        None => {
            out.push_str("# --- From another machine on the network ---\n");
            out.push_str("# no routable LAN address found (check the host's network)");
        }
    }

    let note = match &ep.api_key {
        Some(k) => format!(
            "\n\n# Any OpenAI-compatible agent/harness: point it at the base above\n\
             #   with header  Authorization: Bearer {k}"
        ),
        None => "\n\n# Any OpenAI-compatible agent/harness: point it at the base above (no key)"
            .to_string(),
    };
    out.push_str(&note);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(model: Option<&str>, healthy: bool) -> Endpoint {
        Endpoint {
            base_url: "http://127.0.0.1:8080".into(),
            openai_base: "http://127.0.0.1:8080/v1".into(),
            healthy,
            model: model.map(|s| s.to_string()),
            api_key: None,
            exposed: false,
            local_base: "http://127.0.0.1:8080/v1".into(),
            lan_base: Some("http://192.0.2.10:8080/v1".into()),
            identity_branded: false,
        }
    }

    #[test]
    fn snippets_include_api_key_when_set() {
        let mut e = ep(Some("m"), true);
        e.api_key = Some("llt-secret".into());
        let s = snippets(&e);
        assert!(s.contains("Authorization: Bearer llt-secret"));
        assert!(s.contains("api_key=\"llt-secret\""));
        // and the no-key path stays keyless
        assert!(snippets(&ep(Some("m"), true)).contains("api_key=\"not-needed\""));
    }

    #[test]
    fn snippets_carry_url_and_model() {
        let s = snippets(&ep(Some("Qwen3-30B"), true));
        assert!(s.contains("http://127.0.0.1:8080/v1/chat/completions"));
        assert!(s.contains("\"model\":\"Qwen3-30B\""));
        assert!(s.contains("base_url=\"http://127.0.0.1:8080/v1\""));
        assert!(s.contains("OpenAI"));
    }

    #[test]
    fn snippets_use_placeholder_when_no_model() {
        let s = snippets(&ep(None, false));
        assert!(s.contains("local-model"));
    }

    #[test]
    fn snippets_have_local_and_network_sections() {
        let s = snippets(&ep(Some("m"), true));
        assert!(s.contains("From this machine (local)"), "local header");
        assert!(
            s.contains("From another machine on the network"),
            "network header"
        );
        // both bases show up, not just localhost
        assert!(s.contains("http://127.0.0.1:8080/v1/chat/completions"));
        assert!(s.contains("http://192.0.2.10:8080/v1/chat/completions"));
    }

    #[test]
    fn network_section_shows_expose_command_when_localhost_only() {
        // Not exposed -> the network block must tell the owner how to open it.
        let s = snippets(&ep(Some("m"), true));
        assert!(s.contains("llmtune endpoint expose on"));
        // open server (no key) -> also nudge to set a key first
        assert!(s.contains("api-key set"));
    }

    #[test]
    fn network_section_drops_expose_hint_when_exposed() {
        let mut e = ep(Some("m"), true);
        e.exposed = true;
        let s = snippets(&e);
        assert!(s.contains("exposed now"));
        assert!(!s.contains("expose on"), "no expose nag once exposed");
    }

    #[test]
    fn network_section_notes_when_no_lan_ip() {
        let mut e = ep(Some("m"), true);
        e.lan_base = None;
        let s = snippets(&e);
        assert!(s.contains("no routable LAN address"));
        // local section still present
        assert!(s.contains("http://127.0.0.1:8080/v1/chat/completions"));
    }

    #[test]
    fn snippets_fall_back_to_openai_base_for_old_peers() {
        // A remote peer on an older binary sends no split local_base.
        let mut e = ep(Some("m"), true);
        e.local_base = String::new();
        e.openai_base = "http://198.51.100.5:8080/v1".into();
        let s = snippets(&e);
        assert!(s.contains("http://198.51.100.5:8080/v1/chat/completions"));
    }

    #[test]
    fn probe_strips_trailing_slash() {
        let mut node = crate::config::Config::load_str("")
            .unwrap()
            .local_node()
            .clone();
        node.llama_url = "http://127.0.0.1:8080/".into();
        let e = probe(&node);
        assert_eq!(e.base_url, "http://127.0.0.1:8080");
        assert_eq!(e.openai_base, "http://127.0.0.1:8080/v1");
    }
}
