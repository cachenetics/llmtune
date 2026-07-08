// SPDX-License-Identifier: GPL-2.0-only
//! Persistent node-local settings. Currently just the llama-server bind host:
//! `127.0.0.1` (localhost-only, the default) or `0.0.0.0` (exposed to the LAN).
//! Read at model-load time and baked into the systemd drop-in's `--host`, so it
//! survives reloads/reboots. Stored as `~/.config/llmtune/settings.toml`.
//!
//! Exposing binds llama-server on all interfaces with NO authentication - only
//! do it on a trusted network. Health/served probes stay on 127.0.0.1 (which
//! 0.0.0.0 still covers), so nothing else has to change.

use crate::paths;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const LOCAL: &str = "127.0.0.1";
pub const EXPOSED: &str = "0.0.0.0";

const HEADER: &str = "# llmtune node-local settings\n\n";

#[derive(Debug, Default, Serialize, Deserialize)]
struct SettingsFile {
    #[serde(default)]
    bind_host: Option<String>,
    /// llama-server `--api-key`. When set, the server requires it on the API
    /// endpoints and llmtune's own probes send it too.
    #[serde(default)]
    api_key: Option<String>,
    /// The last model deliberately loaded - so boot-restore knows what to serve
    /// after a reboot reverts the (root-subvol) drop-in.
    #[serde(default)]
    served_model: Option<String>,
    /// Fleet-dashboard alert thresholds (`[alerts]`). Optional: a settings.toml
    /// without the section behaves exactly like the built-in defaults, and an
    /// unrelated edit never bakes the defaults into the file.
    #[serde(default)]
    alerts: Option<AlertsCfg>,
}

/// Alert thresholds for the fleet dashboard: they drive the per-node card temp
/// colors, the alert banner, and the event log. Every field is optional in the
/// file; the defaults match the historical hardcoded values (warn 70 C, crit
/// 80 C, no throughput floor, unreachable alerts immediately).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertsCfg {
    /// Card temp turns yellow at/above this (C).
    #[serde(default = "default_temp_warn")]
    pub temp_warn: f64,
    /// Card temp turns red and the banner alerts at/above this (C).
    #[serde(default = "default_temp_crit")]
    pub temp_crit: f64,
    /// Alert when a node's last bench gen tok/s falls below this (off when unset).
    #[serde(default)]
    pub min_tok_s: Option<f64>,
    /// Only alert "unreachable" after the node has been down this long, so one
    /// dropped probe doesn't trip the banner (immediate when unset).
    #[serde(default)]
    pub unreachable_after_secs: Option<u64>,
}

fn default_temp_warn() -> f64 {
    70.0
}

fn default_temp_crit() -> f64 {
    80.0
}

impl Default for AlertsCfg {
    fn default() -> Self {
        AlertsCfg {
            temp_warn: default_temp_warn(),
            temp_crit: default_temp_crit(),
            min_tok_s: None,
            unreachable_after_secs: None,
        }
    }
}

/// The configured alert thresholds (`[alerts]` in settings.toml), or the
/// defaults when the section is absent.
pub fn alerts() -> AlertsCfg {
    load().alerts.unwrap_or_default()
}

fn settings_path() -> Option<PathBuf> {
    paths::config_file("settings.toml")
}

fn load() -> SettingsFile {
    settings_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| toml::from_str(&s).ok())
        .unwrap_or_default()
}

/// The host llama-server should bind to. Default localhost.
pub fn bind_host() -> String {
    load().bind_host.unwrap_or_else(|| LOCAL.to_string())
}

/// Is the server set to bind all interfaces (exposed to the network)?
pub fn exposed() -> bool {
    bind_host() == EXPOSED
}

/// Set exposure: `true` -> bind `0.0.0.0` (LAN), `false` -> `127.0.0.1` (localhost).
/// Takes effect the next time a model is loaded (the drop-in is regenerated).
/// Atomic (durable temp+rename): settings are written from concurrent load
/// paths (e.g. the proxy's swap-on-demand), so no torn file is observable.
pub fn set_exposed(on: bool) -> Result<()> {
    let path = settings_path().context("no HOME for settings path")?;
    paths::edit_toml_secret(&path, HEADER, |f: &mut SettingsFile| {
        f.bind_host = Some(if on { EXPOSED } else { LOCAL }.to_string());
    })
}

/// The configured API key, if any. When present, llama-server is launched with
/// `--api-key <k>` and every llmtune request carries `Authorization: Bearer`.
pub fn api_key() -> Option<String> {
    load().api_key.filter(|k| !k.is_empty())
}

/// Set (`Some`) or clear (`None`) the API key. Takes effect on the next load.
/// Atomic - see [`set_exposed`].
pub fn set_api_key(key: Option<&str>) -> Result<()> {
    let path = settings_path().context("no HOME for settings path")?;
    paths::edit_toml_secret(&path, HEADER, |f: &mut SettingsFile| {
        f.api_key = key.filter(|k| !k.is_empty()).map(String::from);
    })
}

/// The last deliberately-loaded model (for boot-restore).
pub fn served_model() -> Option<String> {
    load().served_model.filter(|m| !m.is_empty())
}

/// Remember the loaded model so boot-restore can bring it back. Best-effort:
/// called from the load path, where a failure to persist shouldn't fail the
/// load. Atomic - see [`set_exposed`].
pub fn set_served_model(model: &str) {
    let path = match settings_path() {
        Some(p) => p,
        None => return,
    };
    let _ = paths::edit_toml_secret(&path, HEADER, |f: &mut SettingsFile| {
        f.served_model = Some(model.to_string());
    });
}

/// Generate a random API key (24 bytes of /dev/urandom, hex). Falls back to a
/// pid-tagged value if urandom is unreadable (weak, but keys guard a trusted LAN).
pub fn generate_key() -> String {
    use std::io::Read;
    let mut buf = [0u8; 24];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok()
    {
        let hex: String = buf.iter().map(|b| format!("{b:02x}")).collect();
        format!("llt-{hex}")
    } else {
        format!("llt-{}", std::process::id())
    }
}

/// This host's primary LAN IPv4 (best-effort, for showing a reachable address
/// when exposed). Uses a connected UDP socket to discover the outbound-route IP;
/// no packets are sent. Returns None if it can't be determined.
pub fn lan_ip() -> Option<String> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("1.1.1.1:53").ok()?;
    sock.local_addr().ok().map(|a| a.ip().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_alerts_section_yields_defaults() {
        // A settings.toml without [alerts] behaves exactly as before: the
        // historical hardcoded 70/80 thresholds, no optional gates.
        let f: SettingsFile = toml::from_str("bind_host = \"127.0.0.1\"\n").unwrap();
        assert!(f.alerts.is_none());
        let a = f.alerts.unwrap_or_default();
        assert_eq!(a.temp_warn, 70.0);
        assert_eq!(a.temp_crit, 80.0);
        assert_eq!(a.min_tok_s, None);
        assert_eq!(a.unreachable_after_secs, None);
        // an entirely empty file too
        let f: SettingsFile = toml::from_str("").unwrap();
        assert!(f.alerts.is_none());
    }

    #[test]
    fn partial_alerts_section_fills_defaults() {
        // Setting one threshold must not zero the others.
        let f: SettingsFile = toml::from_str("[alerts]\ntemp_crit = 75.0\n").unwrap();
        let a = f.alerts.unwrap();
        assert_eq!(a.temp_crit, 75.0);
        assert_eq!(a.temp_warn, 70.0, "unset fields keep their defaults");
        assert_eq!(a.min_tok_s, None);
    }

    #[test]
    fn full_alerts_section_parses() {
        let f: SettingsFile = toml::from_str(
            "[alerts]\ntemp_warn = 65.0\ntemp_crit = 78.0\nmin_tok_s = 30.0\nunreachable_after_secs = 60\n",
        )
        .unwrap();
        let a = f.alerts.unwrap();
        assert_eq!(a.temp_warn, 65.0);
        assert_eq!(a.temp_crit, 78.0);
        assert_eq!(a.min_tok_s, Some(30.0));
        assert_eq!(a.unreachable_after_secs, Some(60));
    }

    #[test]
    fn settings_roundtrip_preserves_alerts_and_omits_when_absent() {
        // Present: survives the edit_toml_secret read-modify-write shape.
        let mut f: SettingsFile = toml::from_str("[alerts]\ntemp_crit = 75.0\n").unwrap();
        f.api_key = Some("k".into());
        let txt = toml::to_string_pretty(&f).unwrap();
        let back: SettingsFile = toml::from_str(&txt).unwrap();
        assert_eq!(back.alerts.unwrap().temp_crit, 75.0);
        // Absent: an unrelated edit must NOT bake an [alerts] table (so a later
        // change of the built-in defaults still applies to unconfigured nodes).
        let f = SettingsFile {
            api_key: Some("k".into()),
            ..Default::default()
        };
        let txt = toml::to_string_pretty(&f).unwrap();
        assert!(!txt.contains("[alerts]"), "{txt}");
    }
}
