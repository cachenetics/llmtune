// SPDX-License-Identifier: GPL-2.0-only
//! Per-architecture launch profiles - the known-good llama-server flags for the
//! BC-250, shipped as data (`profiles.toml`) and overridable by the user.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// One launch profile: the binary, library path, environment and flags to serve
/// a model of a matching architecture family.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Profile {
    pub id: String,
    /// Architecture-family prefixes this profile matches (empty for `_default`).
    #[serde(default)]
    pub arch_match: Vec<String>,
    /// A managed build to launch from (resolved through the build manager's
    /// `current` version). When set, `bin` is the binary NAME within that build
    /// (e.g. "llama-server") and `ld_path` is taken from the build dir. When
    /// unset, `bin` is a literal path (the pre-build-manager behaviour).
    #[serde(default)]
    pub build: Option<String>,
    pub bin: String,
    #[serde(default)]
    pub ld_path: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub flags: String,
}

impl Profile {
    /// The launch binary + `LD_LIBRARY_PATH` to actually exec. Resolves a
    /// `build = "<name>"` reference through the build manager's current version,
    /// falling back to the literal `bin`/`ld_path` when no build is referenced.
    ///
    /// If a build IS referenced but isn't installed, returns the bare binary name
    /// (so the failure is visible: the swap health-check reverts and `doctor`
    /// flags the missing build) rather than silently using a stale path.
    pub fn launch(&self) -> (String, Option<String>) {
        if let Some(bname) = self.build.as_deref().filter(|b| !b.is_empty()) {
            let binfile = std::path::Path::new(&self.bin)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "llama-server".to_string());
            if let Some(p) = crate::build::current_bin(bname, &binfile) {
                let ld = crate::build::current_dir(bname).map(|d| d.display().to_string());
                return (p.display().to_string(), ld.or_else(|| self.ld_path.clone()));
            }
            return (binfile, self.ld_path.clone());
        }
        (self.bin.clone(), self.ld_path.clone())
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ProfilesFile {
    #[serde(default)]
    profile: Vec<Profile>,
}

/// The bundled seed (compiled in); a user file fully replaces it.
const SEED: &str = include_str!("../profiles.toml");

/// Path to the user override file (`~/.config/llmtune/profiles.toml`).
pub fn user_path() -> Option<PathBuf> {
    crate::paths::config_file("profiles.toml")
}

/// Whether a user override file exists.
pub fn user_file_exists() -> bool {
    user_path().map(|p| p.is_file()).unwrap_or(false)
}

/// Persist `profiles` to the user override file (atomic durable write, creating
/// parent dirs). This is how option edits (`profile set`) are saved; once
/// written it fully replaces the bundled seed on the next load.
pub fn save_user(profiles: &[Profile]) -> Result<()> {
    let path = user_path().context("no HOME for user profiles path")?;
    let body = ProfilesFile {
        profile: profiles.to_vec(),
    };
    let header = "# llmtune per-architecture launch profiles (user overrides).\n\
                  # Managed by `llmtune profile set`; edit by hand if you prefer.\n\n";
    crate::paths::save_toml_atomic(&path, header, &body)
}

// ---------------------------------------------------------------------------
// Per-MODEL flag overrides (keyed by model filename). These take precedence over
// the per-arch profile flags at swap time, so one model can carry its own flags.
// ---------------------------------------------------------------------------

fn overrides_path() -> Option<PathBuf> {
    user_path().map(|p| p.with_file_name("overrides.toml"))
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct OverridesFile {
    #[serde(default)]
    flags: BTreeMap<String, String>,
}

/// Load the per-model flag overrides (model filename -> flags). Empty if absent.
pub fn load_overrides() -> BTreeMap<String, String> {
    overrides_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| toml::from_str::<OverridesFile>(&t).ok())
        .map(|o| o.flags)
        .unwrap_or_default()
}

/// Atomic durable write - overrides are written from concurrent load paths
/// (the proxy's swap-on-demand), so a torn file must never be observable.
fn save_overrides(map: &BTreeMap<String, String>) -> Result<()> {
    let path = overrides_path().context("no HOME for overrides path")?;
    let body = OverridesFile { flags: map.clone() };
    let header = "# llmtune per-model llama.cpp flag overrides (managed by the TUI editor).\n\n";
    crate::paths::save_toml_atomic(&path, header, &body)
}

/// Set (or replace) a model's flag override.
pub fn set_override(model: &str, flags: &str) -> Result<()> {
    let mut map = load_overrides();
    map.insert(model.to_string(), flags.to_string());
    save_overrides(&map)
}

/// Remove a model's flag override (revert to its profile default).
pub fn clear_override(model: &str) -> Result<()> {
    let mut map = load_overrides();
    map.remove(model);
    save_overrides(&map)
}

/// Load profiles: the user's `~/.config/llmtune/profiles.toml` if present,
/// else the bundled seed. Validates the set so downstream resolution can't panic.
pub fn load() -> Result<Vec<Profile>> {
    let profiles = if let Some(p) = user_path().filter(|p| p.is_file()) {
        let txt =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        parse(&txt).with_context(|| format!("parsing {}", p.display()))?
    } else {
        parse(SEED).context("parsing bundled profiles seed")?
    };
    validate(&profiles)?;
    Ok(profiles)
}

fn parse(txt: &str) -> Result<Vec<Profile>> {
    let f: ProfilesFile = toml::from_str(txt)?;
    Ok(f.profile)
}

/// A profile set must be non-empty and contain a `_default` fallback, otherwise
/// resolution has nothing to fall back to. Catches a hand-edited user file.
fn validate(profiles: &[Profile]) -> Result<()> {
    if profiles.is_empty() {
        anyhow::bail!("no profiles defined - check ~/.config/llmtune/profiles.toml");
    }
    if !profiles.iter().any(|p| p.id == "_default") {
        anyhow::bail!(
            "profiles must include a `_default` entry (fallback for unknown architectures) \
             - check ~/.config/llmtune/profiles.toml"
        );
    }
    Ok(())
}

/// Resolve a model architecture to a profile by family prefix, in file order,
/// skipping `_default`. Returns `(profile, used_default)`.
pub fn resolve<'a>(profiles: &'a [Profile], arch: &str) -> (&'a Profile, bool) {
    let a = arch.to_lowercase();
    for p in profiles {
        if p.id == "_default" {
            continue;
        }
        if p.arch_match
            .iter()
            .any(|m| a.starts_with(&m.to_lowercase()))
        {
            return (p, false);
        }
    }
    let def = profiles
        .iter()
        .find(|p| p.id == "_default")
        .expect("profiles must include a _default entry");
    (def, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> Vec<Profile> {
        parse(SEED).unwrap()
    }

    #[test]
    fn seed_parses_and_has_default() {
        let ps = seed();
        assert!(ps.iter().any(|p| p.id == "_default"));
        assert!(ps.len() >= 5);
    }

    #[test]
    fn moe_resolves_before_dense() {
        let ps = seed();
        // qwen35moe must win over qwen35 for an a3b moe arch.
        let (p, used) = resolve(&ps, "qwen35moe");
        assert_eq!(p.id, "qwen35moe");
        assert!(!used);
        let (p, used) = resolve(&ps, "qwen35");
        assert_eq!(p.id, "qwen35");
        assert!(!used);
    }

    #[test]
    fn validate_requires_nonempty_and_default() {
        assert!(validate(&[]).is_err());
        let no_default: Vec<Profile> = seed().into_iter().filter(|p| p.id != "_default").collect();
        assert!(validate(&no_default).is_err());
        assert!(validate(&seed()).is_ok());
    }

    #[test]
    fn serialize_roundtrip_preserves_profiles() {
        // what `profile set` writes must parse back identically (order + flags).
        let ps = parse(SEED).unwrap();
        let body = ProfilesFile {
            profile: ps.clone(),
        };
        let txt = toml::to_string_pretty(&body).unwrap();
        let back = parse(&txt).unwrap();
        assert_eq!(back.len(), ps.len());
        assert_eq!(
            back.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
            ps.iter().map(|p| p.id.clone()).collect::<Vec<_>>()
        );
        let q = back.iter().find(|p| p.id == "qwen35moe").unwrap();
        assert!(q.flags.contains("--spec-type draft-mtp"));
        assert_eq!(q.build.as_deref(), Some("vulkan"));
        assert_eq!(
            q.env.get("GGML_VK_PREFER_HOST_MEMORY").map(|s| s.as_str()),
            Some("1")
        );
    }

    #[test]
    fn family_prefix_and_fallback() {
        let ps = seed();
        let (p, used) = resolve(&ps, "gemma3");
        assert_eq!(p.id, "gemma");
        assert!(!used);
        let (p, used) = resolve(&ps, "phi4");
        assert_eq!(p.id, "_default");
        assert!(used);
    }
}
