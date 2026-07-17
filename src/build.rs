// SPDX-License-Identifier: GPL-2.0-only
//! llama.cpp build manager - llmtune owns the toolchain for the BC-250.
//!
//! Getting a working Vulkan/MTP `llama.cpp` is the hardest part of running LLMs
//! on this silicon, so llmtune installs, updates and rolls back named builds as
//! versioned artifacts. Builds are described as DATA (`builds.toml`, the same
//! philosophy as `profiles.toml`): `{name, git_url, ref, cmake_flags, out_bins}`.
//!
//! Layout under the state dir (`/var/lib/llmtune` as root, else `~/.local/share/
//! llmtune`, overridable with `LLMTUNE_STATE_DIR`):
//!
//! ```text
//!   builds/<name>/<commit>/        one installed version (binaries + runtime libs)
//!   builds/<name>/current -> <commit>   relative symlink to the live version
//!   builds/<name>/installs.json    durable install-order ledger (newest last)
//!   src/<name>/                    persistent git checkout (re-fetched on update)
//! ```
//!
//! The orchestration (resolve ref -> version dir -> atomic install -> flip current
//! -> gc) is parameterized over a [`Builder`] seam, so install/update/rollback/gc
//! are unit-tested with a mock that fabricates binaries - no llama.cpp compile and
//! no BC-250 required. [`RealBuilder`] is the production git + cmake implementation.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One llama.cpp build recipe (a `[[build]]` entry in `builds.toml`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct BuildSpec {
    pub name: String,
    pub git_url: String,
    /// The git ref to track (branch, tag, or commit). `ref` is a reserved word in
    /// Rust, so the field is `git_ref` with a serde rename.
    #[serde(rename = "ref")]
    pub git_ref: String,
    #[serde(default)]
    pub cmake_flags: String,
    /// Binaries the build must produce (also what `has_bins` validates).
    pub out_bins: Vec<String>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct BuildsFile {
    #[serde(default, rename = "build")]
    builds: Vec<BuildSpec>,
}

/// The bundled seed (compiled in); a user file fully replaces it.
const SEED: &str = include_str!("../builds.toml");

/// Default number of prior versions kept for rollback (plus `current`).
pub const DEFAULT_RETAIN: usize = 3;

/// Path to the user override file (`~/.config/llmtune/builds.toml`).
pub fn user_path() -> Option<PathBuf> {
    crate::paths::config_file("builds.toml")
}

/// Load build recipes: the user's file if present, else the bundled seed.
pub fn load() -> Result<Vec<BuildSpec>> {
    let specs = if let Some(p) = user_path().filter(|p| p.is_file()) {
        let txt = fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        parse(&txt).with_context(|| format!("parsing {}", p.display()))?
    } else {
        parse(SEED).context("parsing bundled builds seed")?
    };
    validate(&specs)?;
    Ok(specs)
}

fn parse(txt: &str) -> Result<Vec<BuildSpec>> {
    let f: BuildsFile = toml::from_str(txt)?;
    Ok(f.builds)
}

/// Names must be non-empty, unique, and filesystem-safe (they become directory
/// names); `out_bins` must be non-empty. Catches a hand-edited user file early.
fn validate(specs: &[BuildSpec]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for s in specs {
        if s.name.is_empty() {
            bail!("a build has an empty name in builds.toml");
        }
        if !s
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            bail!(
                "build name `{}` is not filesystem-safe (use [A-Za-z0-9._-])",
                s.name
            );
        }
        if s.name == "current" || s.name.starts_with('.') {
            bail!("build name `{}` is reserved", s.name);
        }
        if !seen.insert(&s.name) {
            bail!("duplicate build name `{}` in builds.toml", s.name);
        }
        if s.git_url.is_empty() {
            bail!("build `{}` has no git_url", s.name);
        }
        // A url/ref beginning with '-' is parsed by git as an option
        // (`--upload-pack=<cmd>` on ls-remote = command execution); restrict the
        // url to known transports (blocks `ext::`/`fd::` transports too).
        if s.git_url.starts_with('-')
            || !["https://", "http://", "ssh://", "git://", "git@"]
                .iter()
                .any(|p| s.git_url.starts_with(p))
        {
            bail!(
                "build `{}`: git_url must be an http(s)/ssh/git URL (got `{}`)",
                s.name,
                s.git_url
            );
        }
        if s.git_ref.is_empty() {
            bail!("build `{}` has no ref", s.name);
        }
        if s.git_ref.starts_with('-') {
            bail!("build `{}`: ref may not start with '-'", s.name);
        }
        if s.out_bins.is_empty() {
            bail!("build `{}` lists no out_bins", s.name);
        }
    }
    Ok(())
}

/// Look up a recipe by name.
pub fn spec<'a>(specs: &'a [BuildSpec], name: &str) -> Option<&'a BuildSpec> {
    specs.iter().find(|s| s.name == name)
}

// ---------------------------------------------------------------------------
// State-dir resolution - the SHARED system home (see paths.rs for the two
// deliberate state-dir flavours).
// ---------------------------------------------------------------------------

/// llmtune's state root - the fixed system home `/var/lib/llmtune` so a user
/// run and the internal sudo (and models next door) all agree on the path.
/// `LLMTUNE_STATE_DIR` overrides; resolution lives in `paths::shared_state_dir`.
pub fn state_dir() -> PathBuf {
    crate::paths::shared_state_dir()
}

fn builds_root() -> PathBuf {
    state_dir().join("builds")
}

fn name_dir(name: &str) -> PathBuf {
    builds_root().join(name)
}

fn source_dir(name: &str) -> PathBuf {
    state_dir().join("src").join(name)
}

// ---------------------------------------------------------------------------
// The build seam.
// ---------------------------------------------------------------------------

/// Produces a build's artifacts. Split out so the install/version/rollback logic
/// is testable without compiling llama.cpp.
pub trait Builder {
    /// Resolve the spec's `ref` to a concrete, full commit sha (so each version
    /// dir is reproducible and auditable even when `ref` tracks a branch).
    fn resolve_ref(&mut self, spec: &BuildSpec) -> Result<String>;
    /// Build `spec` at `commit`, placing every `out_bin` (and its runtime libs)
    /// into `stage` (a fresh empty dir). Heavy: git fetch/checkout + cmake.
    fn produce(&mut self, spec: &BuildSpec, commit: &str, stage: &Path) -> Result<()>;
}

/// Short, stable directory slug for a commit (first 12 hex chars when it looks
/// like a sha; otherwise a filesystem-safe squash of the ref).
pub fn slug(commit: &str) -> String {
    let c = commit.trim();
    if is_hex(c) && c.len() >= 7 {
        return c[..c.len().min(12)].to_lowercase();
    }
    c.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '-'
            }
        })
        .take(24)
        .collect()
}

fn is_hex(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// What an install actually did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Compiled a new version and made it current.
    Built,
    /// The resolved commit was already installed and already current - no-op.
    AlreadyCurrent,
    /// The resolved commit was already installed; just made it current (no rebuild).
    Switched,
}

#[derive(Debug, Clone)]
pub struct InstallOutcome {
    pub name: String,
    pub commit: String,
    pub slug: String,
    pub dir: PathBuf,
    pub action: Action,
    pub previous: Option<String>,
    /// Whether the installed version carries an rpc worker binary (any accepted
    /// name). False means single-node serving works but the node can't join a
    /// cluster - a warning, never an install failure (rpc is optional).
    pub rpc_present: bool,
}

#[derive(Debug, Clone)]
pub struct RollbackOutcome {
    pub name: String,
    pub from: Option<String>,
    pub to: String,
}

/// One installed version on disk.
#[derive(Debug, Clone)]
pub struct Version {
    pub slug: String,
    pub current: bool,
}

/// Status of a build name: configured recipe (if any) + installed versions.
#[derive(Debug, Clone)]
pub struct BuildStatus {
    pub name: String,
    pub spec: Option<BuildSpec>,
    pub current: Option<String>,
    pub versions: Vec<Version>,
}

// ---------------------------------------------------------------------------
// Install-order ledger (durable, deterministic ordering for gc + rollback).
// ---------------------------------------------------------------------------

fn ledger_path(name: &str) -> PathBuf {
    name_dir(name).join("installs.json")
}

fn ledger_load(name: &str) -> Vec<String> {
    fs::read_to_string(ledger_path(name))
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default()
}

fn ledger_save(name: &str, order: &[String]) -> Result<()> {
    let path = ledger_path(name);
    // Crash-safe (temp -> fsync -> rename -> fsync dir), so a power loss never
    // leaves a half-written ledger.
    crate::paths::write_durable(&path, serde_json::to_string_pretty(order)?.as_bytes())
}

// ---------------------------------------------------------------------------
// Symlink / version helpers.
// ---------------------------------------------------------------------------

/// The slug `current` points at, if the symlink exists and is valid.
fn current_slug(name: &str) -> Option<String> {
    let link = name_dir(name).join("current");
    let target = fs::read_link(&link).ok()?;
    target.file_name().map(|s| s.to_string_lossy().into_owned())
}

/// Atomically point `current` at `slug` (relative target, so the tree relocates).
fn flip_current(name: &str, slug: &str) -> Result<()> {
    let dir = name_dir(name);
    fs::create_dir_all(&dir)?;
    let link = dir.join("current");
    let tmp = dir.join(".current.tmp");
    let _ = fs::remove_file(&tmp);
    std::os::unix::fs::symlink(slug, &tmp)
        .with_context(|| format!("creating current symlink for build `{name}`"))?;
    // rename over the existing symlink is atomic on POSIX.
    fs::rename(&tmp, &link).context("activating current symlink")?;
    Ok(())
}

/// Does this version dir hold every required binary?
fn has_bins(vdir: &Path, out_bins: &[String]) -> bool {
    out_bins.iter().all(|b| vdir.join(b).is_file())
}

/// Installed version slugs (dirs that aren't `current`/dot-dirs/ledger/staging).
fn installed_slugs(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(name_dir(name)) {
        for e in rd.flatten() {
            if !e.path().is_dir() {
                continue;
            }
            let n = e.file_name().to_string_lossy().into_owned();
            if n == "current" || n.starts_with('.') {
                continue;
            }
            out.push(n);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Orchestration.
// ---------------------------------------------------------------------------

/// Install (or activate) `spec` at its resolved ref, keeping `retain` prior
/// versions for rollback. Idempotent: an already-current commit is a no-op; an
/// already-built-but-inactive commit is activated without a rebuild.
pub fn install<B: Builder>(
    spec: &BuildSpec,
    builder: &mut B,
    retain: usize,
) -> Result<InstallOutcome> {
    let _lock = crate::lock::LockGuard::build().map_err(|_| {
        anyhow::anyhow!("another build is in progress - try again when it finishes")
    })?;

    let commit = builder.resolve_ref(spec)?;
    if commit.trim().is_empty() {
        bail!(
            "could not resolve ref `{}` for build `{}`",
            spec.git_ref,
            spec.name
        );
    }
    let slug = slug(&commit);
    let ndir = name_dir(&spec.name);
    fs::create_dir_all(&ndir).with_context(|| format!("creating build dir {}", ndir.display()))?;
    let vdir = ndir.join(&slug);
    let cur = current_slug(&spec.name);

    // Already built? Activate without recompiling.
    if vdir.is_dir() && has_bins(&vdir, &spec.out_bins) {
        if cur.as_deref() == Some(slug.as_str()) {
            let rpc_present = has_rpc(&vdir);
            return Ok(InstallOutcome {
                name: spec.name.clone(),
                commit,
                slug,
                dir: vdir,
                action: Action::AlreadyCurrent,
                previous: cur,
                rpc_present,
            });
        }
        let rpc_present = has_rpc(&vdir);
        flip_current(&spec.name, &slug)?;
        ledger_touch(&spec.name, &slug)?;
        gc(&spec.name, retain)?;
        return Ok(InstallOutcome {
            name: spec.name.clone(),
            commit,
            slug,
            dir: vdir,
            action: Action::Switched,
            previous: cur,
            rpc_present,
        });
    }

    // Build into a fresh staging dir, then atomically move into place.
    let stage = ndir.join(format!(".stage-{slug}"));
    let _ = fs::remove_dir_all(&stage);
    fs::create_dir_all(&stage)
        .with_context(|| format!("creating staging dir {}", stage.display()))?;
    let produced = builder.produce(spec, &commit, &stage);
    if let Err(e) = produced {
        let _ = fs::remove_dir_all(&stage);
        return Err(e).with_context(|| format!("building `{}` at {commit}", spec.name));
    }
    if !has_bins(&stage, &spec.out_bins) {
        let missing: Vec<&str> = spec
            .out_bins
            .iter()
            .filter(|b| !stage.join(b).is_file())
            .map(|s| s.as_str())
            .collect();
        let _ = fs::remove_dir_all(&stage);
        bail!(
            "build `{}` finished but is missing binaries: {} - check cmake_flags / out_bins",
            spec.name,
            missing.join(", ")
        );
    }

    // Replace any partial leftover at the target, then atomic rename into place.
    let _ = fs::remove_dir_all(&vdir);
    fs::rename(&stage, &vdir)
        .with_context(|| format!("installing build into {}", vdir.display()))?;

    let rpc_present = has_rpc(&vdir);
    flip_current(&spec.name, &slug)?;
    ledger_touch(&spec.name, &slug)?;
    gc(&spec.name, retain)?;

    Ok(InstallOutcome {
        name: spec.name.clone(),
        commit,
        slug,
        dir: vdir,
        action: Action::Built,
        previous: cur,
        rpc_present,
    })
}

/// Append `slug` to the install ledger as newest (moving it if already present).
fn ledger_touch(name: &str, slug: &str) -> Result<()> {
    let mut order = ledger_load(name);
    order.retain(|s| s != slug);
    order.push(slug.to_string());
    ledger_save(name, &order)
}

/// Roll `current` back to the most-recently-installed version that isn't current.
pub fn rollback(name: &str, _specs: &[BuildSpec]) -> Result<RollbackOutcome> {
    let _lock = crate::lock::LockGuard::build().map_err(|_| {
        anyhow::anyhow!("another build is in progress - try again when it finishes")
    })?;
    let cur = current_slug(name);
    let installed: std::collections::BTreeSet<String> = installed_slugs(name).into_iter().collect();
    // Walk the ledger newest-first; pick the first installed, valid, non-current.
    let order = ledger_load(name);
    let target = order
        .iter()
        .rev()
        .find(|s| Some(s.as_str()) != cur.as_deref() && installed.contains(*s))
        .cloned()
        // Ledger empty/stale? Fall back to any other installed version.
        .or_else(|| {
            installed
                .iter()
                .find(|s| Some(s.as_str()) != cur.as_deref())
                .cloned()
        });
    let target = target
        .ok_or_else(|| anyhow::anyhow!("no prior version of build `{name}` to roll back to"))?;
    flip_current(name, &target)?;
    Ok(RollbackOutcome {
        name: name.to_string(),
        from: cur,
        to: target,
    })
}

/// Remove versions beyond `retain` newest installs, always keeping the current
/// one. Ordering is the durable ledger (deterministic), not mtime.
pub fn gc(name: &str, retain: usize) -> Result<()> {
    let retain = retain.max(1);
    let cur = current_slug(name);
    let order = ledger_load(name);
    let installed: std::collections::BTreeSet<String> = installed_slugs(name).into_iter().collect();

    // Keep set: the `retain` newest ledger entries, plus current.
    let mut keep: std::collections::BTreeSet<String> = order
        .iter()
        .rev()
        .filter(|s| installed.contains(*s))
        .take(retain)
        .cloned()
        .collect();
    if let Some(c) = &cur {
        keep.insert(c.clone());
    }

    for s in &installed {
        if !keep.contains(s) {
            let _ = fs::remove_dir_all(name_dir(name).join(s));
        }
    }
    // Prune ledger entries whose dirs are gone (kept or not).
    let pruned: Vec<String> = order.into_iter().filter(|s| keep.contains(s)).collect();
    // Only rewrite if it changed (avoid churn).
    if pruned != ledger_load(name) {
        ledger_save(name, &pruned)?;
    }
    Ok(())
}

/// Status of every configured build plus any installed-but-unconfigured ones.
pub fn list(specs: &[BuildSpec]) -> Vec<BuildStatus> {
    let mut names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
    if let Ok(rd) = fs::read_dir(builds_root()) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                let n = e.file_name().to_string_lossy().into_owned();
                if !names.contains(&n) {
                    names.push(n);
                }
            }
        }
    }
    names
        .into_iter()
        .map(|name| {
            let cur = current_slug(&name);
            let mut versions: Vec<Version> = installed_slugs(&name)
                .into_iter()
                .map(|slug| Version {
                    current: Some(slug.as_str()) == cur.as_deref(),
                    slug,
                })
                .collect();
            // Newest-first by ledger order; unknowns sort last.
            let order = ledger_load(&name);
            let rank = |s: &str| {
                order
                    .iter()
                    .rev()
                    .position(|x| x == s)
                    .unwrap_or(usize::MAX)
            };
            versions.sort_by_key(|v| rank(&v.slug));
            BuildStatus {
                name: name.clone(),
                spec: spec(specs, &name).cloned(),
                current: cur,
                versions,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Profile resolution API - what a profile referencing `build = "<name>"` uses.
// Consumed when profiles gain a `build` field (the portability/flag-migration
// step); exposed now so that wiring is a one-line lookup, not new plumbing.
// ---------------------------------------------------------------------------

/// Absolute path to `bin` in build `name`'s current version, if installed.
pub fn current_bin(name: &str, bin: &str) -> Option<PathBuf> {
    let p = name_dir(name).join("current").join(bin);
    p.is_file().then_some(p)
}

/// Accepted names for the llama.cpp RPC worker binary, newest first. Upstream
/// renamed `rpc-server` -> `ggml-rpc-server` (the `ggml-` tool prefix); a build
/// pinned to an older commit still emits the old name, so every rpc lookup must
/// accept both. Resolution is name-order, so a current build wins.
pub const RPC_BIN_NAMES: &[&str] = &["ggml-rpc-server", "rpc-server"];

/// Does `vdir` hold an rpc worker binary under any accepted name?
fn has_rpc(vdir: &Path) -> bool {
    RPC_BIN_NAMES.iter().any(|b| vdir.join(b).is_file())
}

/// Absolute path to the rpc worker binary in build `name`'s current version,
/// under whichever accepted name it was built as. None if absent (or the build
/// isn't installed) - i.e. this node cannot serve as a cluster worker.
pub fn current_rpc_bin(name: &str) -> Option<PathBuf> {
    RPC_BIN_NAMES
        .iter()
        .find_map(|b| current_bin(name, b))
}

/// The version slug currently active for build `name` (e.g. for tagging a bench
/// run with the build that produced it). None if the build isn't installed.
pub fn current_version(name: &str) -> Option<String> {
    current_slug(name)
}

/// The current version dir of build `name` (for `LD_LIBRARY_PATH`), if installed.
pub fn current_dir(name: &str) -> Option<PathBuf> {
    let p = name_dir(name).join("current");
    // Resolve the symlink so the path is stable for systemd.
    fs::canonicalize(&p).ok().filter(|p| p.is_dir())
}

// ---------------------------------------------------------------------------
// RealBuilder: production git + cmake.
// ---------------------------------------------------------------------------

/// The real build seam: a persistent git checkout under the state dir, configured
/// and compiled with cmake; the resulting `build/bin/*` (executables + runtime
/// libs) are copied into the staging dir.
pub struct RealBuilder;

impl RealBuilder {
    pub fn new() -> Self {
        RealBuilder
    }
}

impl Default for RealBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder for RealBuilder {
    fn resolve_ref(&mut self, spec: &BuildSpec) -> Result<String> {
        // A full/abbreviated sha resolves to itself (no network round-trip).
        if is_hex(&spec.git_ref) && spec.git_ref.len() >= 7 {
            return Ok(spec.git_ref.to_lowercase());
        }
        let out = run_capture("git", &["ls-remote", &spec.git_url, &spec.git_ref], None)
            .with_context(|| format!("git ls-remote {} {}", spec.git_url, spec.git_ref))?;
        // First whitespace-separated token of the first line is the sha.
        let sha = out
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().next())
            .map(|s| s.to_string())
            .filter(|s| is_hex(s) && s.len() >= 7);
        sha.ok_or_else(|| {
            anyhow::anyhow!(
                "ref `{}` not found at {} (set a valid branch/tag/sha)",
                spec.git_ref,
                spec.git_url
            )
        })
    }

    fn produce(&mut self, spec: &BuildSpec, commit: &str, stage: &Path) -> Result<()> {
        // Fail FAST (before a 15-20 min compile) if the build toolchain or the
        // Vulkan/SPIR-V dev packages are missing - with the exact per-distro
        // install command. This is the only place a real compile happens, so a
        // no-op reinstall (Switched/AlreadyCurrent) never trips it.
        check_build_deps().ensure()?;

        let src = source_dir(&spec.name);
        // Clone (blobless, fast) or fetch the persistent source checkout.
        if src.join(".git").is_dir() {
            run_status(
                "git",
                &[
                    "-C",
                    &src.to_string_lossy(),
                    "fetch",
                    "--all",
                    "--tags",
                    "--prune",
                ],
                None,
            )
            .context("git fetch")?;
        } else {
            if let Some(parent) = src.parent() {
                fs::create_dir_all(parent)?;
            }
            run_status(
                "git",
                &[
                    "clone",
                    "--filter=blob:none",
                    &spec.git_url,
                    &src.to_string_lossy(),
                ],
                None,
            )
            .context("git clone")?;
        }
        run_status(
            "git",
            &[
                "-C",
                &src.to_string_lossy(),
                "checkout",
                "--force",
                "--detach",
                commit,
            ],
            None,
        )
        .with_context(|| format!("git checkout {commit}"))?;

        // Configure + build with cmake. A fresh build dir per name keeps versions
        // from contaminating each other; cmake caches incrementally across updates.
        let build_dir = src.join("build");
        let mut cfg: Vec<String> = vec![
            "-S".into(),
            src.to_string_lossy().into_owned(),
            "-B".into(),
            build_dir.to_string_lossy().into_owned(),
        ];
        cfg.extend(spec.cmake_flags.split_whitespace().map(|s| s.to_string()));
        run_status("cmake", &str_args(&cfg), None).context("cmake configure")?;
        // Cap parallelism to what RAM can feed: `ggml-vulkan.cpp` is one huge
        // translation unit (~3 GB resident), so a bare `-j` (all cores) sends a
        // 16 GB BC-250 into swap and looks hung. `build_jobs()` honors
        // CMAKE_BUILD_PARALLEL_LEVEL, else derives a memory-aware count.
        let jobs = build_jobs();
        run_status(
            "cmake",
            &[
                "--build",
                &build_dir.to_string_lossy(),
                "-j",
                &jobs.to_string(),
            ],
            None,
        )
        .context("cmake build")?;

        // Copy build/bin/* (executables + runtime .so) into the staging dir.
        let bin = build_dir.join("bin");
        copy_dir_files(&bin, stage)
            .with_context(|| format!("copying build output from {}", bin.display()))?;
        Ok(())
    }
}

/// Copy every regular file from `from` into `to` (flat; preserves perms). llama
/// .cpp emits its executables and shared libs side-by-side in `build/bin`.
fn copy_dir_files(from: &Path, to: &Path) -> Result<()> {
    let rd = fs::read_dir(from)
        .with_context(|| format!("reading build output dir {}", from.display()))?;
    fs::create_dir_all(to)?;
    let mut copied = 0u32;
    for e in rd.flatten() {
        let p = e.path();
        let meta = match fs::symlink_metadata(&p) {
            Ok(m) => m,
            Err(_) => continue,
        };
        // Follow symlinks to real files (cmake sometimes versions .so via symlink).
        if meta.file_type().is_dir() {
            continue;
        }
        if let Some(fname) = p.file_name() {
            let dst = to.join(fname);
            if fs::copy(&p, &dst).is_ok() {
                copied += 1;
            }
        }
    }
    if copied == 0 {
        bail!(
            "no files in {} - did the build produce anything?",
            from.display()
        );
    }
    Ok(())
}

fn str_args(v: &[String]) -> Vec<&str> {
    v.iter().map(|s| s.as_str()).collect()
}

/// Run a command, returning stdout; error carries a stderr tail on failure.
fn run_capture(cmd: &str, args: &[&str], cwd: Option<&Path>) -> Result<String> {
    let mut c = Command::new(cmd);
    c.args(args);
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    let out = c
        .output()
        .with_context(|| format!("spawning `{cmd}` (is it installed?)"))?;
    if !out.status.success() {
        let tail = tail_lines(&String::from_utf8_lossy(&out.stderr), 8);
        bail!(
            "`{cmd} {}` failed ({}):\n{tail}",
            args.join(" "),
            out.status
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Run a command for its exit status, streaming its output to the user's terminal
/// (so a long compile shows progress); error carries a short note on failure.
fn run_status(cmd: &str, args: &[&str], cwd: Option<&Path>) -> Result<()> {
    let mut c = Command::new(cmd);
    c.args(args);
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    let st = c
        .status()
        .with_context(|| format!("spawning `{cmd}` (is it installed?)"))?;
    if !st.success() {
        bail!("`{cmd} {}` failed ({st})", args.join(" "));
    }
    Ok(())
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

// ---------------------------------------------------------------------------
// Build parallelism - memory-aware, so a low-RAM BC-250 doesn't swap to death.
// ---------------------------------------------------------------------------

/// Total system RAM in GiB (best-effort from /proc/meminfo; 0 if unreadable).
fn mem_total_gib() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("MemTotal:")
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|kb| kb.parse::<u64>().ok())
            })
        })
        .map(|kb| kb / (1024 * 1024))
        .unwrap_or(0)
}

/// How many parallel compile jobs to run.
///
/// Precedence: an explicit `CMAKE_BUILD_PARALLEL_LEVEL` wins (operator override);
/// otherwise `min(cpus, max(1, RAM_GiB / 3))`. The `/3` reflects that the heavy
/// Vulkan TUs need ~3 GiB resident each, so a 16 GiB / 6-core BC-250 caps at ~5
/// and never thrashes. A bare `-j` would have used all 6 and swapped.
pub fn build_jobs() -> usize {
    if let Some(n) = std::env::var("CMAKE_BUILD_PARALLEL_LEVEL")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
    {
        return n;
    }
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let by_mem = (mem_total_gib() / 3).max(1) as usize;
    // If meminfo was unreadable (0 -> by_mem 1) don't over-throttle a real box:
    // fall back to cpus when we have no memory signal.
    let by_mem = if mem_total_gib() == 0 { cpus } else { by_mem };
    cpus.min(by_mem).max(1)
}

// ---------------------------------------------------------------------------
// Build-dependency preflight - fail in 2s with the exact install command,
// not 20 min into a compile on a missing header. Shared with `llmtune doctor`.
// ---------------------------------------------------------------------------

/// What the Vulkan llama.cpp build needs, and how to get it on this distro.
#[derive(Debug, Clone)]
pub struct DepReport {
    /// Reliably-detected missing items (PATH tools + the canonical Vulkan
    /// header). Non-empty => the compile WILL fail; `ensure()` bails.
    pub missing: Vec<String>,
    /// The full, copy-pasteable install command for the detected distro (always
    /// the complete recommended package set, so heuristics we can't detect - the
    /// ICD loader, SPIR-V headers - are still covered).
    pub install_cmd: String,
}

impl DepReport {
    /// Bail with the install command if anything reliably-detected is missing.
    pub fn ensure(&self) -> Result<()> {
        if self.missing.is_empty() {
            return Ok(());
        }
        bail!(
            "build toolchain incomplete - missing {}.\n  {}",
            self.missing.join(", "),
            self.install_cmd
        )
    }
}

fn dep_in_path(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .map(|path| {
            std::env::split_paths(&path).any(|dir| {
                let p = dir.join(cmd);
                p.is_file() || p.is_symlink()
            })
        })
        .unwrap_or(false)
}

fn distro_id() -> String {
    fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("ID=")
                    .map(|v| v.trim_matches('"').to_string())
            })
        })
        .unwrap_or_default()
}

/// The complete per-distro install command for the Vulkan build toolchain.
/// Lists ALL required packages (compiler, cmake, git, Vulkan headers + ICD
/// loader, SPIR-V headers, shader compiler) so a working-but-drifted board that
/// lost a package to a distro update gets one line to fix it.
fn install_cmd() -> String {
    match distro_id().as_str() {
        "arch" | "cachyos" | "endeavouros" | "manjaro" | "garuda" | "artix" => {
            "install: sudo pacman -S --needed base-devel cmake git vulkan-headers \
             vulkan-icd-loader spirv-headers shaderc"
                .into()
        }
        "debian" | "ubuntu" | "pop" | "linuxmint" | "raspbian" => {
            "install: sudo apt install build-essential cmake git libvulkan-dev \
             glslc spirv-headers"
                .into()
        }
        "fedora" | "rhel" | "centos" | "rocky" | "almalinux" => {
            "install: sudo dnf install gcc-c++ cmake git vulkan-headers \
             vulkan-loader-devel glslc spirv-headers"
                .into()
        }
        _ => "install the Vulkan build toolchain for your distro: a C++ compiler, \
              cmake, git, Vulkan headers + ICD loader, SPIR-V headers, and a \
              shader compiler (glslc/shaderc)"
            .into(),
    }
}

/// True if any of `candidates` exists on disk (a header dir/file or a library
/// under one of the common multiarch libdirs).
fn any_path_exists(candidates: &[&str]) -> bool {
    candidates.iter().any(|p| Path::new(p).exists())
}

/// Preflight the build toolchain. Every check here is a RELIABLE, path-based
/// signal of a package the Vulkan `llama.cpp` build genuinely needs, so a `miss`
/// means the compile *will* fail - we just turn a 20-min-then-fail into a 2-s
/// bail with the exact install command. The install command still lists the full
/// set, so a distro whose layout we don't probe is covered too.
pub fn check_build_deps() -> DepReport {
    let mut missing: Vec<String> = Vec::new();
    if !dep_in_path("git") {
        missing.push("git".into());
    }
    if !dep_in_path("cmake") {
        missing.push("cmake".into());
    }
    if !["c++", "g++", "clang++"].iter().any(|c| dep_in_path(c)) {
        missing.push("a C++ compiler (g++/clang++)".into());
    }
    if !["glslc", "glslangValidator"].iter().any(|c| dep_in_path(c)) {
        missing.push("glslc/glslangValidator (shaderc)".into());
    }
    // The canonical Vulkan header path is identical across distros, so its
    // absence is a reliable signal (this is what a drifted board typically lost).
    if !Path::new("/usr/include/vulkan/vulkan.h").exists() {
        missing.push("Vulkan headers (vulkan/vulkan.h)".into());
    }
    // Vulkan ICD loader dev lib - GGML_VULKAN links against it; cmake's
    // find_package(Vulkan) fails at configure without it.
    if !any_path_exists(&[
        "/usr/lib/libvulkan.so",
        "/usr/lib64/libvulkan.so",
        "/usr/lib/x86_64-linux-gnu/libvulkan.so",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so",
    ]) {
        missing.push("Vulkan ICD loader (libvulkan.so / vulkan-icd-loader)".into());
    }
    // SPIR-V headers - the shader-gen step needs them, and this is the miss that
    // fails LATE (deep in the compile), so detecting it up front is the whole
    // point. Header layout is `spirv/unified1/spirv.h` under the include root.
    if !any_path_exists(&[
        "/usr/include/spirv/unified1/spirv.h",
        "/usr/include/spirv/spirv.h",
        "/usr/include/spirv",
    ]) {
        missing.push("SPIR-V headers (spirv/, spirv-headers)".into());
    }
    DepReport {
        missing,
        install_cmd: install_cmd(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Each test gets a private state dir via LLMTUNE_STATE_DIR. The env var is
    // process-global, so tests that set it are serialized through this mutex.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_state() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("llmtune-build-{}-{}", std::process::id(), n))
    }

    fn spec(name: &str) -> BuildSpec {
        BuildSpec {
            name: name.to_string(),
            git_url: "https://example.invalid/llama.cpp".into(),
            git_ref: "master".into(),
            cmake_flags: "-DGGML_VULKAN=ON".into(),
            // Required set only; the rpc worker is optional (see rpc_present).
            out_bins: vec!["llama-server".into(), "llama-cli".into()],
        }
    }

    /// A builder that fabricates the requested binaries from a scripted commit.
    struct MockBuilder {
        commit: String,
        produce_calls: Cell<u32>,
        /// If true, produce writes NO binaries (simulates a broken build).
        produce_empty: bool,
    }
    impl MockBuilder {
        fn new(commit: &str) -> Self {
            MockBuilder {
                commit: commit.to_string(),
                produce_calls: Cell::new(0),
                produce_empty: false,
            }
        }
    }
    impl Builder for MockBuilder {
        fn resolve_ref(&mut self, _spec: &BuildSpec) -> Result<String> {
            Ok(self.commit.clone())
        }
        fn produce(&mut self, spec: &BuildSpec, _commit: &str, stage: &Path) -> Result<()> {
            self.produce_calls.set(self.produce_calls.get() + 1);
            if self.produce_empty {
                return Ok(());
            }
            for b in &spec.out_bins {
                fs::write(stage.join(b), b"#!fake\n").unwrap();
            }
            // The rpc worker binary lands under its current upstream name; it is
            // copied wholesale (not required by out_bins) and drives rpc_present.
            fs::write(stage.join("ggml-rpc-server"), b"#!fake\n").unwrap();
            // a runtime lib alongside, to mimic real output
            fs::write(stage.join("libggml.so"), b"lib").unwrap();
            Ok(())
        }
    }

    /// Run `f` with a fresh private state dir bound; cleaned up after.
    ///
    /// Panic-safe: if `f` panics, the env var and temp dir are still restored (via
    /// the RAII guard) and the mutex is not left poisoned (we recover the guard),
    /// so a single failing test cannot cascade into every later one.
    fn with_state<T>(f: impl FnOnce() -> T) -> T {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = temp_state();
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        std::env::set_var("LLMTUNE_STATE_DIR", &dir);
        // Restore on the way out even if `f` unwinds.
        struct Restore(PathBuf);
        impl Drop for Restore {
            fn drop(&mut self) {
                std::env::remove_var("LLMTUNE_STATE_DIR");
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _restore = Restore(dir);
        f()
    }

    #[test]
    fn seed_parses_and_validates() {
        let specs = parse(SEED).unwrap();
        validate(&specs).unwrap();
        assert!(specs.iter().any(|s| s.name == "vulkan"));
        let v = specs.iter().find(|s| s.name == "vulkan").unwrap();
        assert!(v.out_bins.contains(&"llama-server".to_string()));
        assert!(v.cmake_flags.contains("GGML_VULKAN=ON"));
    }

    #[test]
    fn validate_rejects_bad_recipes() {
        let dup = vec![spec("a"), spec("a")];
        assert!(validate(&dup).is_err());
        let mut bad = spec("../escape");
        assert!(validate(std::slice::from_ref(&bad)).is_err());
        bad = spec("current");
        assert!(validate(std::slice::from_ref(&bad)).is_err());
        let mut nobins = spec("x");
        nobins.out_bins.clear();
        assert!(validate(&[nobins]).is_err());
        // git url/ref injection footguns
        let mut badurl = spec("y");
        badurl.git_url = "ext::sh -c evil".into();
        assert!(validate(std::slice::from_ref(&badurl)).is_err());
        badurl.git_url = "--upload-pack=evil".into();
        assert!(validate(std::slice::from_ref(&badurl)).is_err());
        let mut badref = spec("z");
        badref.git_ref = "--upload-pack=evil".into();
        assert!(validate(&[badref]).is_err());
    }

    #[test]
    fn slug_is_short_hex_or_safe() {
        assert_eq!(slug("a1b2c3d4e5f6a7b8"), "a1b2c3d4e5f6");
        assert_eq!(slug("master"), "master");
        assert_eq!(slug("release/v1.2"), "release-v1.2");
    }

    #[test]
    fn install_builds_and_sets_current() {
        with_state(|| {
            let s = spec("vulkan");
            let mut b = MockBuilder::new("deadbeefcafe0001");
            let out = install(&s, &mut b, DEFAULT_RETAIN).unwrap();
            assert_eq!(out.action, Action::Built);
            assert_eq!(out.slug, "deadbeefcafe");
            assert!(out.dir.join("llama-server").is_file());
            assert!(out.dir.join("llama-cli").is_file());
            // rpc worker copied under its upstream name and reported present.
            assert!(out.rpc_present);
            assert!(current_rpc_bin("vulkan").is_some());
            assert_eq!(current_slug("vulkan").as_deref(), Some("deadbeefcafe"));
            // resolution API points at the live binary + dir
            assert!(current_bin("vulkan", "llama-server").is_some());
            assert!(current_dir("vulkan").is_some());
            assert!(current_bin("vulkan", "nope").is_none());
        });
    }

    #[test]
    fn install_is_idempotent_no_rebuild() {
        with_state(|| {
            let s = spec("vulkan");
            let mut b = MockBuilder::new("deadbeefcafe0001");
            install(&s, &mut b, DEFAULT_RETAIN).unwrap();
            let out = install(&s, &mut b, DEFAULT_RETAIN).unwrap();
            assert_eq!(out.action, Action::AlreadyCurrent);
            assert_eq!(
                b.produce_calls.get(),
                1,
                "must not rebuild a current commit"
            );
        });
    }

    #[test]
    fn update_to_new_commit_keeps_old_for_rollback() {
        with_state(|| {
            let s = spec("vulkan");
            let mut b1 = MockBuilder::new("1111111111110000");
            let o1 = install(&s, &mut b1, DEFAULT_RETAIN).unwrap();
            let mut b2 = MockBuilder::new("2222222222220000");
            let o2 = install(&s, &mut b2, DEFAULT_RETAIN).unwrap();
            assert_eq!(o2.action, Action::Built);
            assert_eq!(o2.previous.as_deref(), Some(o1.slug.as_str()));
            assert_eq!(current_slug("vulkan").as_deref(), Some("222222222222"));
            // both versions still on disk
            assert!(name_dir("vulkan").join("111111111111").is_dir());
            assert!(name_dir("vulkan").join("222222222222").is_dir());
        });
    }

    #[test]
    fn reinstall_existing_version_switches_without_rebuild() {
        with_state(|| {
            let s = spec("vulkan");
            let mut a = MockBuilder::new("aaaaaaaaaaaa0000");
            install(&s, &mut a, DEFAULT_RETAIN).unwrap();
            let mut b = MockBuilder::new("bbbbbbbbbbbb0000");
            install(&s, &mut b, DEFAULT_RETAIN).unwrap();
            // go back to A's commit: dir exists -> Switched, no produce call.
            let mut a2 = MockBuilder::new("aaaaaaaaaaaa0000");
            let out = install(&s, &mut a2, DEFAULT_RETAIN).unwrap();
            assert_eq!(out.action, Action::Switched);
            assert_eq!(a2.produce_calls.get(), 0);
            assert_eq!(current_slug("vulkan").as_deref(), Some("aaaaaaaaaaaa"));
        });
    }

    #[test]
    fn rollback_flips_to_previous_then_errors_when_alone() {
        with_state(|| {
            let s = spec("vulkan");
            let mut b1 = MockBuilder::new("1111111111110000");
            install(&s, &mut b1, DEFAULT_RETAIN).unwrap();
            let mut b2 = MockBuilder::new("2222222222220000");
            install(&s, &mut b2, DEFAULT_RETAIN).unwrap();
            let r = rollback("vulkan", &[]).unwrap();
            assert_eq!(r.from.as_deref(), Some("222222222222"));
            assert_eq!(r.to, "111111111111");
            assert_eq!(current_slug("vulkan").as_deref(), Some("111111111111"));
        });
    }

    #[test]
    fn rollback_with_single_version_errors() {
        with_state(|| {
            let s = spec("vulkan");
            let mut b = MockBuilder::new("1111111111110000");
            install(&s, &mut b, DEFAULT_RETAIN).unwrap();
            assert!(rollback("vulkan", &[]).is_err());
        });
    }

    #[test]
    fn gc_retains_n_newest_plus_current() {
        with_state(|| {
            let s = spec("vulkan");
            // install 4 distinct commits with retain=2
            for i in 0..4u32 {
                let commit = format!("{:012x}0000", i + 1);
                let mut b = MockBuilder::new(&commit);
                install(&s, &mut b, 2).unwrap();
            }
            let remaining = installed_slugs("vulkan");
            // retain=2 newest; current is newest so it's within the 2 -> exactly 2 dirs
            assert_eq!(remaining.len(), 2, "got {remaining:?}");
            // the two newest commits survive
            assert!(name_dir("vulkan").join(slug("0000000000040000")).is_dir());
            assert!(name_dir("vulkan").join(slug("0000000000030000")).is_dir());
            assert!(!name_dir("vulkan").join(slug("0000000000010000")).is_dir());
        });
    }

    #[test]
    fn broken_build_leaves_no_current_and_cleans_stage() {
        with_state(|| {
            let s = spec("vulkan");
            let mut b = MockBuilder::new("deadbeefcafe0001");
            b.produce_empty = true;
            let res = install(&s, &mut b, DEFAULT_RETAIN);
            assert!(res.is_err(), "missing binaries must fail the install");
            assert!(current_slug("vulkan").is_none());
            // no staging dir left behind
            let stage_leftovers = fs::read_dir(name_dir("vulkan"))
                .map(|rd| {
                    rd.flatten()
                        .filter(|e| e.file_name().to_string_lossy().starts_with(".stage"))
                        .count()
                })
                .unwrap_or(0);
            assert_eq!(stage_leftovers, 0, "staging dir must be cleaned up");
        });
    }

    #[test]
    fn build_jobs_honors_explicit_env_then_derives() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("CMAKE_BUILD_PARALLEL_LEVEL", "2");
        assert_eq!(build_jobs(), 2, "explicit override must win");
        std::env::set_var("CMAKE_BUILD_PARALLEL_LEVEL", "0"); // invalid -> ignored
        assert!(build_jobs() >= 1, "invalid override falls through to derived");
        std::env::remove_var("CMAKE_BUILD_PARALLEL_LEVEL");
        assert!(build_jobs() >= 1, "derived count is always positive");
    }

    #[test]
    fn rpc_bin_accepts_either_upstream_name() {
        let d = std::env::temp_dir().join(format!("llmtune-rpc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        assert!(!has_rpc(&d), "no rpc binary yet");
        // The pre-rename name is still accepted (older pinned builds emit it).
        fs::write(d.join("rpc-server"), b"x").unwrap();
        assert!(has_rpc(&d));
        fs::remove_file(d.join("rpc-server")).unwrap();
        fs::write(d.join("ggml-rpc-server"), b"x").unwrap();
        assert!(has_rpc(&d), "current upstream name accepted");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn list_reports_versions_and_current() {
        with_state(|| {
            let specs = vec![spec("vulkan")];
            let mut b1 = MockBuilder::new("1111111111110000");
            install(&specs[0], &mut b1, DEFAULT_RETAIN).unwrap();
            let mut b2 = MockBuilder::new("2222222222220000");
            install(&specs[0], &mut b2, DEFAULT_RETAIN).unwrap();
            let statuses = list(&specs);
            let v = statuses.iter().find(|s| s.name == "vulkan").unwrap();
            assert_eq!(v.current.as_deref(), Some("222222222222"));
            assert_eq!(v.versions.len(), 2);
            // newest first
            assert_eq!(v.versions[0].slug, "222222222222");
            assert!(v.versions[0].current);
            assert!(v.spec.is_some());
        });
    }
}
