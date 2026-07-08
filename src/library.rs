// SPDX-License-Identifier: GPL-2.0-only
//! The control-host model library (fleet M5): the `[netboot] models_dir` that
//! `netboot up` exports read-only over NFS. Nodes automount it, so anything
//! added here is visible to every node on next access - no remount, no
//! node-side action. That same property is why `add` must be ATOMIC (a node's
//! automount must never see a half-written .gguf) and why `rm` guards against
//! deleting a model a node is currently serving (the server holds it mmapped
//! over NFS; unlinking it under the mount is exactly the failure the guard
//! exists to prevent).
//!
//! Mechanics:
//! - `add` stages into a dot-prefixed `.part` temp file IN the library dir
//!   (same filesystem, so the final `rename` is atomic) and validates the GGUF
//!   magic before committing. Discovery - ours and the nodes' - only considers
//!   `*.gguf`, so the temp file is invisible until the rename.
//! - `rm` queries every configured fleet node's status (the same
//!   transport/ssh path `node served` uses) and refuses while any reachable
//!   node reports the model as served, unless forced.
//!
//! Pure decision logic (name derivation, the rm guard) is separated from IO so
//! it unit-tests without a fleet.

use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::{model, transport};

// ---------------------------------------------------------------------------
// Library location
// ---------------------------------------------------------------------------

/// Resolve the library directory: the `[netboot] models_dir` (the NFS export)
/// when netboot is configured, else the local node's models_dir so `models`
/// still manages the single-box library in local mode.
pub fn dir(cfg: &Config) -> PathBuf {
    match &cfg.netboot {
        Some(nb) => PathBuf::from(&nb.models_dir),
        None => PathBuf::from(&cfg.local_node().models_dir),
    }
}

/// List the library's models - the same discovery (`*.gguf`, chat models,
/// header-parsed facts) a node runs over its automount, so `models list` and a
/// node's `node list` agree on the same directory.
pub fn list(dir: &Path) -> Result<Vec<model::Model>> {
    model::discover(dir)
}

// ---------------------------------------------------------------------------
// GGUF validation
// ---------------------------------------------------------------------------

/// True if the file starts with the GGUF magic. The cheap commit gate for
/// `add`: a truncated download, an HTML error page, or a mispointed path all
/// fail here before anything lands in the export.
pub fn is_gguf(path: &Path) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && &magic == b"GGUF"
}

// ---------------------------------------------------------------------------
// Name derivation (pure)
// ---------------------------------------------------------------------------

/// Derive the library file name from an `add` source (local path or URL).
/// For URLs the query/fragment is stripped first. The name must be a plain
/// `*.gguf` basename - no path separators, not hidden, nothing a shell or the
/// exports file would misread.
pub fn dest_name(source: &str) -> Result<String> {
    let base = if is_url(source) {
        let no_q = source.split(['?', '#']).next().unwrap_or(source);
        no_q.rsplit('/').next().unwrap_or("").to_string()
    } else {
        Path::new(source)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    if base.is_empty() {
        bail!("cannot derive a file name from `{source}`");
    }
    if !base.to_lowercase().ends_with(".gguf") {
        bail!("`{base}` is not a .gguf file - the library only holds GGUF models");
    }
    if base.starts_with('.') || base.starts_with('-') {
        bail!("refusing file name `{base}` (hidden/option-like)");
    }
    if base
        .chars()
        .any(|c| c.is_control() || matches!(c, '/' | '\\' | '\'' | '"' | '`' | '$'))
    {
        bail!("refusing file name `{base}` (control or shell-special characters)");
    }
    Ok(base)
}

/// True if the source is a URL we download (vs a local path we copy).
pub fn is_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

// ---------------------------------------------------------------------------
// Atomic add
// ---------------------------------------------------------------------------

/// The staging path for `name`: dot-prefixed and `.part`-suffixed in the SAME
/// directory (same filesystem → `rename` is atomic), and not `*.gguf`, so no
/// discovery - control host or node - ever lists it.
pub fn temp_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!(".llmtune-add-{name}.{}.part", std::process::id()))
}

/// Removes the staging file on drop unless the add committed. Covers every
/// early-return AND a panic mid-copy, so a failed add never leaves a `.part`
/// turd in the export.
struct TempGuard {
    path: PathBuf,
    committed: bool,
}

impl Drop for TempGuard {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// What a successful `add` reports.
#[derive(Debug)]
pub struct AddOutcome {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

/// Validate the staged temp file and atomically rename it into place.
fn commit(temp: &mut TempGuard, dest: &Path, name: &str) -> Result<AddOutcome> {
    if !is_gguf(&temp.path) {
        bail!(
            "`{name}` is not a valid GGUF (bad magic) - refusing to add it to the library. \
             If this was a download, the server may have returned an error page."
        );
    }
    let size_bytes = fs::metadata(&temp.path)?.len();
    fs::rename(&temp.path, dest)
        .with_context(|| format!("committing {} into the library", dest.display()))?;
    temp.committed = true;
    Ok(AddOutcome {
        name: name.to_string(),
        path: dest.to_path_buf(),
        size_bytes,
    })
}

/// Common preamble for both add paths: ensure the dir exists, refuse a
/// duplicate, and hand back the (dest, staged-temp) pair.
fn stage(dir: &Path, name: &str) -> Result<(PathBuf, TempGuard)> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let dest = dir.join(name);
    if dest.exists() {
        bail!(
            "`{name}` already exists in the library - `llmtune models rm {name}` first \
             if you mean to replace it"
        );
    }
    Ok((
        dest,
        TempGuard {
            path: temp_path(dir, name),
            committed: false,
        },
    ))
}

/// Copy a local file into the library atomically (validate → temp → rename).
pub fn add_local(dir: &Path, src: &Path, name: &str) -> Result<AddOutcome> {
    if !src.is_file() {
        bail!("`{}` is not a file", src.display());
    }
    let (dest, mut temp) = stage(dir, name)?;
    fs::copy(src, &temp.path).with_context(|| {
        format!(
            "copying {} into the library at {}",
            src.display(),
            dir.display()
        )
    })?;
    commit(&mut temp, &dest, name)
}

/// Hard ceiling on a single download. No real GGUF approaches this (the
/// largest multi-node RPC models are well under it); an adversarial or
/// mispointed URL must not fill the shared NFS export.
const MAX_DOWNLOAD_BYTES: u64 = 128 * (1u64 << 30);

/// Per-read stall timeout: a server that stops sending bytes for this long
/// aborts the download instead of hanging `models add` forever. (A server
/// that keeps trickling ANY bytes keeps the transfer alive - the size cap
/// still bounds the damage.)
const READ_STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Stream `reader` into `out`, erroring once more than `max` bytes arrive.
/// Split out (with the limit injected) so the cap is unit-testable.
fn copy_capped(reader: &mut dyn Read, out: &mut File, max: u64) -> Result<u64> {
    let mut written = 0u64;
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            return Ok(written);
        }
        written += n as u64;
        if written > max {
            bail!("download exceeded the {} GiB size cap", max >> 30);
        }
        std::io::Write::write_all(out, &chunk[..n])?;
    }
}

/// Download a URL into the library atomically. Plain http(s) GET, streamed to
/// the staging file; the GGUF gate runs on the complete download. Bounded:
/// a stalled server times out ([`READ_STALL_TIMEOUT`]) and the stream is
/// size-capped ([`MAX_DOWNLOAD_BYTES`]) - a bad URL can neither hang the
/// command forever nor fill the export's disk.
pub fn add_url(dir: &Path, url: &str, name: &str) -> Result<AddOutcome> {
    let (dest, mut temp) = stage(dir, name)?;
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(READ_STALL_TIMEOUT)
        .redirects(8)
        .build();
    let resp = agent
        .get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    // Preflight: refuse an advertised size over the cap before moving bytes.
    if let Some(len) = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
    {
        if len > MAX_DOWNLOAD_BYTES {
            bail!(
                "server advertises {} GiB, over the {} GiB download cap",
                len >> 30,
                MAX_DOWNLOAD_BYTES >> 30
            );
        }
    }
    let mut reader = resp.into_reader();
    let mut out = File::create(&temp.path)
        .with_context(|| format!("creating staging file in {}", dir.display()))?;
    copy_capped(&mut reader, &mut out, MAX_DOWNLOAD_BYTES)
        .with_context(|| format!("downloading {url}"))?;
    out.sync_all().ok();
    drop(out);
    commit(&mut temp, &dest, name)
}

// ---------------------------------------------------------------------------
// rm + the served-guard
// ---------------------------------------------------------------------------

/// One node's answer to "what are you serving?" - reachability kept separate
/// from the served name so the guard can distinguish "confirmed not serving"
/// from "couldn't ask".
pub struct NodeServed {
    pub node: String,
    pub reachable: bool,
    pub served: Option<String>,
}

/// Ask every configured fleet node what it serves, over the same transport
/// (local or ssh) `node served` / `fleet status` use.
pub fn served_across_fleet(cfg: &Config) -> Vec<NodeServed> {
    cfg.nodes
        .iter()
        .map(|n| match transport::for_node(n) {
            Ok(t) => {
                let s = t.status();
                NodeServed {
                    node: n.name.clone(),
                    reachable: s.reachable,
                    served: s.served,
                }
            }
            Err(_) => NodeServed {
                node: n.name.clone(),
                reachable: false,
                served: None,
            },
        })
        .collect()
}

/// The guard's verdict on removing `target`.
pub struct RmGuard {
    /// Nodes CONFIRMED serving the target - these block the rm.
    pub serving: Vec<String>,
    /// Nodes we couldn't ask (unreachable) - surfaced as a warning, not a block:
    /// an unreachable node is typically off, and blocking every rm while any
    /// board is powered down would make the guard unusable.
    pub unknown: Vec<String>,
}

/// Pure guard decision over the fleet's served reports.
pub fn rm_guard(target: &str, reports: &[NodeServed]) -> RmGuard {
    let mut serving = Vec::new();
    let mut unknown = Vec::new();
    for r in reports {
        if !r.reachable {
            unknown.push(r.node.clone());
        } else if r.served.as_deref() == Some(target) {
            serving.push(r.node.clone());
        }
    }
    RmGuard { serving, unknown }
}

/// The block message, if the guard blocks (serving nodes present and not
/// forced). Pure, so refusal-vs-force is unit-testable.
pub fn rm_block_message(target: &str, guard: &RmGuard, force: bool) -> Option<String> {
    if force || guard.serving.is_empty() {
        return None;
    }
    Some(format!(
        "`{target}` is currently SERVED on: {} - removing it would yank the file out from \
         under a live llama-server (it is mmapped over NFS). Swap those nodes to another \
         model first (`llmtune fleet swap-all <other>`, or drill into the node in the TUI), \
         or pass --force to remove anyway.",
        guard.serving.join(", ")
    ))
}

/// Resolve an rm target against the library: exact file name first, else a
/// unique substring match over ALL `*.gguf` files (not just chat models, so a
/// stray non-chat GGUF can still be removed).
pub fn resolve_rm(dir: &Path, query: &str) -> Result<String> {
    let q = query.trim();
    if q.is_empty() {
        bail!("provide a model file name (or unique substring) to remove");
    }
    let mut names: Vec<String> = Vec::new();
    let rd = fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    for ent in rd.flatten() {
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) == Some("gguf") && p.is_file() {
            if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
                names.push(n.to_string());
            }
        }
    }
    if names.iter().any(|n| n == q) {
        return Ok(q.to_string());
    }
    let ql = q.to_lowercase();
    let matches: Vec<&String> = names
        .iter()
        .filter(|n| n.to_lowercase().contains(&ql))
        .collect();
    match matches.len() {
        0 => bail!("no model matches `{query}` in {}", dir.display()),
        1 => Ok(matches[0].clone()),
        _ => {
            let list: Vec<&str> = matches.iter().take(8).map(|s| s.as_str()).collect();
            bail!("ambiguous `{query}` - matches: {}", list.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "llmtune-lib-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// A minimal structurally-valid GGUF: magic + version + 0 tensors + 0 KVs.
    fn minimal_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        let mut f = File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    // -- dest_name ----------------------------------------------------------

    #[test]
    fn dest_name_from_path_and_url() {
        assert_eq!(
            dest_name("/some/dir/Model-7B-Q4_K_M.gguf").unwrap(),
            "Model-7B-Q4_K_M.gguf"
        );
        assert_eq!(
            dest_name("https://host/repo/resolve/main/Model-7B-Q4_K_M.gguf?download=true").unwrap(),
            "Model-7B-Q4_K_M.gguf",
            "URL query string is stripped"
        );
        assert_eq!(
            dest_name("http://host/a/b/M.gguf#frag").unwrap(),
            "M.gguf",
            "URL fragment is stripped"
        );
    }

    #[test]
    fn dest_name_rejects_non_gguf_hidden_and_empty() {
        assert!(dest_name("/dir/model.bin").is_err(), "not .gguf");
        assert!(dest_name("https://host/dir/").is_err(), "no basename");
        assert!(dest_name("/dir/.hidden.gguf").is_err(), "hidden");
        assert!(
            dest_name("https://h/a$b.gguf").is_err(),
            "shell-special char"
        );
    }

    // -- GGUF magic validation ----------------------------------------------

    #[test]
    fn is_gguf_accepts_magic_rejects_other() {
        let d = tmpdir("magic");
        let good = d.join("good.gguf");
        write_file(&good, &minimal_gguf());
        assert!(is_gguf(&good));

        let bad = d.join("bad.gguf");
        write_file(&bad, b"<html>404 not found</html>");
        assert!(!is_gguf(&bad), "an HTML error page is not a GGUF");

        let short = d.join("short.gguf");
        write_file(&short, b"GG");
        assert!(!is_gguf(&short), "truncated before the magic completes");

        assert!(!is_gguf(&d.join("missing.gguf")));
        fs::remove_dir_all(&d).ok();
    }

    // -- atomic add ----------------------------------------------------------

    #[test]
    fn add_local_commits_via_temp_rename_and_cleans_up() {
        let lib = tmpdir("add-ok");
        let srcd = tmpdir("add-src");
        let src = srcd.join("TestModel-7B-Q4_K_M.gguf");
        write_file(&src, &minimal_gguf());

        let out = add_local(&lib, &src, "TestModel-7B-Q4_K_M.gguf").unwrap();
        assert!(out.path.is_file(), "final file exists");
        assert_eq!(out.size_bytes, minimal_gguf().len() as u64);
        // No staging residue: the only entry is the committed .gguf.
        let leftovers: Vec<String> = fs::read_dir(&lib)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".part"))
            .collect();
        assert!(leftovers.is_empty(), "no .part left: {leftovers:?}");
        fs::remove_dir_all(&lib).ok();
        fs::remove_dir_all(&srcd).ok();
    }

    #[test]
    fn add_local_rejects_bad_magic_and_leaves_nothing() {
        let lib = tmpdir("add-bad");
        let srcd = tmpdir("add-bad-src");
        let src = srcd.join("NotAModel-7B.gguf");
        write_file(&src, b"this is not a gguf");

        let err = add_local(&lib, &src, "NotAModel-7B.gguf").unwrap_err();
        assert!(err.to_string().contains("not a valid GGUF"), "{err}");
        // Neither the final file nor the temp survives a failed add.
        let entries: Vec<_> = fs::read_dir(&lib).unwrap().flatten().collect();
        assert!(entries.is_empty(), "library stays empty on failure");
        fs::remove_dir_all(&lib).ok();
        fs::remove_dir_all(&srcd).ok();
    }

    #[test]
    fn add_refuses_overwrite() {
        let lib = tmpdir("add-dup");
        let srcd = tmpdir("add-dup-src");
        let src = srcd.join("Dup-7B.gguf");
        write_file(&src, &minimal_gguf());
        add_local(&lib, &src, "Dup-7B.gguf").unwrap();
        let err = add_local(&lib, &src, "Dup-7B.gguf").unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        fs::remove_dir_all(&lib).ok();
        fs::remove_dir_all(&srcd).ok();
    }

    #[test]
    fn copy_capped_enforces_the_injected_limit() {
        let d = tmpdir("cap");
        // Under the cap: copied whole.
        let mut src: &[u8] = b"0123456789";
        let mut out = File::create(d.join("ok.part")).unwrap();
        assert_eq!(copy_capped(&mut src, &mut out, 10).unwrap(), 10);
        // One byte over: errors, names the cap.
        let big = vec![0u8; 4096];
        let mut src: &[u8] = &big;
        let mut out = File::create(d.join("big.part")).unwrap();
        let err = copy_capped(&mut src, &mut out, 4095).unwrap_err();
        assert!(err.to_string().contains("size cap"), "{err}");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn temp_path_is_not_discoverable() {
        let t = temp_path(Path::new("/lib"), "M-7B.gguf");
        let name = t.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with('.'), "dot-prefixed");
        assert!(!name.ends_with(".gguf"), "never matches *.gguf discovery");
    }

    // -- rm guard -------------------------------------------------------------

    fn rep(node: &str, reachable: bool, served: Option<&str>) -> NodeServed {
        NodeServed {
            node: node.to_string(),
            reachable,
            served: served.map(|s| s.to_string()),
        }
    }

    #[test]
    fn rm_guard_blocks_on_confirmed_serving_nodes() {
        let g = rm_guard(
            "A-7B.gguf",
            &[
                rep("bc250-a", true, Some("A-7B.gguf")),
                rep("bc250-b", true, Some("B-27B.gguf")),
                rep("bc250-c", true, Some("A-7B.gguf")),
            ],
        );
        assert_eq!(g.serving, vec!["bc250-a", "bc250-c"], "multi-node block");
        assert!(g.unknown.is_empty());
        let msg = rm_block_message("A-7B.gguf", &g, false).expect("blocks");
        assert!(msg.contains("bc250-a") && msg.contains("bc250-c"), "{msg}");
    }

    #[test]
    fn rm_guard_passes_when_nothing_serves_it() {
        let g = rm_guard(
            "A-7B.gguf",
            &[
                rep("bc250-a", true, Some("B-27B.gguf")),
                rep("bc250-b", true, None),
            ],
        );
        assert!(g.serving.is_empty());
        assert!(rm_block_message("A-7B.gguf", &g, false).is_none());
    }

    #[test]
    fn rm_guard_unreachable_warns_but_does_not_block() {
        let g = rm_guard(
            "A-7B.gguf",
            &[rep("bc250-a", false, None), rep("bc250-b", true, None)],
        );
        assert_eq!(g.unknown, vec!["bc250-a"]);
        assert!(rm_block_message("A-7B.gguf", &g, false).is_none());
    }

    #[test]
    fn rm_force_overrides_the_block() {
        let g = rm_guard("A-7B.gguf", &[rep("bc250-a", true, Some("A-7B.gguf"))]);
        assert!(
            rm_block_message("A-7B.gguf", &g, false).is_some(),
            "refused"
        );
        assert!(rm_block_message("A-7B.gguf", &g, true).is_none(), "--force");
    }

    // -- resolve_rm ------------------------------------------------------------

    #[test]
    fn resolve_rm_exact_substring_ambiguous() {
        let d = tmpdir("resolve");
        write_file(&d.join("Alpha-7B-Q4.gguf"), &minimal_gguf());
        write_file(&d.join("Alpha-13B-Q4.gguf"), &minimal_gguf());
        write_file(&d.join("Beta-7B-Q4.gguf"), &minimal_gguf());

        assert_eq!(
            resolve_rm(&d, "Alpha-7B-Q4.gguf").unwrap(),
            "Alpha-7B-Q4.gguf",
            "exact name wins even with a sibling substring match"
        );
        assert_eq!(resolve_rm(&d, "beta").unwrap(), "Beta-7B-Q4.gguf");
        assert!(resolve_rm(&d, "alpha").is_err(), "ambiguous");
        assert!(resolve_rm(&d, "gamma").is_err(), "no match");
        assert!(resolve_rm(&d, "  ").is_err(), "blank");
        fs::remove_dir_all(&d).ok();
    }
}
