// SPDX-License-Identifier: GPL-2.0-only
//! The NixOS netboot image pipeline: build, stage, and serve the ATOMIC
//! co-built triple (kernel, initrd, netboot.ipxe) so an init=/initrd mismatch
//! is structurally impossible.
//!
//! WHY THIS EXISTS (the bug this module makes unrepresentable): the NixOS
//! netboot boot script carries `init=/nix/store/<system>/init`, a path INSIDE
//! the squashfs embedded in the co-built initrd. Serving an `init=` from build
//! A with build B's initrd boots stage 1 fine, then stage 1 cannot find that
//! init in the mounted store -> "stage 2 init script not found", which on a
//! headless board reads like a stage-1 hang (this cost a whole session of
//! power-cycles). The fix is structural:
//!
//! 1. `netboot image build` produces all three artifacts from ONE `nix build`
//!    invocation of the image flake (`.#netbootKernel .#netbootRamdisk
//!    .#netbootIpxe`, or the `[netboot] image_variant`-suffixed set, e.g.
//!    `.#netbootKernelLlmtune ...`) - they can only ever be from the same
//!    system closure.
//! 2. The `init=` path is EXTRACTED from the co-built netboot.ipxe and recorded
//!    together with the initrd's sha256 in the image `manifest.toml`.
//! 3. Staging and serving derive `init=` and the initrd FROM THAT MANIFEST and
//!    re-verify the sha256 before every served boot script. A mismatch is a
//!    hard error - the board gets a 500, never a mismatched pair.
//!
//! There is deliberately NO way to hand-pick an `init=` or an initrd
//! independently: the only inputs are "build an image" and "stage image <id>".

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::config::Netboot;

/// Manifest file name inside an image directory.
pub const MANIFEST_FILE: &str = "manifest.toml";
/// Kernel artifact file name inside an image directory.
pub const KERNEL_FILE: &str = "vmlinuz";
/// Initrd artifact file name inside an image directory.
pub const INITRD_FILE: &str = "initrd";
/// The co-built iPXE script, kept verbatim for audit.
pub const IPXE_FILE: &str = "netboot.ipxe";
/// The active-image pointer file inside the images root.
pub const ACTIVE_FILE: &str = "active";
/// Manifest schema version this build writes.
pub const MANIFEST_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Manifest: the record that binds init= to the initrd it was built with.
// ---------------------------------------------------------------------------

/// One artifact of the co-built triple: its file name (relative to the image
/// directory, so the directory can be moved) and its sha256.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub file: String,
    pub sha256: String,
}

/// The image manifest. `init` and `initrd.sha256` are recorded TOGETHER from
/// one build; every stage/serve verifies them together.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version (see [`MANIFEST_VERSION`]).
    pub version: u32,
    /// Image id (also the directory name), e.g. `20260707-2130-ab12cd34ef56`.
    pub id: String,
    /// Build time, unix seconds UTC.
    pub built_at_unix: u64,
    /// The stage-2 init path extracted from the CO-BUILT netboot.ipxe
    /// (`/nix/store/<system>/init`). Must exist inside `initrd`'s squashfs.
    pub init: String,
    /// The full kernel argument string from the co-built netboot.ipxe,
    /// verbatim (includes `init=` and the iPXE `${cmdline}` passthrough).
    pub cmdline: String,
    pub kernel: Artifact,
    pub initrd: Artifact,
}

impl Manifest {
    /// Load and parse `<dir>/manifest.toml`.
    pub fn load(dir: &Path) -> Result<Manifest> {
        let p = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&p)
            .with_context(|| format!("reading image manifest {}", p.display()))?;
        let m: Manifest = toml::from_str(&text)
            .with_context(|| format!("parsing image manifest {}", p.display()))?;
        Ok(m)
    }

    /// Write `<dir>/manifest.toml` durably.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let text = toml::to_string_pretty(self).context("serializing image manifest")?;
        crate::paths::write_durable(&dir.join(MANIFEST_FILE), text.as_bytes())
            .with_context(|| format!("writing {}/{MANIFEST_FILE}", dir.display()))
    }
}

// ---------------------------------------------------------------------------
// Extraction: pull init= + the kernel args out of the co-built netboot.ipxe.
// ---------------------------------------------------------------------------

/// The kernel argument string from a NixOS `netboot.ipxe`: everything on the
/// `kernel` line after the image token (`bzImage`). Verbatim, so the served
/// boot script boots with exactly the co-built parameters.
pub fn extract_kernel_args(ipxe: &str) -> Result<String> {
    for line in ipxe.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("kernel ") {
            let mut parts = rest.trim().splitn(2, char::is_whitespace);
            let _image = parts.next(); // bzImage (or a path/URL to it)
            let args = parts.next().unwrap_or("").trim();
            if args.is_empty() {
                bail!("netboot.ipxe kernel line has no arguments: '{line}'");
            }
            return Ok(args.to_string());
        }
    }
    bail!("no 'kernel' line found in netboot.ipxe")
}

/// The `init=/nix/store/<system>/init` value from a kernel argument string.
/// Rejects anything that is not a store init path - this is the value whose
/// pairing with the initrd we guarantee.
pub fn extract_init(kernel_args: &str) -> Result<String> {
    for tok in kernel_args.split_whitespace() {
        if let Some(v) = tok.strip_prefix("init=") {
            if !v.starts_with("/nix/store/") || !v.ends_with("/init") {
                bail!("init= is not a /nix/store/<system>/init path: '{v}'");
            }
            return Ok(v.to_string());
        }
    }
    bail!("no init= parameter in the netboot.ipxe kernel arguments")
}

// ---------------------------------------------------------------------------
// Verification: the mismatch gate. Hard errors, never warnings.
// ---------------------------------------------------------------------------

/// Streaming sha256 of a file, lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("opening {} for hashing", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Verify an image directory against its manifest. This is THE gate: it runs
/// on stage and again before every served boot script. Checks, all hard
/// errors:
/// - manifest schema version is known;
/// - `init` is a store init path and appears verbatim as `init=<init>` in the
///   recorded cmdline (a fabricated (init, cmdline) pair is rejected);
/// - the on-disk initrd's sha256 matches the manifest (the co-built pairing);
/// - the on-disk kernel's sha256 matches the manifest.
pub fn verify(dir: &Path, m: &Manifest) -> Result<()> {
    if m.version != MANIFEST_VERSION {
        bail!(
            "image {}: unsupported manifest version {} (this llmtune understands {})",
            m.id,
            m.version,
            MANIFEST_VERSION
        );
    }
    if !m.init.starts_with("/nix/store/") || !m.init.ends_with("/init") {
        bail!(
            "image {}: manifest init '{}' is not a store init path",
            m.id,
            m.init
        );
    }
    let init_tok = format!("init={}", m.init);
    if !m.cmdline.split_whitespace().any(|t| t == init_tok) {
        bail!(
            "image {}: manifest init= ({}) does not match the recorded cmdline - \
             the (init, initrd) pair is not from one build; refusing to stage/serve",
            m.id,
            m.init
        );
    }
    for (what, a) in [("initrd", &m.initrd), ("kernel", &m.kernel)] {
        let p = dir.join(&a.file);
        let got = sha256_file(&p)?;
        if got != a.sha256 {
            bail!(
                "image {}: {what} sha256 MISMATCH\n  manifest: {}\n  on disk:  {got}\n  \
                 file: {}\nThe served init= would not match this {what} - a board booting \
                 this pair dies with 'stage 2 init script not found'. Rebuild the image \
                 (`llmtune netboot image build`); never replace one artifact by hand.",
                m.id,
                a.sha256,
                p.display()
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The image store: <images_dir>/<id>/{vmlinuz,initrd,netboot.ipxe,manifest.toml}
// plus an `active` pointer file naming the staged image.
// ---------------------------------------------------------------------------

/// A safe image id: what `build` generates, and the only thing `stage`
/// accepts (it becomes a path component).
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && !id.starts_with('.')
}

/// List built images: (manifest, dir), newest first. Unreadable or unparseable
/// directories are skipped (they cannot be staged anyway).
pub fn list(images_dir: &Path) -> Vec<(Manifest, PathBuf)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(images_dir) else {
        return out;
    };
    for e in rd.flatten() {
        let dir = e.path();
        if dir.is_dir() {
            if let Ok(m) = Manifest::load(&dir) {
                out.push((m, dir));
            }
        }
    }
    out.sort_by_key(|(m, _)| std::cmp::Reverse(m.built_at_unix));
    out
}

/// The staged (active) image id, if any.
pub fn active_id(images_dir: &Path) -> Option<String> {
    let s = std::fs::read_to_string(images_dir.join(ACTIVE_FILE)).ok()?;
    let id = s.trim().to_string();
    (!id.is_empty()).then_some(id)
}

/// Stage an image: VERIFY it against its manifest, then atomically point the
/// `active` file at it. A failing verification leaves the previous staging
/// untouched.
pub fn stage(images_dir: &Path, id: &str) -> Result<Manifest> {
    if !valid_id(id) {
        bail!("invalid image id '{id}'");
    }
    let dir = images_dir.join(id);
    let m = Manifest::load(&dir)
        .with_context(|| format!("no built image '{id}' under {}", images_dir.display()))?;
    verify(&dir, &m)?;
    crate::paths::write_durable(&images_dir.join(ACTIVE_FILE), format!("{id}\n").as_bytes())
        .context("writing the active-image pointer")?;
    Ok(m)
}

/// Load AND VERIFY the active image. `Ok(None)` when nothing is staged;
/// `Err` when the staged image fails verification (serving must hard-fail,
/// not fall back).
pub fn load_active(images_dir: &Path) -> Result<Option<(Manifest, PathBuf)>> {
    let Some(id) = active_id(images_dir) else {
        return Ok(None);
    };
    if !valid_id(&id) {
        bail!("active-image pointer contains an invalid id '{id}'");
    }
    let dir = images_dir.join(&id);
    let m = Manifest::load(&dir)?;
    verify(&dir, &m)?;
    Ok(Some((m, dir)))
}

/// The boot script served to a board, derived ONLY from the manifest: the
/// kernel/initrd URLs point at this image's artifacts and the argument string
/// (including `init=`) is the co-built one, verbatim.
pub fn boot_script(m: &Manifest, http_base: &str) -> String {
    format!(
        "#!ipxe\n\
         # AUTO-GENERATED by llmtune from image {id} - init= is co-built with the\n\
         # served initrd (verified sha256 {sha}). Do not edit by hand.\n\
         kernel {base}/{KERNEL_FILE} {args}\n\
         initrd {base}/{INITRD_FILE}\n\
         boot\n",
        id = m.id,
        sha = &m.initrd.sha256[..12.min(m.initrd.sha256.len())],
        base = http_base,
        args = m.cmdline,
    )
}

// ---------------------------------------------------------------------------
// Build: ONE nix invocation -> the co-built triple -> a manifest-sealed image.
// ---------------------------------------------------------------------------

/// The three flake output base names, built together. Order matters: it is the
/// order `--print-out-paths` reports (kernel, ramdisk, ipxe).
const FLAKE_TARGET_BASES: [&str; 3] = ["netbootKernel", "netbootRamdisk", "netbootIpxe"];

/// Resolve the three `nix build` installables from the configured
/// `[netboot] image_variant`. Unset/empty: the flake's base outputs
/// (`.#netbootKernel` / `.#netbootRamdisk` / `.#netbootIpxe`). A variant
/// (e.g. `Llmtune`) is appended to each output name:
/// `.#netbootKernelLlmtune` / `.#netbootRamdiskLlmtune` / `.#netbootIpxeLlmtune`.
fn flake_targets(image_variant: Option<&str>) -> [String; 3] {
    let suffix = image_variant.map(str::trim).unwrap_or("");
    FLAKE_TARGET_BASES.map(|base| format!(".#{base}{suffix}"))
}

pub struct BuildOpts {
    pub apply: bool,
    /// Also stage (activate) the image after a successful build.
    pub stage: bool,
    pub json: bool,
}

/// `netboot image build` for a flake-backed image. Previews the plan by
/// default; `--apply` runs the build. Requires `[netboot] flake` in
/// fleet.toml.
pub fn build(nb: &Netboot, opts: &BuildOpts) -> Result<()> {
    let flake = nb.flake.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "netboot image build needs [netboot] flake = \
             \"/path/to/your/netboot-flake\" in fleet.toml (a NixOS flake you \
             provide, exposing the co-built .#netbootKernel / .#netbootRamdisk \
             / .#netbootIpxe outputs; set [netboot] image_variant to select a \
             suffixed output set, e.g. \"Llmtune\")"
        )
    })?;
    let images_dir = Path::new(&nb.images_dir);
    let targets = flake_targets(nb.image_variant.as_deref());

    if !opts.apply {
        let plan = format!(
            "netboot image build plan (flake {flake}):\n\
             \x20 1. [nix]      {nix} build {targets} --impure --print-out-paths --no-link\n\
             \x20             (ONE invocation - kernel, initrd, and netboot.ipxe are co-built\n\
             \x20              from the same system closure; a mismatch cannot exist)\n\
             \x20 2. [extract]  read init=/nix/store/<system>/init + kernel args from the\n\
             \x20              co-built netboot.ipxe\n\
             \x20 3. [seal]     copy the triple -> {root}/<id>/ and record init= + initrd\n\
             \x20              sha256 + kernel sha256 together in manifest.toml\n\
             \x20 4. [verify]   re-hash the sealed artifacts against the manifest\n\
             {stage}",
            nix = nb.nix_cmd,
            targets = targets.join(" "),
            root = images_dir.display(),
            stage = if opts.stage {
                "  5. [stage]    verify + point the active image at it\n"
            } else {
                "  (not staged automatically - `llmtune netboot image stage <id>` when ready)\n"
            },
        );
        if opts.json {
            println!(
                "{}",
                serde_json::json!({"plan": plan, "flake": flake, "apply": false})
            );
        } else {
            println!("{plan}\nRe-run with --apply to execute.");
        }
        return Ok(());
    }

    let m = build_apply(nb, flake, images_dir, opts.json)?;
    let staged = if opts.stage {
        stage(images_dir, &m.id)?;
        true
    } else {
        false
    };

    if opts.json {
        println!(
            "{}",
            serde_json::json!({
                "id": m.id,
                "dir": images_dir.join(&m.id).display().to_string(),
                "built_at": crate::fmt::fmt_ts(m.built_at_unix),
                "init": m.init,
                "kernel_sha256": m.kernel.sha256,
                "initrd_sha256": m.initrd.sha256,
                "staged": staged,
            })
        );
    } else {
        println!(
            "[ok] image {} built + sealed (init= {})\n     initrd sha256 {}",
            m.id, m.init, m.initrd.sha256
        );
        if staged {
            println!("[ok] staged as the active image");
        } else {
            println!("     stage it with `llmtune netboot image stage {}`", m.id);
        }
    }
    Ok(())
}

/// Run the ONE co-build and seal the result under `<images_dir>/<id>/`.
/// Under `--json` progress goes to stderr (stdout carries only the final
/// summary object the caller prints).
fn build_apply(nb: &Netboot, flake: &str, images_dir: &Path, json: bool) -> Result<Manifest> {
    // Resolve a flake-relative nix command (e.g. `.tool/np`, the nix-portable
    // wrapper) against the flake dir, since we run with cwd = flake.
    let nix: PathBuf = if nb.nix_cmd.contains('/') && !nb.nix_cmd.starts_with('/') {
        Path::new(flake).join(&nb.nix_cmd)
    } else {
        PathBuf::from(&nb.nix_cmd)
    };

    let targets = flake_targets(nb.image_variant.as_deref());
    let mut cmd = std::process::Command::new(&nix);
    cmd.current_dir(flake)
        .arg("build")
        .args(&targets)
        .args(["--impure", "--print-out-paths", "--no-link"])
        .stderr(std::process::Stdio::inherit());
    let progress = format!(
        "[..] {} build {} (one invocation; this can take a while)",
        nix.display(),
        targets.join(" ")
    );
    if json {
        eprintln!("{progress}");
    } else {
        println!("{progress}");
    }
    let out = cmd
        .output()
        .with_context(|| format!("running {} build", nix.display()))?;
    if !out.status.success() {
        bail!("nix build failed ({})", out.status);
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let paths: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if paths.len() != targets.len() {
        bail!(
            "expected {} output paths from `nix build --print-out-paths`, got {}:\n{}",
            targets.len(),
            paths.len(),
            stdout
        );
    }
    let (kernel_out, ramdisk_out, ipxe_out) = (paths[0], paths[1], paths[2]);

    // Extract the pairing facts from the CO-BUILT netboot.ipxe.
    let ipxe_src = Path::new(ipxe_out).join(IPXE_FILE);
    let ipxe_text = std::fs::read_to_string(&ipxe_src)
        .with_context(|| format!("reading co-built {}", ipxe_src.display()))?;
    let cmdline = extract_kernel_args(&ipxe_text)?;
    let init = extract_init(&cmdline)?;

    let kernel_src = Path::new(kernel_out).join("bzImage");
    let initrd_src = Path::new(ramdisk_out).join("initrd");

    let kernel_sha = sha256_file(&kernel_src)?;
    let initrd_sha = sha256_file(&initrd_src)?;

    let now = crate::history::now_unix();
    // fmt_ts renders "YYYY-MM-DD HH:MM"; compact it into the id.
    let ts = crate::fmt::fmt_ts(now)
        .replace(['-', ':'], "")
        .replace(' ', "-");
    let id = format!("{ts}-{}", &initrd_sha[..12]);
    let dir = images_dir.join(&id);

    let m = Manifest {
        version: MANIFEST_VERSION,
        id: id.clone(),
        built_at_unix: now,
        init,
        cmdline,
        kernel: Artifact {
            file: KERNEL_FILE.into(),
            sha256: kernel_sha,
        },
        initrd: Artifact {
            file: INITRD_FILE.into(),
            sha256: initrd_sha,
        },
    };

    // An identical rebuild (same minute + same initrd) is a no-op reuse.
    if dir.join(MANIFEST_FILE).exists() {
        let existing = Manifest::load(&dir)?;
        if existing.initrd.sha256 == m.initrd.sha256 && existing.kernel.sha256 == m.kernel.sha256 {
            let msg = format!("[ok] identical image already sealed as {id}");
            if json {
                eprintln!("{msg}");
            } else {
                println!("{msg}");
            }
            return Ok(existing);
        }
        bail!(
            "image dir {} already exists with different contents",
            dir.display()
        );
    }

    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating image dir {}", dir.display()))?;
    // fs::copy follows symlinks (nix outputs link initrd -> initrd.zst).
    std::fs::copy(&kernel_src, dir.join(KERNEL_FILE))
        .with_context(|| format!("copying kernel from {}", kernel_src.display()))?;
    std::fs::copy(&initrd_src, dir.join(INITRD_FILE))
        .with_context(|| format!("copying initrd from {}", initrd_src.display()))?;
    crate::paths::write_durable(&dir.join(IPXE_FILE), ipxe_text.as_bytes())
        .context("recording the co-built netboot.ipxe")?;
    m.save(&dir)?;

    // Self-check: the sealed copies must match what we hashed from the store.
    verify(&dir, &m).context("post-seal self-verification")?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A netboot.ipxe fixture shaped exactly like the nix-built one from the
    /// reference image (store hash + params from the working session).
    const IPXE_FIXTURE: &str = "#!ipxe\n\
        # Use the cmdline variable to allow the user to specify custom kernel params\n\
        # when chainloading this script from other iPXE scripts like netboot.xyz\n\
        kernel bzImage init=/nix/store/zm5mw56vf9yvwbnwpa84ida5r7xn6rsl-nixos-system-bc250-24.11.20250630.50ab793/init initrd=initrd ttm.pages_limit=4194304 amdgpu.gartsize=16384 loglevel=4 ${cmdline}\n\
        initrd initrd\n\
        boot\n";

    fn tdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "llmtune-nbimg-{tag}-{}-{}",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("t")
                .replace("::", "-")
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Write a sealed image dir whose manifest genuinely matches its bytes.
    fn seal(dir: &Path, id: &str) -> Manifest {
        std::fs::write(dir.join(KERNEL_FILE), b"kernel-bytes-A").unwrap();
        std::fs::write(dir.join(INITRD_FILE), b"initrd-bytes-A").unwrap();
        let cmdline = extract_kernel_args(IPXE_FIXTURE).unwrap();
        let init = extract_init(&cmdline).unwrap();
        let m = Manifest {
            version: MANIFEST_VERSION,
            id: id.into(),
            built_at_unix: 1_782_909_240,
            init,
            cmdline,
            kernel: Artifact {
                file: KERNEL_FILE.into(),
                sha256: sha256_file(&dir.join(KERNEL_FILE)).unwrap(),
            },
            initrd: Artifact {
                file: INITRD_FILE.into(),
                sha256: sha256_file(&dir.join(INITRD_FILE)).unwrap(),
            },
        };
        m.save(dir).unwrap();
        m
    }

    #[test]
    fn flake_targets_default_and_variant() {
        // Unset: the flake's base (default) outputs, in kernel/ramdisk/ipxe order.
        assert_eq!(
            flake_targets(None),
            [".#netbootKernel", ".#netbootRamdisk", ".#netbootIpxe"].map(String::from)
        );
        // A variant suffixes all three names, same order.
        assert_eq!(
            flake_targets(Some("Llmtune")),
            [
                ".#netbootKernelLlmtune",
                ".#netbootRamdiskLlmtune",
                ".#netbootIpxeLlmtune"
            ]
            .map(String::from)
        );
        // Empty / whitespace-only means unset (the base outputs).
        assert_eq!(flake_targets(Some("")), flake_targets(None));
        assert_eq!(flake_targets(Some("  ")), flake_targets(None));
    }

    #[test]
    fn sha256_matches_known_vector() {
        let d = tdir("sha");
        let p = d.join("abc");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn extracts_init_and_args_from_the_cobuilt_ipxe() {
        let args = extract_kernel_args(IPXE_FIXTURE).unwrap();
        assert!(args.starts_with("init=/nix/store/"), "init= leads the args");
        assert!(args.contains("ttm.pages_limit=4194304"));
        assert!(
            args.contains("${cmdline}"),
            "iPXE passthrough kept verbatim"
        );
        let init = extract_init(&args).unwrap();
        assert_eq!(
            init,
            "/nix/store/zm5mw56vf9yvwbnwpa84ida5r7xn6rsl-nixos-system-bc250-24.11.20250630.50ab793/init"
        );
    }

    #[test]
    fn extraction_rejects_missing_or_bogus_init() {
        assert!(
            extract_kernel_args("#!ipxe\nboot\n").is_err(),
            "no kernel line"
        );
        assert!(
            extract_init("root=/dev/sda1 loglevel=4").is_err(),
            "no init="
        );
        assert!(
            extract_init("init=/sbin/init loglevel=4").is_err(),
            "non-store init rejected: it cannot be pairing-verified"
        );
    }

    #[test]
    fn manifest_roundtrips_through_toml() {
        let d = tdir("rt");
        let m = seal(&d, "img-rt");
        let loaded = Manifest::load(&d).unwrap();
        assert_eq!(loaded, m);
    }

    #[test]
    fn verify_accepts_the_cobuilt_pair() {
        let d = tdir("ok");
        let m = seal(&d, "img-ok");
        verify(&d, &m).unwrap();
    }

    #[test]
    fn verify_rejects_a_mismatched_initrd() {
        // THE bug: init= from build A served with build B's initrd. Fabricate
        // it by swapping the initrd bytes under a sealed manifest.
        let d = tdir("mm");
        let m = seal(&d, "img-mm");
        std::fs::write(d.join(INITRD_FILE), b"initrd-bytes-B-other-build").unwrap();
        let err = verify(&d, &m).unwrap_err().to_string();
        assert!(err.contains("MISMATCH"), "hard error, got: {err}");
        assert!(
            err.contains("stage 2 init script not found"),
            "names the failure mode"
        );
    }

    #[test]
    fn verify_rejects_an_init_not_from_the_recorded_cmdline() {
        // Fabricated (init=, initrd) pair: manifest init points at build B's
        // system while the recorded co-built cmdline carries build A's.
        let d = tdir("init");
        let mut m = seal(&d, "img-init");
        m.init = "/nix/store/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-nixos-system-bc250-other/init".into();
        let err = verify(&d, &m).unwrap_err().to_string();
        assert!(err.contains("not from one build"), "got: {err}");
    }

    #[test]
    fn verify_rejects_a_tampered_kernel_and_unknown_version() {
        let d = tdir("k");
        let mut m = seal(&d, "img-k");
        std::fs::write(d.join(KERNEL_FILE), b"kernel-bytes-B").unwrap();
        assert!(verify(&d, &m).is_err(), "kernel hash gate");
        // restore, then break the version
        std::fs::write(d.join(KERNEL_FILE), b"kernel-bytes-A").unwrap();
        m.version = 99;
        assert!(verify(&d, &m).unwrap_err().to_string().contains("version"));
    }

    #[test]
    fn stage_refuses_a_mismatched_image_and_leaves_staging_untouched() {
        let root = tdir("stage");
        let good = root.join("img-good");
        std::fs::create_dir_all(&good).unwrap();
        seal(&good, "img-good");
        stage(&root, "img-good").unwrap();
        assert_eq!(active_id(&root).as_deref(), Some("img-good"));

        let bad = root.join("img-bad");
        std::fs::create_dir_all(&bad).unwrap();
        seal(&bad, "img-bad");
        std::fs::write(bad.join(INITRD_FILE), b"other-builds-initrd").unwrap();
        assert!(stage(&root, "img-bad").is_err(), "mismatch must not stage");
        assert_eq!(
            active_id(&root).as_deref(),
            Some("img-good"),
            "previous staging survives a refused stage"
        );
        // Path-shaped ids never reach the filesystem.
        assert!(stage(&root, "../escape").is_err());
        assert!(stage(&root, ".hidden").is_err());
    }

    #[test]
    fn load_active_hard_fails_on_a_corrupted_staged_image() {
        let root = tdir("act");
        let img = root.join("img-a");
        std::fs::create_dir_all(&img).unwrap();
        seal(&img, "img-a");
        stage(&root, "img-a").unwrap();
        // Corrupt AFTER staging: serving must hard-error, not fall back.
        std::fs::write(img.join(INITRD_FILE), b"swapped-behind-our-back").unwrap();
        assert!(load_active(&root).is_err());
        // Nothing staged -> cleanly None.
        let empty = tdir("act-empty");
        assert!(load_active(&empty).unwrap().is_none());
    }

    #[test]
    fn boot_script_derives_everything_from_the_manifest() {
        let d = tdir("bs");
        let m = seal(&d, "img-bs");
        let s = boot_script(&m, "http://198.51.100.225:8090");
        assert!(s.starts_with("#!ipxe"));
        assert!(
            s.contains(&format!("init={}", m.init)),
            "served init= is the manifest's"
        );
        assert!(s.contains("kernel http://198.51.100.225:8090/vmlinuz "));
        assert!(s.contains("initrd http://198.51.100.225:8090/initrd"));
        assert!(s.contains("${cmdline}"), "iPXE passthrough survives");
    }

    #[test]
    fn list_orders_newest_first_and_skips_junk() {
        let root = tdir("list");
        for (id, at) in [("img-old", 100u64), ("img-new", 200u64)] {
            let d = root.join(id);
            std::fs::create_dir_all(&d).unwrap();
            let mut m = seal(&d, id);
            m.built_at_unix = at;
            m.save(&d).unwrap();
        }
        std::fs::create_dir_all(root.join("not-an-image")).unwrap();
        std::fs::write(root.join("stray-file"), b"x").unwrap();
        let l = list(&root);
        assert_eq!(
            l.iter().map(|(m, _)| m.id.as_str()).collect::<Vec<_>>(),
            vec!["img-new", "img-old"]
        );
    }
}
