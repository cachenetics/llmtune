// SPDX-License-Identifier: GPL-2.0-only
//! Preflight: a best-effort report of what is and isn't ready to run AI on the
//! local box. Every check degrades gracefully - nothing here panics or requires
//! a specific environment.

use crate::config::Node;
use crate::{llama, model, profile};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Warn,
    Fail,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Status::Ok => "[ok]  ",
            Status::Warn => "[warn]",
            Status::Fail => "[fail]",
        };
        f.write_str(s)
    }
}

/// Serialized as the `node doctor --json` wire shape (label/status/detail), so
/// the SSH transport parses exactly what the node-side CLI emits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Check {
    pub label: String,
    pub status: Status,
    pub detail: String,
}

impl Check {
    fn new(label: &str, status: Status, detail: impl Into<String>) -> Check {
        Check {
            label: label.to_string(),
            status,
            detail: detail.into(),
        }
    }
}

/// The BC-250 APU PCI id (vendor 0x1002 AMD, device 0x13fe).
const BC250_PCI: (&str, &str) = ("0x1002", "0x13fe");

fn pci_has(vendor: &str, device: &str) -> bool {
    let base = Path::new("/sys/bus/pci/devices");
    let rd = match std::fs::read_dir(base) {
        Ok(rd) => rd,
        Err(_) => return false,
    };
    for ent in rd.flatten() {
        let p = ent.path();
        let v = std::fs::read_to_string(p.join("vendor")).unwrap_or_default();
        let d = std::fs::read_to_string(p.join("device")).unwrap_or_default();
        if v.trim().eq_ignore_ascii_case(vendor) && d.trim().eq_ignore_ascii_case(device) {
            return true;
        }
    }
    false
}

fn module_loaded(name: &str) -> bool {
    Path::new(&format!("/sys/module/{name}")).exists()
}

/// Is `cmd` an executable found on `$PATH`?
fn in_path(cmd: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        let p = dir.join(cmd);
        p.is_file() || p.is_symlink()
    })
}

/// First of several alternative commands found on `$PATH` (e.g. a C++ compiler).
fn any_in_path(cmds: &[&'static str]) -> Option<&'static str> {
    cmds.iter().copied().find(|c| in_path(c))
}

/// Best-effort distro id from /etc/os-release (for an install hint).
fn distro_id() -> String {
    std::fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("ID=")
                    .map(|v| v.trim_matches('"').to_string())
            })
        })
        .unwrap_or_default()
}

/// Install hint for the build toolchain, tailored to the detected distro.
fn deps_hint(missing: &[&str]) -> String {
    let pkgs = missing.join(" ");
    match distro_id().as_str() {
        "arch" | "cachyos" | "endeavouros" | "manjaro" => {
            "install: sudo pacman -S --needed base-devel cmake git vulkan-headers shaderc".into()
        }
        "debian" | "ubuntu" | "pop" | "linuxmint" => {
            "install: sudo apt install build-essential cmake git libvulkan-dev glslc".into()
        }
        "fedora" | "rhel" | "centos" => {
            "install: sudo dnf install gcc-c++ cmake git vulkan-headers glslc".into()
        }
        _ => format!("install the build toolchain for your distro (missing: {pkgs})"),
    }
}

fn glob_exists(dir: &str, prefix: &str) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.file_name().to_string_lossy().starts_with(prefix))
        })
        .unwrap_or(false)
}

/// Run all preflight checks for a node's local environment.
pub fn run(node: &Node) -> Vec<Check> {
    let mut out = Vec::new();

    // Hardware.
    if pci_has(BC250_PCI.0, BC250_PCI.1) {
        out.push(Check::new(
            "BC-250 APU",
            Status::Ok,
            "PCI 1002:13fe present",
        ));
    } else {
        out.push(Check::new(
            "BC-250 APU",
            Status::Warn,
            "1002:13fe not found - not a BC-250, or running remotely",
        ));
    }

    // GPU stack.
    if module_loaded("amdgpu") {
        out.push(Check::new("amdgpu", Status::Ok, "module loaded"));
    } else {
        out.push(Check::new(
            "amdgpu",
            Status::Fail,
            "amdgpu not loaded - no GPU inference backend",
        ));
    }

    if glob_exists("/dev/dri", "renderD") {
        out.push(Check::new(
            "DRM render node",
            Status::Ok,
            "/dev/dri/renderD* present",
        ));
    } else {
        out.push(Check::new(
            "DRM render node",
            Status::Fail,
            "no /dev/dri/renderD* - GPU not usable",
        ));
    }

    // Vulkan ICD (best-effort: presence of an ICD manifest dir).
    let vk_icd = ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"]
        .iter()
        .any(|d| glob_exists(d, ""));
    if vk_icd {
        out.push(Check::new(
            "Vulkan ICD",
            Status::Ok,
            "ICD manifest dir present",
        ));
    } else {
        out.push(Check::new(
            "Vulkan ICD",
            Status::Warn,
            "no Vulkan ICD manifest found - needed for the inference backend",
        ));
    }

    // Build toolchain - what `llmtune build install` needs to compile a Vulkan
    // llama.cpp. Only relevant on a box that builds its own engine, so a miss is a
    // Warn (with a per-distro install hint), never a Fail.
    {
        let mut missing: Vec<&str> = Vec::new();
        if !in_path("git") {
            missing.push("git");
        }
        if !in_path("cmake") {
            missing.push("cmake");
        }
        if any_in_path(&["c++", "g++", "clang++"]).is_none() {
            missing.push("c++");
        }
        // glslc (shaderc) compiles the Vulkan compute shaders for the GGML backend.
        if any_in_path(&["glslc", "glslangValidator"]).is_none() {
            missing.push("glslc");
        }
        if missing.is_empty() {
            out.push(Check::new(
                "build toolchain",
                Status::Ok,
                "git, cmake, c++ and a Vulkan shader compiler present",
            ));
        } else {
            out.push(Check::new(
                "build toolchain",
                Status::Warn,
                format!("missing {} - {}", missing.join(", "), deps_hint(&missing)),
            ));
        }
    }

    // Models directory.
    match model::discover(Path::new(&node.models_dir)) {
        Ok(ms) if !ms.is_empty() => out.push(Check::new(
            "models dir",
            Status::Ok,
            format!("{} - {} model(s)", node.models_dir, ms.len()),
        )),
        Ok(_) => out.push(Check::new(
            "models dir",
            Status::Warn,
            format!(
                "{} - no GGUF models found; drop .gguf files here (see `llmtune node list`)",
                node.models_dir
            ),
        )),
        Err(e) => out.push(Check::new(
            "models dir",
            Status::Fail,
            format!("{}: {e}", node.models_dir),
        )),
    }

    // Profile launch binaries - catch a profile whose build isn't installed (the
    // common footgun on a fresh box) or whose literal `bin` path is missing.
    // `launch()` resolves a managed `build` to a concrete path; an unresolved
    // build returns a bare name (not absolute), which is the signal it's missing.
    match profile::load() {
        Ok(ps) => {
            let mut missing_builds: Vec<&str> = Vec::new();
            let mut missing_bins: Vec<String> = Vec::new();
            for p in &ps {
                let (bin, _ld) = p.launch();
                if let Some(b) = p.build.as_deref().filter(|b| !b.is_empty()) {
                    // A managed build resolves to an absolute path when installed.
                    if !Path::new(&bin).is_absolute() {
                        missing_builds.push(b);
                    }
                } else if !Path::new(&bin).exists() {
                    missing_bins.push(bin);
                }
            }
            missing_builds.sort();
            missing_builds.dedup();
            missing_bins.sort();
            missing_bins.dedup();
            if missing_builds.is_empty() && missing_bins.is_empty() {
                out.push(Check::new(
                    "profile bins",
                    Status::Ok,
                    "all launch binaries present",
                ));
            } else {
                let mut parts = Vec::new();
                if !missing_builds.is_empty() {
                    parts.push(format!(
                        "build(s) not installed: {} - `llmtune build install {}`",
                        missing_builds.join(", "),
                        missing_builds[0]
                    ));
                }
                if !missing_bins.is_empty() {
                    parts.push(format!("missing bin path(s): {}", missing_bins.join(", ")));
                }
                out.push(Check::new("profile bins", Status::Warn, parts.join("; ")));
            }

            // Cluster readiness: does the managed build actually provide
            // `rpc-server` (SPEC 6.5)? The builds.toml recipe compiles with
            // -DGGML_RPC=ON, but a build installed before the recipe gained
            // that flag has no rpc-server and can never join a cluster (the
            // worker start fails and `cluster up` auto-reverts). Reported
            // whenever a profile-referenced build is installed - informational
            // on a single-node box, load-bearing once clusters are configured.
            // A Warn, never a Fail: single-node serving doesn't need it. (An
            // uninstalled build is already covered by profile-bins above.)
            let mut names: Vec<&str> = ps
                .iter()
                .filter_map(|p| p.build.as_deref())
                .filter(|b| !b.is_empty())
                .collect();
            names.sort_unstable();
            names.dedup();
            let installed: Vec<(String, bool)> = names
                .iter()
                .filter(|b| crate::build::current_version(b).is_some())
                .map(|b| {
                    (
                        b.to_string(),
                        crate::build::current_bin(b, "rpc-server").is_some(),
                    )
                })
                .collect();
            if let Some(c) = rpc_server_check(&installed) {
                out.push(c);
            }
        }
        Err(e) => out.push(Check::new("profiles", Status::Fail, format!("{e}"))),
    }

    // Crash-safe fallback: without llmtune's safe drop-in, an unloaded or
    // cleared unit falls back to the base ExecStart, which can crash-loop if it
    // names a stale/missing binary. `setup` installs it. Checked across every
    // drop-in root (/etc, /run, vendor) - on the netboot image it lives in /run.
    let safe = crate::swap::dropin_file(&node.llama_unit, "00-llmtune-safe.conf").is_some();
    out.push(if safe {
        Check::new("crash-safe fallback", Status::Ok, "installed")
    } else {
        Check::new(
            "crash-safe fallback",
            Status::Warn,
            "missing - run `llmtune setup` (prevents crash-loops when no model is loaded)",
        )
    });

    // Crash-loop guard: caps restarts so a model that fails to load can't storm
    // Vulkan init and wedge the GPU. `setup` installs it; without it a bad model
    // restarts forever. Also confirm the base unit actually carries a StartLimit
    // (a drop-in can set it, but if the base unit predates llmtune the guard
    // drop-in is the only thing standing between a bad load and a wedge).
    let guard = crate::swap::dropin_file(&node.llama_unit, "01-llmtune-limits.conf").is_some();
    out.push(if guard {
        Check::new("crash-loop guard", Status::Ok, "installed (StartLimit)")
    } else {
        Check::new(
            "crash-loop guard",
            Status::Warn,
            "missing - run `llmtune setup` (a bad model could crash-loop and wedge the GPU)",
        )
    });

    // Live server (optional - only Ok/Warn, never Fail; a down server is fine).
    if llama::health_ok(&node.llama_url) {
        let served = llama::served_name(&node.llama_url).unwrap_or_else(|| "?".into());
        out.push(Check::new(
            "llama-server",
            Status::Ok,
            format!("{} healthy, serving {served}", node.llama_url),
        ));
    } else {
        out.push(Check::new(
            "llama-server",
            Status::Warn,
            format!("{} not responding (no model served yet)", node.llama_url),
        ));
    }

    out
}

/// The rpc-server presence check, pure for testability. `builds` is
/// `(name, has_rpc_server)` for every INSTALLED managed build a profile
/// references. None when no managed build is installed (nothing to check -
/// the profile-bins check already flags uninstalled builds).
fn rpc_server_check(builds: &[(String, bool)]) -> Option<Check> {
    if builds.is_empty() {
        return None;
    }
    let missing: Vec<&str> = builds
        .iter()
        .filter(|(_, has)| !has)
        .map(|(n, _)| n.as_str())
        .collect();
    Some(if missing.is_empty() {
        Check::new(
            "rpc-server",
            Status::Ok,
            "managed build(s) provide rpc-server (cluster-ready)",
        )
    } else {
        Check::new(
            "rpc-server",
            Status::Warn,
            format!(
                "build(s) without rpc-server: {} - this node cannot join a cluster; \
                 rebuild with RPC: `llmtune build install {}`",
                missing.join(", "),
                missing[0]
            ),
        )
    })
}

/// Worst status across a check set (for an exit code / summary).
pub fn worst(checks: &[Check]) -> Status {
    if checks.iter().any(|c| c.status == Status::Fail) {
        Status::Fail
    } else if checks.iter().any(|c| c.status == Status::Warn) {
        Status::Warn
    } else {
        Status::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_server_check_ok_when_present() {
        let c = rpc_server_check(&[("vulkan".into(), true)]).expect("installed build -> a check");
        assert_eq!(c.status, Status::Ok);
        assert_eq!(c.label, "rpc-server");
        assert!(c.detail.contains("cluster-ready"));
    }

    #[test]
    fn rpc_server_check_warns_when_absent_with_rebuild_hint() {
        // An installed build predating -DGGML_RPC=ON has no rpc-server: the box
        // can never join a cluster. Warn (not fail) with the rebuild remedy.
        let c = rpc_server_check(&[("vulkan".into(), false)]).unwrap();
        assert_eq!(c.status, Status::Warn);
        assert!(c.detail.contains("vulkan"));
        assert!(
            c.detail.contains("llmtune build install vulkan"),
            "warn must carry the actionable rebuild hint: {}",
            c.detail
        );
    }

    #[test]
    fn rpc_server_check_skipped_when_no_managed_build_installed() {
        // Nothing installed -> nothing to report (profile-bins already warns
        // about the missing build itself).
        assert!(rpc_server_check(&[]).is_none());
    }

    #[test]
    fn rpc_server_check_mixed_builds_warns_on_the_stale_one() {
        let c = rpc_server_check(&[("mtp-cross".into(), false), ("vulkan".into(), true)]).unwrap();
        assert_eq!(c.status, Status::Warn);
        assert!(c.detail.contains("mtp-cross"));
        assert!(!c.detail.contains("without rpc-server: vulkan"));
    }
}
