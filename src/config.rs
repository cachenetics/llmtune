// SPDX-License-Identifier: GPL-2.0-only
//! Fleet configuration: the set of nodes llmtune controls. A single local
//! BC-250 needs no config file - it is the implicit `localhost` node.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    Local,
    Ssh,
    Agent,
}

impl Transport {
    /// Parse a transport string. A missing/empty value is `Local`; an
    /// unrecognized value is an ERROR (silently defaulting a typo'd `ssh` to
    /// `Local` would actuate the wrong machine).
    fn parse(s: Option<&str>) -> Result<Transport> {
        match s.map(|x| x.to_lowercase()).as_deref() {
            None | Some("") | Some("local") => Ok(Transport::Local),
            Some("ssh") => Ok(Transport::Ssh),
            Some("agent") => Ok(Transport::Agent),
            Some(other) => bail!("unknown transport '{other}' (expected local, ssh, or agent)"),
        }
    }
}

/// A systemd unit name safe to interpolate into a path (`/etc/systemd/system/
/// <unit>`, its `.d/` drop-in dir, and `systemctl <unit>`). Rejects `/` and `..`
/// so a config value can't turn setup's `sudo tee` into an arbitrary-path write.
fn valid_unit_name(s: &str) -> bool {
    !s.is_empty()
        && s.ends_with(".service")
        && !s.contains('/')
        && !s.contains("..")
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '_' | '@' | '.' | '-'))
}

#[derive(Debug, Default, Deserialize)]
struct Defaults {
    models_dir: Option<String>,
    llama_unit: Option<String>,
    llama_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NodeCfg {
    name: String,
    #[serde(default)]
    host: Option<String>,
    #[serde(default)]
    transport: Option<String>,
    #[serde(default)]
    ssh_user: Option<String>,
    #[serde(default)]
    ssh_key: Option<String>,
    #[serde(default)]
    models_dir: Option<String>,
    #[serde(default)]
    llama_unit: Option<String>,
    #[serde(default)]
    llama_url: Option<String>,
    /// Local command (run through `sh -c` on the CONTROL host) that cold
    /// power-cycles this board - a smart-plug hook. Used by `netboot boot`;
    /// absent means the operator is prompted to pull power by hand.
    #[serde(default)]
    power_cmd: Option<String>,
}

/// A cluster: several nodes pooled (via llama.cpp RPC) to serve one model too
/// big for a single BC-250. The head runs llama-server --rpc <workers>; each
/// worker runs rpc-server.
#[derive(Debug, Clone, Deserialize)]
pub struct ClusterCfg {
    pub name: String,
    pub head: String,
    #[serde(default)]
    pub workers: Vec<String>,
    pub rpc_port: u16,
    #[serde(default)]
    pub rpc_bin: Option<String>,
    /// Address each worker's rpc-server binds. llama.cpp RPC is UNAUTHENTICATED
    /// (an arbitrary-memory surface), so set this to the fleet-facing interface
    /// IP rather than exposing all interfaces. Default: `127.0.0.1` (loopback);
    /// a cluster with REMOTE workers requires this to be set explicitly (the
    /// opt-in for an off-host-reachable rpc-server), and any non-loopback bind
    /// is warned about on `cluster up`.
    #[serde(default)]
    pub rpc_bind: Option<String>,
}

/// The netboot (PXE) control plane: llmtune orchestrates a proxyDHCP dnsmasq +
/// an NFS root and serves a per-node iPXE script, so BC-250s netboot diskless and
/// auto-register as ssh nodes. All optional - absent means netboot is unconfigured.
#[derive(Debug, Deserialize)]
struct NetbootCfg {
    /// Boot NIC dnsmasq binds to (e.g. `enp6s0`). Absent: auto-detected as the
    /// host's default-route interface.
    #[serde(default)]
    interface: Option<String>,
    /// This host's LAN IPv4 - the address boards fetch the kernel/rootfs from.
    /// Absent: auto-detected as the default-route interface's address.
    #[serde(default)]
    server_ip: Option<String>,
    /// The boot subnet base for proxyDHCP + the NFS export CIDR (e.g.
    /// `192.168.1.0`). Absent: derived from `server_ip` + `prefix_len` by
    /// masking off the host bits.
    #[serde(default)]
    subnet: Option<String>,
    /// CIDR prefix length for the NFS export (default 24).
    #[serde(default)]
    prefix_len: Option<u8>,
    /// Port llmtune's embedded artifact/iPXE server listens on (default 8090).
    #[serde(default)]
    http_port: Option<u16>,
    /// The NFS-exported diskless root (default `/srv/nfs/bc250-root`).
    #[serde(default)]
    rootfs: Option<String>,
    /// TFTP root dnsmasq serves `ipxe.efi` from (default `/srv/tftp`).
    #[serde(default)]
    tftp_root: Option<String>,
    /// Kernel image llmtune serves as `/vmlinuz` (default `<tftp_root>/vmlinuz`).
    #[serde(default)]
    kernel: Option<String>,
    /// NFS initramfs llmtune serves as `/initramfs-nfs.img`.
    #[serde(default)]
    initrd: Option<String>,
    /// Default OC profile baked into the boot cmdline (`eco|balanced|performance`).
    #[serde(default)]
    oc_profile: Option<String>,
    /// dnsmasq lease file to discover booted boards from.
    #[serde(default)]
    leases: Option<String>,
    /// Only treat leases whose MAC starts with one of these OUI prefixes as
    /// BC-250s (lowercase, colon-separated, e.g. `["58:11:22"]`). Empty = match by
    /// the `hostname_prefix` instead.
    #[serde(default)]
    mac_ouis: Vec<String>,
    /// Treat leases whose hostname starts with this as BC-250s (default `bc250`).
    #[serde(default)]
    hostname_prefix: Option<String>,

    // --- boot-server control plane (`netboot up/down/status`) ---
    /// The LAN CIDR the firewall openings and NFS exports are scoped to
    /// (default `<subnet>/<prefix_len>`).
    #[serde(default)]
    lan_cidr: Option<String>,
    /// The model library exported read-only over NFS to the boards
    /// (default `/var/lib/llmtune/models`).
    #[serde(default)]
    models_dir: Option<String>,
    /// Run the proxyDHCP dnsmasq as part of the stack (default false - boards
    /// chainloading iPXE from their ESP fetch boot.ipxe over HTTP directly).
    #[serde(default)]
    dnsmasq: Option<bool>,
    /// UDP port `netboot console` listens on for the boards' netconsole
    /// stream (default 6666, matching the image's netconsole sender).
    #[serde(default)]
    console_port: Option<u16>,

    // --- image pipeline (`netboot image build`) ---
    /// Source CachyOS ISO the diskless rootfs is extracted from.
    #[serde(default)]
    iso: Option<String>,
    /// Scratch dir for extract/build (default `/var/lib/llmtune/netboot`).
    #[serde(default)]
    work_dir: Option<String>,
    /// SSH public key baked into the image so llmtune can drive booted boards
    /// (default: the first `~/.ssh/*.pub` of the invoking user).
    #[serde(default)]
    ssh_pubkey: Option<String>,
    /// GTT (GPU) allocation baked into the TTM config, in GiB (default 12).
    #[serde(default)]
    ttm_gtt_gb: Option<u32>,

    // --- NixOS image pipeline (the co-built triple) ---
    /// Path to an operator-supplied NixOS image flake. The flake must expose
    /// the co-built `.#netbootKernel` / `.#netbootRamdisk` / `.#netbootIpxe`
    /// outputs (one system closure: kernel, initrd, and the iPXE script with
    /// its `init=` argument). When set, `netboot image build` builds the
    /// atomic (kernel, initrd, netboot.ipxe) triple from it instead of the
    /// CachyOS ISO pipeline. There is no default: the image source is
    /// bring-your-own (this flake, or `iso` for the legacy pipeline).
    #[serde(default)]
    flake: Option<String>,
    /// The nix command used to build the flake (default `nix`). A relative
    /// path containing `/` (e.g. `.tool/np`, a nix-portable wrapper) is
    /// resolved against the flake directory.
    #[serde(default)]
    nix_cmd: Option<String>,
    /// Image variant suffix appended to the three flake output names. Set to
    /// e.g. `"Llmtune"` to build `.#netbootKernelLlmtune` /
    /// `.#netbootRamdiskLlmtune` / `.#netbootIpxeLlmtune` instead of the base
    /// outputs. Absent or empty: the flake's default (base) outputs,
    /// `.#netbootKernel` / `.#netbootRamdisk` / `.#netbootIpxe`.
    #[serde(default)]
    image_variant: Option<String>,
    /// Where sealed images live (default `<work_dir>/images`). Each image is
    /// `<images_dir>/<id>/` with its manifest; `<images_dir>/active` names the
    /// staged one.
    #[serde(default)]
    images_dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RawConfig {
    #[serde(default)]
    defaults: Defaults,
    #[serde(default, rename = "node")]
    nodes: Vec<NodeCfg>,
    #[serde(default, rename = "cluster")]
    clusters: Vec<ClusterCfg>,
    #[serde(default)]
    netboot: Option<NetbootCfg>,
}

/// A fully-resolved netboot config (defaults applied, fields validated). Every
/// value here is trusted to be interpolated into a generated dnsmasq/NFS config,
/// an iPXE script, and a `sudo` argv, so `resolve` rejects anything unsafe.
#[derive(Debug, Clone)]
pub struct Netboot {
    pub interface: String,
    pub server_ip: String,
    pub subnet: String,
    pub prefix_len: u8,
    pub http_port: u16,
    pub rootfs: String,
    pub tftp_root: String,
    pub kernel: String,
    pub initrd: String,
    pub oc_profile: String,
    pub leases: String,
    pub mac_ouis: Vec<String>,
    pub hostname_prefix: String,
    pub lan_cidr: String,
    pub models_dir: String,
    pub dnsmasq: bool,
    pub console_port: u16,
    pub iso: Option<String>,
    pub work_dir: String,
    pub ssh_pubkey: Option<String>,
    pub ttm_gtt_gb: u32,
    pub flake: Option<String>,
    pub nix_cmd: String,
    /// Flake output-name suffix selecting the image variant (None = base).
    pub image_variant: Option<String>,
    pub images_dir: String,
}

impl Netboot {
    /// The NFS export CIDR, e.g. `192.168.1.0/24`.
    pub fn export_cidr(&self) -> String {
        format!("{}/{}", self.subnet, self.prefix_len)
    }
    /// The base URL of llmtune's artifact/iPXE server.
    pub fn http_base(&self) -> String {
        format!("http://{}:{}", self.server_ip, self.http_port)
    }
}

/// A network-facing token safe to interpolate into a config file / URL / argv: no
/// whitespace, control chars, quotes, or leading '-' (which an argv would read as
/// an option). Used for interface, IPs, and the OC profile.
fn safe_token(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('-')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | ':' | '_' | '-'))
}

/// The dnsmasq lease-file location is distro-dependent: Arch/Debian use
/// `/var/lib/misc/dnsmasq.leases`, Fedora/RHEL `/var/lib/dnsmasq/dnsmasq.leases`.
/// Default to whichever exists on this host (explicit `[netboot] leases` always
/// wins), so node auto-discovery doesn't silently come up empty on Fedora.
fn default_leases_path() -> String {
    default_leases_from(|p| std::path::Path::new(p).is_file())
}

/// Pure core of [`default_leases_path`] (existence probe injected).
fn default_leases_from(exists: impl Fn(&str) -> bool) -> String {
    const CANDIDATES: [&str; 2] = [
        "/var/lib/misc/dnsmasq.leases",    // Arch, Debian/Ubuntu
        "/var/lib/dnsmasq/dnsmasq.leases", // Fedora, RHEL
    ];
    CANDIDATES
        .iter()
        .find(|p| exists(p))
        .unwrap_or(&CANDIDATES[0])
        .to_string()
}

/// An absolute path with no newline/NUL - it reaches `sudo tee <path>` and config
/// bodies. Rejects `..` so a value can't traverse out of its intended tree.
fn safe_abs_path(s: &str) -> bool {
    s.starts_with('/')
        && !s.contains("..")
        && !s.contains('\n')
        && !s.contains('\0')
        && !s.contains('\r')
}

/// What network auto-detection found on this host: the default-route NIC and
/// its outbound IPv4.
struct DetectedNet {
    interface: String,
    ip: String,
}

/// Detect the host's boot-facing network: the default-route device from
/// `/proc/net/route`, plus the outbound-route IPv4 via `settings::lan_ip`
/// (a connected-UDP socket: same route, no packets sent).
fn detect_host_net() -> Option<DetectedNet> {
    let route = std::fs::read_to_string("/proc/net/route").ok()?;
    let interface = default_route_dev(&route)?;
    let ip = crate::settings::lan_ip()?;
    Some(DetectedNet { interface, ip })
}

/// The default-route device from `/proc/net/route` contents: the first UP
/// entry whose destination and mask are both 0.0.0.0.
fn default_route_dev(route: &str) -> Option<String> {
    const RTF_UP: u64 = 0x1;
    for line in route.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        // Iface Destination Gateway Flags RefCnt Use Metric Mask ...
        if f.len() >= 8
            && f[1] == "00000000"
            && f[7] == "00000000"
            && u64::from_str_radix(f[3], 16).is_ok_and(|flags| flags & RTF_UP != 0)
        {
            return Some(f[0].to_string());
        }
    }
    None
}

/// The network base address: `ip` with the host bits masked off
/// (e.g. 198.51.100.225/24 -> 198.51.100.0).
fn subnet_base(ip: std::net::Ipv4Addr, prefix_len: u8) -> std::net::Ipv4Addr {
    let mask = if prefix_len == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix_len.min(32)))
    };
    std::net::Ipv4Addr::from(u32::from(ip) & mask)
}

fn resolve_netboot(n: &NetbootCfg) -> Result<Netboot> {
    resolve_netboot_with(n, detect_host_net)
}

/// `detect` is the host-probe seam: production passes `detect_host_net`, tests
/// inject a fixed answer so resolution is host-independent.
fn resolve_netboot_with(
    n: &NetbootCfg,
    detect: impl FnOnce() -> Option<DetectedNet>,
) -> Result<Netboot> {
    // An empty string counts as absent: `interface = ""` means "detect", not a
    // (rejected) empty token.
    let explicit = |v: &Option<String>| v.clone().filter(|s| !s.is_empty());
    let (cfg_iface, cfg_ip) = (explicit(&n.interface), explicit(&n.server_ip));
    // Probe only when something is actually missing (subnet derives from
    // server_ip, so it alone never needs the probe). An explicit value always
    // wins over detection.
    let detected = if cfg_iface.is_none() || cfg_ip.is_none() {
        detect()
    } else {
        None
    };
    let interface = match cfg_iface {
        Some(v) => v,
        None => detected.as_ref().map(|d| d.interface.clone()).context(
            "netboot: interface not set and auto-detection failed (no default \
                 route) - set interface = \"<nic>\" under [netboot]",
        )?,
    };
    if !safe_token(&interface) {
        bail!("netboot: invalid interface '{interface}'");
    }
    let server_ip = match cfg_ip {
        Some(v) => v,
        None => detected.as_ref().map(|d| d.ip.clone()).context(
            "netboot: server_ip not set and auto-detection failed (no default \
             route) - set server_ip = \"<lan ipv4>\" under [netboot]",
        )?,
    };
    let server_v4 = server_ip
        .parse::<std::net::Ipv4Addr>()
        .map_err(|_| anyhow::anyhow!("netboot: server_ip '{server_ip}' is not an IPv4 address"))?;
    let prefix_len = n.prefix_len.unwrap_or(24);
    if prefix_len > 32 {
        bail!("netboot: prefix_len {prefix_len} out of range (0-32)");
    }
    let subnet = match explicit(&n.subnet) {
        Some(v) => v,
        None => subnet_base(server_v4, prefix_len).to_string(),
    };
    if subnet.parse::<std::net::Ipv4Addr>().is_err() {
        bail!("netboot: subnet '{subnet}' is not an IPv4 address");
    }
    let tftp_root = n.tftp_root.clone().unwrap_or_else(|| "/srv/tftp".into());
    let rootfs = n
        .rootfs
        .clone()
        .unwrap_or_else(|| "/srv/nfs/bc250-root".into());
    let kernel = n
        .kernel
        .clone()
        .unwrap_or_else(|| format!("{tftp_root}/vmlinuz"));
    let initrd = n
        .initrd
        .clone()
        .unwrap_or_else(|| format!("{tftp_root}/initramfs-nfs.img"));
    let oc_profile = n.oc_profile.clone().unwrap_or_else(|| "balanced".into());
    let leases = n.leases.clone().unwrap_or_else(default_leases_path);
    for (label, p) in [
        ("rootfs", &rootfs),
        ("tftp_root", &tftp_root),
        ("kernel", &kernel),
        ("initrd", &initrd),
        ("leases", &leases),
    ] {
        if !safe_abs_path(p) {
            bail!("netboot: {label} '{p}' must be an absolute path (no '..', no newlines)");
        }
    }
    if !matches!(oc_profile.as_str(), "eco" | "balanced" | "performance") {
        bail!("netboot: oc_profile '{oc_profile}' must be eco, balanced, or performance");
    }
    for o in &n.mac_ouis {
        if !safe_token(o) {
            bail!("netboot: invalid mac_oui '{o}'");
        }
    }
    let hostname_prefix = n.hostname_prefix.clone().unwrap_or_else(|| "bc250".into());
    if !safe_token(&hostname_prefix) {
        bail!("netboot: invalid hostname_prefix '{hostname_prefix}'");
    }
    let lan_cidr = n
        .lan_cidr
        .clone()
        .unwrap_or_else(|| format!("{subnet}/{prefix_len}"));
    // lan_cidr reaches firewall argv and the exports line: must parse as
    // <ipv4>/<prefix>.
    match lan_cidr.split_once('/') {
        Some((ip, plen))
            if ip.parse::<std::net::Ipv4Addr>().is_ok()
                && plen.parse::<u8>().map(|p| p <= 32).unwrap_or(false) => {}
        _ => bail!("netboot: lan_cidr '{lan_cidr}' is not an IPv4 CIDR (a.b.c.d/len)"),
    }
    let models_dir = n
        .models_dir
        .clone()
        .unwrap_or_else(|| DEFAULT_MODELS_DIR.into());
    if !safe_abs_path(&models_dir) {
        bail!("netboot: models_dir '{models_dir}' must be an absolute path");
    }
    let work_dir = n
        .work_dir
        .clone()
        .unwrap_or_else(|| "/var/lib/llmtune/netboot".into());
    for (label, p) in [("iso", &n.iso), ("ssh_pubkey", &n.ssh_pubkey)] {
        if let Some(p) = p {
            if !safe_abs_path(p) {
                bail!("netboot: {label} '{p}' must be an absolute path (no '..', no newlines)");
            }
        }
    }
    if !safe_abs_path(&work_dir) {
        bail!("netboot: work_dir '{work_dir}' must be an absolute path");
    }
    let ttm_gtt_gb = n.ttm_gtt_gb.unwrap_or(12);
    if !(1..=16).contains(&ttm_gtt_gb) {
        bail!("netboot: ttm_gtt_gb {ttm_gtt_gb} out of range (1-16 on a 16 GiB board)");
    }
    if let Some(f) = &n.flake {
        if !safe_abs_path(f) {
            bail!("netboot: flake '{f}' must be an absolute path (no '..', no newlines)");
        }
    }
    let nix_cmd = n.nix_cmd.clone().unwrap_or_else(|| "nix".into());
    // nix_cmd becomes an argv[0]; allow `nix`, an absolute path, or a
    // flake-relative wrapper like `.tool/np` - but nothing shell-shaped.
    if nix_cmd.is_empty()
        || nix_cmd.starts_with('-')
        || nix_cmd.contains("..")
        || !nix_cmd
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-'))
    {
        bail!("netboot: invalid nix_cmd '{nix_cmd}'");
    }
    // image_variant is spliced into a nix flake attr name (`.#netbootKernel<v>`),
    // so it must be a plain identifier tail. Empty/whitespace means unset.
    let image_variant = match n.image_variant.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(v) => {
            if !v
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
            {
                bail!("netboot: invalid image_variant '{v}' (alphanumeric, '_', '-' only)");
            }
            Some(v.to_string())
        }
    };
    let images_dir = n
        .images_dir
        .clone()
        .unwrap_or_else(|| format!("{work_dir}/images"));
    if !safe_abs_path(&images_dir) {
        bail!("netboot: images_dir '{images_dir}' must be an absolute path");
    }
    Ok(Netboot {
        interface,
        server_ip,
        subnet,
        prefix_len,
        http_port: n.http_port.unwrap_or(8090),
        rootfs,
        tftp_root,
        kernel,
        initrd,
        oc_profile,
        leases,
        mac_ouis: n.mac_ouis.iter().map(|o| o.to_lowercase()).collect(),
        hostname_prefix,
        lan_cidr,
        models_dir,
        dnsmasq: n.dnsmasq.unwrap_or(false),
        console_port: n.console_port.unwrap_or(6666),
        iso: n.iso.clone(),
        work_dir,
        ssh_pubkey: n.ssh_pubkey.clone(),
        ttm_gtt_gb,
        flake: n.flake.clone(),
        nix_cmd,
        image_variant,
        images_dir,
    })
}

/// A fully-resolved node (defaults + env + hardcoded fallbacks applied).
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub host: Option<String>,
    pub transport: Transport,
    pub ssh_user: Option<String>,
    pub ssh_key: Option<String>,
    pub models_dir: String,
    pub llama_unit: String,
    pub llama_url: String,
    /// Smart-plug hook to cold power-cycle the board (see `NodeCfg::power_cmd`).
    pub power_cmd: Option<String>,
}

/// The standard models directory. Fixed under llmtune's home so it sits next
/// to the managed builds (`/var/lib/llmtune/{models,builds}`) - one obvious,
/// namespaced place regardless of how llmtune is invoked (not privilege- or
/// $HOME-dependent, so a user run and the internal sudo agree). `setup` creates
/// it and hands ownership to the invoking user so models can be dropped without
/// root. Override with $LLMTUNE_MODELS_DIR or `models_dir` in fleet.toml.
pub const DEFAULT_MODELS_DIR: &str = "/var/lib/llmtune/models";

fn default_models_dir() -> String {
    std::env::var("LLMTUNE_MODELS_DIR").unwrap_or_else(|_| DEFAULT_MODELS_DIR.to_string())
}

fn config_path() -> Option<PathBuf> {
    crate::paths::config_file("fleet.toml")
}

#[derive(Debug, Clone)]
pub struct Config {
    pub nodes: Vec<Node>,
    pub clusters: Vec<ClusterCfg>,
    pub netboot: Option<Netboot>,
}

impl Config {
    /// Load the fleet config from `path` (or the default location). If neither
    /// exists, synthesize a single `localhost` node.
    pub fn load(path: Option<&str>) -> Result<Config> {
        let p = match path {
            Some(p) => Some(PathBuf::from(p)),
            None => config_path(),
        };
        let explicit = path.is_some();
        let raw = match &p {
            Some(p) if p.is_file() => {
                let txt = std::fs::read_to_string(p)
                    .with_context(|| format!("reading {}", p.display()))?;
                toml::from_str::<RawConfig>(&txt)
                    .with_context(|| format!("parsing {}", p.display()))?
            }
            // An explicitly-passed --config that doesn't exist is an error, not a
            // silent fall-through to the synthesized localhost node.
            Some(p) if explicit => {
                bail!("config file not found: {}", p.display());
            }
            _ => RawConfig::default(),
        };
        Self::from_raw(raw)
    }

    /// Parse a config from a TOML string (used by tests and config generators).
    pub fn load_str(txt: &str) -> Result<Config> {
        let raw: RawConfig = toml::from_str(txt).context("parsing config")?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawConfig) -> Result<Config> {
        let d = &raw.defaults;
        let def_models = d.models_dir.clone().unwrap_or_else(default_models_dir);
        let def_unit = d
            .llama_unit
            .clone()
            .unwrap_or_else(|| "llama-server.service".to_string());
        let def_url = d
            .llama_url
            .clone()
            .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());

        let mut nodes: Vec<Node> = Vec::with_capacity(raw.nodes.len());
        for n in &raw.nodes {
            let transport = Transport::parse(n.transport.as_deref())
                .with_context(|| format!("node '{}'", n.name))?;
            // ssh builds `[options] user@host` - a host/user starting with '-'
            // would be parsed by ssh as an option (e.g. -oProxyCommand=...).
            if n.host.as_deref().is_some_and(|h| h.starts_with('-')) {
                bail!("node '{}': host may not start with '-'", n.name);
            }
            if n.ssh_user.as_deref().is_some_and(|u| u.starts_with('-')) {
                bail!("node '{}': ssh_user may not start with '-'", n.name);
            }
            let llama_unit = n.llama_unit.clone().unwrap_or_else(|| def_unit.clone());
            if !valid_unit_name(&llama_unit) {
                bail!(
                    "node '{}': invalid llama_unit '{llama_unit}' \
                     (must be a systemd unit name ending in .service, no '/' or '..')",
                    n.name
                );
            }
            // power_cmd runs through `sh -c` on the control host; an
            // empty/whitespace value is a config mistake, not a manual plan.
            if n.power_cmd.as_deref().is_some_and(|c| c.trim().is_empty()) {
                bail!(
                    "node '{}': power_cmd is empty (omit it for a manual cold cycle)",
                    n.name
                );
            }
            nodes.push(Node {
                name: n.name.clone(),
                host: n.host.clone(),
                transport,
                ssh_user: n.ssh_user.clone(),
                ssh_key: n.ssh_key.clone(),
                models_dir: n.models_dir.clone().unwrap_or_else(|| def_models.clone()),
                llama_unit,
                llama_url: n.llama_url.clone().unwrap_or_else(|| def_url.clone()),
                power_cmd: n.power_cmd.clone(),
            });
        }

        if nodes.is_empty() {
            if !valid_unit_name(&def_unit) {
                bail!("invalid default llama_unit '{def_unit}' (must end in .service, no '/' or '..')");
            }
            nodes.push(Node {
                name: "localhost".to_string(),
                host: None,
                transport: Transport::Local,
                ssh_user: None,
                ssh_key: None,
                models_dir: def_models,
                llama_unit: def_unit,
                llama_url: def_url,
                power_cmd: None,
            });
        }
        // rpc_bin (the remote command systemd-run runs) and rpc_bind (an -H
        // value) reach a `sudo systemd-run` argv; a leading '-' would be parsed as
        // an option (arbitrary systemd-run property = root command execution).
        for c in &raw.clusters {
            if c.rpc_bin
                .as_deref()
                .is_some_and(|b| b.is_empty() || b.starts_with('-'))
            {
                bail!(
                    "cluster '{}': rpc_bin may not be empty or start with '-'",
                    c.name
                );
            }
            if c.rpc_bind
                .as_deref()
                .is_some_and(|b| b.is_empty() || b.starts_with('-'))
            {
                bail!(
                    "cluster '{}': rpc_bind may not be empty or start with '-'",
                    c.name
                );
            }
        }
        let netboot = match &raw.netboot {
            Some(n) => Some(resolve_netboot(n)?),
            None => None,
        };
        Ok(Config {
            nodes,
            clusters: raw.clusters,
            netboot,
        })
    }

    /// The local node: the first `Local`-transport node, else the first node.
    pub fn local_node(&self) -> &Node {
        self.nodes
            .iter()
            .find(|n| n.transport == Transport::Local)
            .unwrap_or(&self.nodes[0])
    }

    pub fn node(&self, name: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.name == name)
    }

    pub fn cluster(&self, name: &str) -> Option<&ClusterCfg> {
        self.clusters.iter().find(|c| c.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesizes_localhost_when_empty() {
        // An empty config synthesizes a single local node.
        let c = Config::load_str("").unwrap();
        assert_eq!(c.nodes.len(), 1);
        assert_eq!(c.local_node().transport, Transport::Local);
        assert_eq!(c.local_node().llama_url, "http://127.0.0.1:8080");
    }

    #[test]
    fn explicit_missing_config_errors() {
        // An explicitly-passed path that doesn't exist must error, not fall
        // through to a synthesized localhost (which would actuate the wrong box).
        let err = Config::load(Some("/nonexistent/llmtune-fleet.toml")).unwrap_err();
        assert!(err.to_string().contains("config file not found"));
    }

    #[test]
    fn unknown_transport_errors() {
        let err = Config::load_str("[[node]]\nname=\"a\"\ntransport=\"shh\"\n").unwrap_err();
        assert!(format!("{err:#}").contains("unknown transport"));
    }

    #[test]
    fn known_transports_and_default_parse() {
        assert_eq!(Transport::parse(None).unwrap(), Transport::Local);
        assert_eq!(Transport::parse(Some("SSH")).unwrap(), Transport::Ssh);
        assert_eq!(Transport::parse(Some("agent")).unwrap(), Transport::Agent);
    }

    #[test]
    fn cluster_rpc_bin_bind_leading_dash_rejected() {
        let bad_bin = "[[cluster]]\nname=\"c\"\nhead=\"localhost\"\nrpc_port=50052\nrpc_bin=\"--upload-pack=x\"\n";
        assert!(format!("{:#}", Config::load_str(bad_bin).unwrap_err()).contains("rpc_bin"));
        let bad_bind =
            "[[cluster]]\nname=\"c\"\nhead=\"localhost\"\nrpc_port=50052\nrpc_bind=\"-oX\"\n";
        assert!(format!("{:#}", Config::load_str(bad_bind).unwrap_err()).contains("rpc_bind"));
        // a normal cluster still parses
        let ok =
            "[[cluster]]\nname=\"c\"\nhead=\"localhost\"\nrpc_port=50052\nrpc_bin=\"rpc-server\"\n";
        assert!(Config::load_str(ok).is_ok());
    }

    fn netboot_cfg(toml: &str) -> NetbootCfg {
        toml::from_str::<RawConfig>(toml).unwrap().netboot.unwrap()
    }

    fn fake_detect() -> Option<DetectedNet> {
        Some(DetectedNet {
            interface: "enp6s0".into(),
            ip: "198.51.100.225".into(),
        })
    }

    #[test]
    fn subnet_base_masks_host_bits() {
        let ip: std::net::Ipv4Addr = "198.51.100.225".parse().unwrap();
        assert_eq!(subnet_base(ip, 24).to_string(), "198.51.100.0");
        assert_eq!(subnet_base(ip, 16).to_string(), "198.51.0.0");
        assert_eq!(subnet_base(ip, 8).to_string(), "198.0.0.0");
        assert_eq!(subnet_base(ip, 32).to_string(), "198.51.100.225");
        assert_eq!(subnet_base(ip, 0).to_string(), "0.0.0.0");
        // a non-octet-aligned prefix masks mid-octet
        let ip: std::net::Ipv4Addr = "10.0.0.130".parse().unwrap();
        assert_eq!(subnet_base(ip, 25).to_string(), "10.0.0.128");
    }

    #[test]
    fn default_leases_prefers_an_existing_lease_file() {
        // Arch/Debian layout present -> Arch/Debian path.
        assert_eq!(
            default_leases_from(|p| p == "/var/lib/misc/dnsmasq.leases"),
            "/var/lib/misc/dnsmasq.leases"
        );
        // Fedora layout present -> Fedora path.
        assert_eq!(
            default_leases_from(|p| p == "/var/lib/dnsmasq/dnsmasq.leases"),
            "/var/lib/dnsmasq/dnsmasq.leases"
        );
        // Neither exists yet (fresh box, dnsmasq not started): the
        // Arch/Debian default, same as before.
        assert_eq!(
            default_leases_from(|_| false),
            "/var/lib/misc/dnsmasq.leases"
        );
    }

    #[test]
    fn default_route_dev_parses_proc_net_route() {
        // A real-shaped /proc/net/route: header, the default route, a link route.
        let route =
            "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n\
             enp2s0f0np0\t00000000\t0101A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0\n\
             enp2s0f0np0\t0001A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\t0\t0\t0\n";
        assert_eq!(default_route_dev(route).as_deref(), Some("enp2s0f0np0"));
        // No default route: only the link route remains.
        let no_default =
            "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n\
             enp2s0f0np0\t0001A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\t0\t0\t0\n";
        assert_eq!(default_route_dev(no_default), None);
        // A downed default route (RTF_UP clear) doesn't count.
        let down =
            "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n\
             enp2s0f0np0\t00000000\t0101A8C0\t0002\t0\t0\t100\t00000000\t0\t0\t0\n";
        assert_eq!(default_route_dev(down), None);
        assert_eq!(default_route_dev(""), None);
    }

    #[test]
    fn netboot_empty_section_autodetects() {
        // An empty [netboot] resolves entirely from detection: interface and
        // server_ip from the probe, subnet derived by masking (default /24).
        let cfg = netboot_cfg("[netboot]\n");
        let nb = resolve_netboot_with(&cfg, fake_detect).unwrap();
        assert_eq!(nb.interface, "enp6s0");
        assert_eq!(nb.server_ip, "198.51.100.225");
        assert_eq!(nb.subnet, "198.51.100.0");
        assert_eq!(nb.lan_cidr, "198.51.100.0/24");
        assert_eq!(nb.export_cidr(), "198.51.100.0/24");
    }

    #[test]
    fn netboot_models_dir_only_autodetects() {
        // The motivating case: [netboot] with just models_dir parses + resolves.
        let cfg = netboot_cfg("[netboot]\nmodels_dir = \"/srv/llmtune/models\"\n");
        let nb = resolve_netboot_with(&cfg, fake_detect).unwrap();
        assert_eq!(nb.models_dir, "/srv/llmtune/models");
        assert_eq!(nb.server_ip, "198.51.100.225");
        assert_eq!(nb.subnet, "198.51.100.0");
    }

    #[test]
    fn netboot_explicit_values_beat_detection() {
        // All three set: the probe must not run at all (it panics if it does),
        // and the explicit values come through untouched.
        let cfg = netboot_cfg(
            "[netboot]\ninterface = \"eth9\"\nserver_ip = \"10.1.2.3\"\nsubnet = \"10.1.2.0\"\n",
        );
        let nb = resolve_netboot_with(&cfg, || panic!("detection must not run")).unwrap();
        assert_eq!(nb.interface, "eth9");
        assert_eq!(nb.server_ip, "10.1.2.3");
        assert_eq!(nb.subnet, "10.1.2.0");
        // Partially set: explicit server_ip wins over the detected one; the
        // absent interface is detected; subnet derives from the EXPLICIT ip.
        let cfg = netboot_cfg("[netboot]\nserver_ip = \"10.1.2.3\"\n");
        let nb = resolve_netboot_with(&cfg, fake_detect).unwrap();
        assert_eq!(nb.interface, "enp6s0");
        assert_eq!(nb.server_ip, "10.1.2.3");
        assert_eq!(nb.subnet, "10.1.2.0");
    }

    #[test]
    fn netboot_image_variant_backcompat_and_resolution() {
        // Back-compat: a [netboot] section without image_variant parses and
        // resolves with the field None (the base flake outputs).
        let cfg = netboot_cfg("[netboot]\n");
        let nb = resolve_netboot_with(&cfg, fake_detect).unwrap();
        assert_eq!(nb.image_variant, None);
        // Set: the suffix comes through verbatim.
        let cfg = netboot_cfg("[netboot]\nimage_variant = \"Llmtune\"\n");
        let nb = resolve_netboot_with(&cfg, fake_detect).unwrap();
        assert_eq!(nb.image_variant.as_deref(), Some("Llmtune"));
        // Empty / whitespace-only normalizes to None.
        let cfg = netboot_cfg("[netboot]\nimage_variant = \"  \"\n");
        let nb = resolve_netboot_with(&cfg, fake_detect).unwrap();
        assert_eq!(nb.image_variant, None);
        // A non-identifier variant (would corrupt the nix attr name) errors.
        let cfg = netboot_cfg("[netboot]\nimage_variant = \"Llm tune;rm\"\n");
        let err = format!("{:#}", resolve_netboot_with(&cfg, fake_detect).unwrap_err());
        assert!(err.contains("invalid image_variant"), "got: {err}");
    }

    #[test]
    fn netboot_detection_failure_is_a_clear_error() {
        // No default route: resolution errors telling the user which field to
        // set, rather than fabricating a value.
        let cfg = netboot_cfg("[netboot]\n");
        let err = format!("{:#}", resolve_netboot_with(&cfg, || None).unwrap_err());
        assert!(err.contains("auto-detection failed"), "got: {err}");
        assert!(err.contains("interface"), "got: {err}");
    }

    #[test]
    fn netboot_detected_values_still_validated() {
        // Detection output goes through the same safety gates as config input.
        let cfg = netboot_cfg("[netboot]\n");
        let bad_iface = || {
            Some(DetectedNet {
                interface: "eth0; rm -rf /".into(),
                ip: "198.51.100.225".into(),
            })
        };
        let err = format!("{:#}", resolve_netboot_with(&cfg, bad_iface).unwrap_err());
        assert!(err.contains("invalid interface"), "got: {err}");
        let bad_ip = || {
            Some(DetectedNet {
                interface: "eth0".into(),
                ip: "fe80::1".into(),
            })
        };
        let err = format!("{:#}", resolve_netboot_with(&cfg, bad_ip).unwrap_err());
        assert!(err.contains("not an IPv4 address"), "got: {err}");
    }

    #[test]
    fn invalid_unit_name_rejected() {
        assert!(valid_unit_name("llama-server.service"));
        assert!(valid_unit_name("llama@bc250.service"));
        assert!(!valid_unit_name("../etc/evil.service"));
        assert!(!valid_unit_name("foo/bar.service"));
        assert!(!valid_unit_name("noext"));
        let err =
            Config::load_str("[[node]]\nname=\"a\"\nllama_unit=\"../x.service\"\n").unwrap_err();
        assert!(err.to_string().contains("invalid llama_unit"));
    }
}
