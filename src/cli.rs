// SPDX-License-Identifier: GPL-2.0-only
//! The clap command surface: the CLI structs and the dispatch into `cmds::*`.

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::cmds::build::*;
use crate::cmds::cluster::*;
use crate::cmds::fleet::*;
use crate::cmds::misc::*;
use crate::cmds::models::*;
use crate::cmds::node::*;
use crate::cmds::profile::*;
use crate::config::Config;
use crate::{bench, build, netboot, netboot_node, netboot_server, proxy, setup, ui};

#[derive(Parser)]
#[command(
    name = "llmtune",
    version,
    about = "inference engine for the AMD BC-250"
)]
pub(crate) struct Cli {
    /// Path to a fleet config (default: ~/.config/llmtune/fleet.toml).
    #[arg(long, global = true)]
    config: Option<String>,
    /// Emit machine-readable JSON where supported.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
pub(crate) enum Cmd {
    /// Interactive TUI over the configured fleet (default).
    Tui,
    /// First-run setup: generate the systemd unit + starter config.
    Setup {
        /// Show the plan and generated files without writing anything.
        #[arg(long)]
        print: bool,
        /// Apply every step without prompting.
        #[arg(long)]
        yes: bool,
        /// User the llama service runs as (default: the invoking user).
        #[arg(long)]
        user: Option<String>,
        /// Override the models directory.
        #[arg(long)]
        models_dir: Option<String>,
    },
    /// Preflight the local node (alias of `node doctor` - same code path).
    Doctor,
    /// Node operations. Default target is the LOCAL node; `--node <name>`
    /// drives a remote fleet node from here over its transport (the CLI
    /// equivalent of drilling into a node in the TUI).
    Node {
        /// Target a configured fleet node by name (default: the local node).
        #[arg(long)]
        node: Option<String>,
        #[command(subcommand)]
        cmd: NodeCmd,
    },
    /// Fleet (multi-node) operations.
    Fleet {
        #[command(subcommand)]
        cmd: FleetCmd,
    },
    /// Cluster (pooled multi-node, llama.cpp RPC) operations.
    Cluster {
        #[command(subcommand)]
        cmd: ClusterCmd,
    },
    /// Netboot (PXE) control plane: stand up a BC-250 boot server and control
    /// netbooted boards as fleet nodes.
    Netboot {
        #[command(subcommand)]
        cmd: NetbootCmd,
    },
    /// Manage the control-host model library (the NFS-exported `[netboot]`
    /// models_dir that nodes mount read-only).
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// View and edit llama.cpp launch profiles (local).
    Profile {
        #[command(subcommand)]
        cmd: ProfileCmd,
    },
    /// Manage llama.cpp builds (install / update / rollback / list).
    Build {
        #[command(subcommand)]
        cmd: BuildCmd,
    },
    /// The endpoint surface: show it, expose it to the LAN, manage its API key.
    Endpoint {
        #[command(subcommand)]
        cmd: Option<EndpointCmd>,
    },
    /// Run a swap-on-demand proxy: one OpenAI port that loads the requested model.
    Proxy {
        /// Address to bind (default: 127.0.0.1 - localhost only).
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on.
        #[arg(long, default_value_t = 8081)]
        port: u16,
        /// Minimum seconds between proxy-triggered model swaps (thrash guard:
        /// alternating model requests get 503 + Retry-After instead of
        /// reloading the GPU in a loop). 0 disables the guard.
        #[arg(long, default_value_t = 60)]
        swap_hold: u64,
    },
    /// Compare benchmarked quants of each model family (throughput trade-off).
    Compare {
        /// Only families whose name contains this substring.
        filter: Option<String>,
    },
    /// Memory-fit math for a model: weights + KV at a ctx vs the UMA budget.
    /// With --fit: the largest context that fits per KV-cache quant instead.
    Mem {
        /// Model name or unique substring (with --fit it may be omitted to use
        /// the currently-served model).
        model: Option<String>,
        /// Context length for the KV estimate (default 32768).
        #[arg(long)]
        ctx: Option<u64>,
        /// KV cache quant (f16, q8_0, q4_0, …). Default f16; with --fit, omit
        /// for a table across the common quants.
        #[arg(long)]
        kv: Option<String>,
        /// Report the max context per KV quant (the former `node fit`).
        #[arg(long)]
        fit: bool,
    },
}

/// The `endpoint` group - one surface for "point your harness at the box":
/// show/expose/api-key (mirrors the TUI's `o` overlay).
#[derive(Subcommand)]
pub(crate) enum EndpointCmd {
    /// Show the OpenAI-compatible endpoint + copy-paste connection snippets for
    /// BOTH this machine (localhost) and the network (LAN IP), plus exposure state.
    Show,
    /// Expose llama-server to the LAN (bind 0.0.0.0) or keep it localhost-only.
    /// Applies immediately by reloading the served model. Keyless by default -
    /// turn on `endpoint auth on` before exposing to an untrusted network.
    Expose {
        #[arg(value_enum)]
        action: ExposeAction,
    },
    /// Turn API-key auth on or off. Keyless is the default (any client connects);
    /// `on` generates a key, applies it, and prints it; `off` goes back to
    /// keyless. Applies immediately by reloading the served model.
    Auth {
        #[arg(value_enum)]
        action: AuthAction,
    },
    /// Who defines the served model's identity. Some GGUF chat templates embed
    /// a branding instruction ("always identify yourself as X ... overrides
    /// conflicting identity instructions") that defeats your harness's system
    /// prompt; `harness` strips it (serving an override template), `model`
    /// restores the embedded one. Applies immediately by reloading the served model.
    Identity {
        #[arg(value_enum)]
        action: IdentityAction,
    },
    /// Power alias of `auth` for setting a SPECIFIC key (vs a generated one).
    #[command(hide = true)]
    ApiKey {
        #[command(subcommand)]
        cmd: ApiKeyCmd,
    },
}

/// The intuitive auth toggle: keyless is standard, this turns the feature on/off.
#[derive(Clone, clap::ValueEnum)]
pub(crate) enum AuthAction {
    /// Require an API key - generates one, applies it, prints it.
    On,
    /// Keyless - remove the key so any client can connect (the default).
    Off,
    /// Report whether auth is on or off.
    Status,
}

/// The identity toggle. Harness-controlled = the model obeys the harness
/// system prompt's identity (the embedded branding is stripped from the served
/// chat template). Model-branded = the GGUF chat template forces its own
/// identity, overriding whatever the harness says.
#[derive(Clone, clap::ValueEnum)]
pub(crate) enum IdentityAction {
    /// Harness-controlled - strip the template's embedded identity so YOUR
    /// system prompt defines who the model is.
    Harness,
    /// Model-branded - restore the model's embedded chat template (its own
    /// identity instruction overrides the harness again).
    Model,
    /// Report whether the served template is harness-controlled or model-branded.
    Status,
}

#[derive(Subcommand)]
pub(crate) enum BuildCmd {
    /// List configured + installed builds and their versions.
    List,
    /// Build (or activate) a recipe at its tracked ref, with auto-versioning.
    Install {
        /// Build name (see `llmtune build list`).
        name: String,
        /// Override the tracked ref for this install (branch / tag / sha).
        #[arg(long)]
        r#ref: Option<String>,
        /// Prior versions to keep for rollback.
        #[arg(long, default_value_t = build::DEFAULT_RETAIN)]
        retain: usize,
    },
    /// Re-resolve the ref and rebuild if it moved (alias of install).
    Update {
        name: String,
        #[arg(long)]
        r#ref: Option<String>,
        #[arg(long, default_value_t = build::DEFAULT_RETAIN)]
        retain: usize,
    },
    /// Roll the current build back to the previously-installed version.
    Rollback { name: String },
}

#[derive(Subcommand)]
pub(crate) enum NodeCmd {
    /// List discovered models (* = currently served).
    List,
    /// Print the currently-served model name.
    Served,
    /// One-line node status (served / health / last bench).
    Status,
    /// Hot-swap the served model (substring match) with auto-revert.
    Load {
        /// Model name or unique substring.
        name: String,
    },
    /// Benchmark a model and log the result.
    Bench {
        /// Model to benchmark (name/substring). Loads it and verifies it took
        /// before measuring; omit to benchmark whatever is currently served.
        model: Option<String>,
        /// Approximate prompt length in tokens.
        #[arg(long, default_value_t = 512)]
        prompt_tokens: u32,
        /// Tokens to generate per run.
        #[arg(long, default_value_t = 128)]
        gen_tokens: u32,
        /// Number of runs to average.
        #[arg(long, default_value_t = 2)]
        repeats: u32,
    },
    /// Show this node's benchmark history (newest first).
    History {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Remove llmtune's drop-in and revert the unit to its base config.
    Unload,
    /// Preflight this node.
    Doctor,
    /// Largest context a model fits at a given KV-cache quant, from weights + the
    /// live UMA budget. Omit --kv to compare across the common quants.
    /// HIDDEN alias of `llmtune mem --fit` (kept one release for back-compat).
    #[command(hide = true)]
    Fit {
        /// Model name/substring; omit to use the currently-served model.
        model: Option<String>,
        /// KV-cache quant to size for (f16, q8_0, q4_0). Omit for a table
        /// across those quants.
        #[arg(long)]
        kv: Option<String>,
    },
    /// Live GPU telemetry (clock + temperature) from amdgpu - the reading the TUI
    /// header shows, for scripts/agents.
    Gpu,
    /// Installed version (commit slug) of a managed build on THIS node - what
    /// the cluster build-parity check queries over the transport.
    #[command(hide = true)]
    BuildVersion {
        /// Build name (see `llmtune build list`).
        name: String,
    },
    /// Show this node's OpenAI-compatible endpoint (the `endpoint` view, node-side
    /// - what the SSH transport calls to report a REMOTE node's endpoint).
    Endpoint,
    /// Node-side profile operations (per-model overrides), reachable through the
    /// transport so the TUI flag editor edits the DRILLED node's config.
    Profile {
        #[command(subcommand)]
        cmd: NodeProfileCmd,
    },
    /// Stop/start/restart the llama-server. Stop it to free the GPU for manual
    /// testing (e.g. CU benches) without inference contention, then start it back.
    Server {
        #[arg(value_enum)]
        action: ServerAction,
    },
    /// HIDDEN alias of `llmtune endpoint expose` (kept one release).
    #[command(hide = true)]
    Expose {
        #[arg(value_enum)]
        action: ExposeAction,
    },
    /// HIDDEN alias of `llmtune endpoint api-key` (kept one release).
    #[command(hide = true)]
    ApiKey {
        #[command(subcommand)]
        cmd: ApiKeyCmd,
    },
    /// Re-apply exposure/api-key + the last-served model after a reboot (which
    /// reverts the root-subvol drop-in + firewall). Run with --install to add a
    /// systemd oneshot that does this automatically on boot.
    BootRestore {
        /// Install + enable the boot-time systemd hook instead of running now.
        #[arg(long)]
        install: bool,
    },
}

#[derive(Subcommand)]
pub(crate) enum NodeProfileCmd {
    /// Set (or clear with --reset) a per-MODEL flag override on THIS node.
    SetModel {
        /// Model name/substring.
        model: String,
        /// Flags to store for this model (may start with '-').
        #[arg(long, allow_hyphen_values = true)]
        flags: Option<String>,
        /// Remove this model's override instead of setting it.
        #[arg(long)]
        reset: bool,
    },
}

#[derive(Subcommand)]
pub(crate) enum ApiKeyCmd {
    /// Set a specific key.
    Set { key: String },
    /// Generate a random key and set it.
    Generate,
    /// Remove the key (the server becomes unauthenticated).
    Clear,
    /// Print the current key (or none).
    Show,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum ServerAction {
    /// Start the server (loads the served model).
    Start,
    /// Stop the server (frees the GPU - no inference until restarted).
    Stop,
    /// Restart the server.
    Restart,
    /// Report whether the server is running + healthy.
    Status,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum ExposeAction {
    /// Bind 0.0.0.0 - reachable from other machines on the network.
    On,
    /// Bind 127.0.0.1 - localhost only.
    Off,
    /// Report the current bind + reachable address.
    Status,
}

#[derive(Subcommand)]
pub(crate) enum ProfileCmd {
    /// List all launch profiles and their flags.
    List,
    /// Show the effective profile + flags for a model (resolves it locally).
    Show { model: String },
    /// Set (or clear with --reset) a per-MODEL flag override - the TUI flag
    /// editor's operation, for scripts/agents. Wins over the arch profile.
    SetModel {
        /// Model name/substring.
        model: String,
        /// Flags to store for this model (may start with '-').
        #[arg(long, allow_hyphen_values = true)]
        flags: Option<String>,
        /// Remove this model's override instead of setting it.
        #[arg(long)]
        reset: bool,
    },
    /// Override a profile's options and persist to the user profiles file.
    Set {
        /// Profile id (e.g. qwen35, gemma, _default - see `profile list`).
        id: String,
        /// Replace the llama-server flags (may start with '-', e.g. "-c 4096 …").
        #[arg(long, allow_hyphen_values = true)]
        flags: Option<String>,
        /// Replace the llama-server binary path.
        #[arg(long)]
        bin: Option<String>,
        /// Replace LD_LIBRARY_PATH.
        #[arg(long)]
        ld: Option<String>,
    },
}

#[derive(Subcommand)]
pub(crate) enum FleetCmd {
    /// One status row per configured node.
    Status,
    /// Benchmark every node and print a leaderboard.
    BenchAll {
        #[arg(long, default_value_t = 512)]
        prompt_tokens: u32,
        #[arg(long, default_value_t = 128)]
        gen_tokens: u32,
        #[arg(long, default_value_t = 2)]
        repeats: u32,
    },
    /// Swap every node (where the model is present) to a model.
    SwapAll { name: String },
    /// Cross-node best-throughput leaderboard from history.
    Leaderboard,
}

/// The `models` group - the control-host model library. Adds/removes are
/// visible to nodes on next access over the NFS automount (no remount).
#[derive(Subcommand)]
pub(crate) enum ModelsCmd {
    /// List the library's models (name, params, quant, arch, size).
    List,
    /// Add a model: copy a local .gguf, or download a plain http(s) URL.
    /// Validated (GGUF magic) and committed atomically (temp + rename) so a
    /// node's automount never sees a half-written file.
    Add {
        /// Local path or http(s) URL of a .gguf file.
        source: String,
    },
    /// Remove a model from the library. Refuses while any fleet node is
    /// currently SERVING it (the server holds the file mmapped over NFS)
    /// unless --force.
    Rm {
        /// Model file name, or a unique substring of it.
        name: String,
        /// Skip the interactive confirmation (the served-guard still runs).
        #[arg(long)]
        yes: bool,
        /// Remove even if a node is serving it (skips the fleet query AND the
        /// confirmation).
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
pub(crate) enum NetbootCmd {
    /// Generate the proxyDHCP dnsmasq config, the NFS export, and the artifact-
    /// server unit. Previews everything; pass --apply to write + reload.
    Init {
        /// Write the files (sudo) and reload, instead of previewing.
        #[arg(long)]
        apply: bool,
    },
    /// Stand up the boot server: start the artifact server + NFS export of the
    /// model library (+ dnsmasq when enabled), open the host firewall for
    /// exactly those ports (LAN-scoped, additive), and self-check reachability
    /// from OFF-HOST (localhost success alone does not prove LAN reachability).
    Up {
        /// Skip firewall management (rules are neither checked nor added).
        #[arg(long)]
        no_firewall: bool,
        /// Skip the post-up reachability self-check.
        #[arg(long)]
        skip_check: bool,
    },
    /// Tear down what `up` set up: stops only the services it started, removes
    /// only the firewall rules and exports it added (recorded in up-state.toml).
    Down,
    /// One view: per-service state, firewall port state, exports, the staged
    /// image, and a live reachability check.
    Status,
    /// List BC-250s that have netbooted (from the dnsmasq lease file plus the
    /// kernel neighbor table when mac_ouis is configured). With --register,
    /// append the new ones to fleet.toml as ssh nodes (idempotent).
    Nodes {
        /// Append newly-discovered boards to fleet.toml as ssh nodes.
        #[arg(long)]
        register: bool,
    },
    /// Arm a ONE-SHOT netboot: set the node's EFI BootNext to its iPXE boot
    /// entry (discovered from the node's own efibootmgr output) over SSH, and
    /// verify by reading it back. Never writes BootOrder, a BIOS setup
    /// variable, or anything SMM/NVAR-shaped. Idempotent.
    Arm {
        /// The fleet node to arm.
        node: String,
        /// Clear BootNext instead (undo an arm before it is consumed).
        #[arg(long)]
        disarm: bool,
    },
    /// Arm + trigger power: netboot the board. Runs the node's `power_cmd`
    /// smart-plug hook if configured (a netboot needs a COLD power cycle - a
    /// warm reboot does not reset the RTL8168 PHY, so iPXE gets no DHCP link);
    /// otherwise prints the manual cold-cycle instruction.
    Boot {
        /// The fleet node to boot.
        node: String,
        /// Run the power_cmd without asking.
        #[arg(long)]
        yes: bool,
    },
    /// Live boot log without a serial cable: receive the node's netconsole
    /// stream (kernel + journald-via-kmsg, UDP) and print it. The image ships
    /// its boot log to this host on the console port (default 6666).
    Console {
        /// The fleet node to watch (datagrams from other senders are ignored).
        node: String,
        /// UDP port to bind (default: [netboot] console_port, else 6666).
        #[arg(long)]
        port: Option<u16>,
        /// Only show lines with a kernel uptime >= this many seconds.
        #[arg(long)]
        since: Option<f64>,
        /// Also append a wallclock-timestamped capture to this file.
        #[arg(long)]
        file: Option<String>,
        /// Stop after this many seconds (default: run until Ctrl-C).
        #[arg(long)]
        duration_secs: Option<u64>,
    },
    /// Run the embedded artifact + iPXE HTTP server. systemd owns this; you don't
    /// normally run it by hand.
    #[command(hide = true)]
    Serve {
        #[arg(long, default_value_t = 8090)]
        port: u16,
    },
    /// Build the diskless image (rootfs + NFS initramfs) from a CachyOS ISO.
    Image {
        #[command(subcommand)]
        cmd: ImageCmd,
    },
}

#[derive(Subcommand)]
pub(crate) enum ImageCmd {
    /// Build the diskless image. With [netboot] flake set, ONE nix build of the
    /// image flake yields the atomic co-built (kernel, initrd, netboot.ipxe)
    /// triple, sealed under a sha256 manifest (an init=/initrd mismatch cannot
    /// be staged or served). Without a flake, the CachyOS ISO pipeline runs.
    /// Previews the plan; --apply executes.
    Build {
        /// Execute the pipeline instead of previewing it.
        #[arg(long)]
        apply: bool,
        /// Also stage (activate) the built image after verification.
        #[arg(long)]
        stage: bool,
    },
    /// Stage a built image: verify its manifest (initrd sha256 + init= pairing)
    /// and make it the one the boot server serves. Refuses on any mismatch.
    Stage { id: String },
    /// List built images with their manifests (id, init=, initrd sha, date);
    /// the staged/active image is marked. Also reports legacy ISO artifacts.
    List,
}

#[derive(Subcommand)]
pub(crate) enum ClusterCmd {
    /// List configured clusters.
    List,
    /// Live status: each cluster's head (serving?) + worker RPC reachability.
    Status {
        /// A specific cluster (default: all configured clusters).
        cluster: Option<String>,
    },
    /// Bring a cluster up serving a model (start workers + launch head --rpc).
    Up {
        cluster: String,
        model: String,
        /// Proceed even when the llama.cpp build version differs across nodes.
        /// Normally refused: a mixed-version cluster aborts at the RPC handshake.
        #[arg(long)]
        allow_version_skew: bool,
    },
    /// Tear a cluster down (stop workers, revert head).
    Down { cluster: String },
}

/// Parse + dispatch + render failures. Returns the process exit code under
/// the agentic convention (0 ok / 1 error / 2 refusal - see `src/agentic.rs`).
pub(crate) fn run() -> i32 {
    let cli = Cli::parse();
    let json = cli.json;
    match dispatch(cli) {
        Ok(()) => 0,
        Err(e) => crate::agentic::report_failure(&e, json),
    }
}

/// The TUI guard for non-interactive callers: `--json` (or a piped stdout
/// with no subcommand) means an agent/script is driving, and silently
/// launching a fullscreen TUI would hang it. Pure, so it's unit-testable.
pub(crate) fn tui_refusal(json: bool, explicit: bool, stdout_tty: bool) -> Option<String> {
    if json {
        return Some(
            "the TUI is interactive and has no --json surface - run a subcommand \
             (e.g. `llmtune node status --json`, `llmtune fleet status --json`; \
             see `llmtune --help`)"
                .into(),
        );
    }
    if !explicit && !stdout_tty {
        return Some(
            "stdout is not a terminal and no subcommand was given - the default \
             surface is the interactive TUI. Run a subcommand (see `llmtune --help`), \
             or `llmtune tui` explicitly to force the TUI"
                .into(),
        );
    }
    None
}

fn dispatch(cli: Cli) -> Result<()> {
    let cfg = Config::load(cli.config.as_deref())?;

    let explicit_tui = matches!(cli.cmd, Some(Cmd::Tui));
    match cli.cmd.unwrap_or(Cmd::Tui) {
        Cmd::Tui => {
            use std::io::IsTerminal;
            if let Some(msg) = tui_refusal(cli.json, explicit_tui, std::io::stdout().is_terminal())
            {
                return Err(crate::agentic::refusal(msg));
            }
            ui::run(cfg, cli.config.clone())
        }
        Cmd::Setup {
            print,
            yes,
            user,
            models_dir,
        } => setup::run(
            &cfg,
            &setup::Opts {
                print,
                assume_yes: yes,
                user,
                models_dir,
            },
        ),
        Cmd::Doctor => cmd_doctor_on(&cfg, None, cli.json),
        Cmd::Node { node, cmd } => {
            let sel = node.as_deref();
            match cmd {
                NodeCmd::List => cmd_node_list(&cfg, sel, cli.json),
                NodeCmd::Served => cmd_node_served(&cfg, sel, cli.json),
                NodeCmd::Status => cmd_node_status(&cfg, sel, cli.json),
                NodeCmd::Load { name } => cmd_node_load(&cfg, sel, &name, cli.json),
                NodeCmd::Bench {
                    model,
                    prompt_tokens,
                    gen_tokens,
                    repeats,
                } => cmd_node_bench(
                    &cfg,
                    sel,
                    model.as_deref(),
                    bench::BenchSpec {
                        prompt_tokens,
                        gen_tokens,
                        repeats,
                    },
                    cli.json,
                ),
                NodeCmd::History { limit } => cmd_node_history(&cfg, sel, limit, cli.json),
                NodeCmd::Unload => cmd_node_unload(&cfg, sel, cli.json),
                NodeCmd::Doctor => cmd_doctor_on(&cfg, sel, cli.json),
                NodeCmd::Fit { model, kv } => {
                    reject_remote(&cfg, sel, "node fit")?;
                    cmd_node_fit(&cfg, model.as_deref(), kv.as_deref(), cli.json)
                }
                NodeCmd::Gpu => cmd_node_gpu(&cfg, sel, cli.json),
                NodeCmd::BuildVersion { name } => {
                    reject_remote(&cfg, sel, "node build-version")?;
                    cmd_node_build_version(&name, cli.json)
                }
                NodeCmd::Endpoint => cmd_endpoint_on(&cfg, sel, cli.json),
                NodeCmd::Profile { cmd } => match cmd {
                    NodeProfileCmd::SetModel {
                        model,
                        flags,
                        reset,
                    } => {
                        reject_remote(&cfg, sel, "node profile set-model")?;
                        cmd_profile_set_model(&cfg, &model, flags, reset, cli.json)
                    }
                },
                NodeCmd::Server { action } => cmd_node_server(&cfg, sel, action, cli.json),
                NodeCmd::Expose { action } => {
                    reject_remote(&cfg, sel, "node expose")?;
                    cmd_node_expose(&cfg, action, cli.json)
                }
                NodeCmd::ApiKey { cmd } => {
                    reject_remote(&cfg, sel, "node api-key")?;
                    cmd_node_api_key(&cfg, cmd, cli.json)
                }
                NodeCmd::BootRestore { install } => {
                    reject_remote(&cfg, sel, "node boot-restore")?;
                    cmd_node_boot_restore(&cfg, install, cli.json)
                }
            }
        }
        Cmd::Fleet { cmd } => match cmd {
            FleetCmd::Status => cmd_fleet_status(&cfg, cli.json),
            FleetCmd::BenchAll {
                prompt_tokens,
                gen_tokens,
                repeats,
            } => cmd_fleet_bench_all(
                &cfg,
                bench::BenchSpec {
                    prompt_tokens,
                    gen_tokens,
                    repeats,
                },
                cli.json,
            ),
            FleetCmd::SwapAll { name } => cmd_fleet_swap_all(&cfg, &name, cli.json),
            FleetCmd::Leaderboard => cmd_fleet_leaderboard(&cfg, cli.json),
        },
        Cmd::Cluster { cmd } => match cmd {
            ClusterCmd::List => cmd_cluster_list(&cfg, cli.json),
            ClusterCmd::Status { cluster } => {
                cmd_cluster_status(&cfg, cluster.as_deref(), cli.json)
            }
            ClusterCmd::Up {
                cluster,
                model,
                allow_version_skew,
            } => cmd_cluster_up(&cfg, &cluster, &model, allow_version_skew, cli.json),
            ClusterCmd::Down { cluster } => cmd_cluster_down(&cfg, &cluster, cli.json),
        },
        Cmd::Netboot { cmd } => match cmd {
            NetbootCmd::Init { apply } => netboot::init(
                &cfg,
                &netboot::InitOpts {
                    apply,
                    config: cli.config.clone(),
                    json: cli.json,
                },
            ),
            NetbootCmd::Up {
                no_firewall,
                skip_check,
            } => netboot_server::up(
                &cfg,
                &netboot_server::UpOpts {
                    no_firewall,
                    skip_check,
                    json: cli.json,
                },
            ),
            NetbootCmd::Down => netboot_server::down(&cfg, cli.json),
            NetbootCmd::Status => netboot_server::status(&cfg, cli.json),
            NetbootCmd::Nodes { register } => {
                netboot::nodes(&cfg, register, cli.json, cli.config.as_deref())
            }
            NetbootCmd::Arm { node, disarm } => netboot_node::arm(&cfg, &node, disarm, cli.json),
            NetbootCmd::Boot { node, yes } => netboot_node::boot(&cfg, &node, yes, cli.json),
            NetbootCmd::Console {
                node,
                port,
                since,
                file,
                duration_secs,
            } => netboot_node::console(
                &cfg,
                &node,
                &netboot_node::ConsoleOpts {
                    port: port
                        .or(cfg.netboot.as_ref().map(|n| n.console_port))
                        .unwrap_or(netboot_node::DEFAULT_CONSOLE_PORT),
                    since,
                    file,
                    duration_secs,
                    json: cli.json,
                },
            ),
            NetbootCmd::Serve { port } => netboot::serve(&cfg, port),
            NetbootCmd::Image { cmd } => match cmd {
                ImageCmd::Build { apply, stage } => {
                    netboot::image_build(&cfg, apply, stage, cli.json)
                }
                ImageCmd::Stage { id } => netboot::image_stage(&cfg, &id, cli.json),
                ImageCmd::List => netboot::image_list(&cfg, cli.json),
            },
        },
        Cmd::Models { cmd } => match cmd {
            ModelsCmd::List => cmd_models_list(&cfg, cli.json),
            ModelsCmd::Add { source } => cmd_models_add(&cfg, &source, cli.json),
            ModelsCmd::Rm { name, yes, force } => cmd_models_rm(&cfg, &name, yes, force, cli.json),
        },
        Cmd::Profile { cmd } => match cmd {
            ProfileCmd::List => cmd_profile_list(cli.json),
            ProfileCmd::Show { model } => cmd_profile_show(&cfg, &model, cli.json),
            ProfileCmd::SetModel {
                model,
                flags,
                reset,
            } => cmd_profile_set_model(&cfg, &model, flags, reset, cli.json),
            ProfileCmd::Set { id, flags, bin, ld } => {
                cmd_profile_set(&id, flags, bin, ld, cli.json)
            }
        },
        Cmd::Build { cmd } => match cmd {
            BuildCmd::List => cmd_build_list(cli.json),
            BuildCmd::Install {
                name,
                r#ref,
                retain,
            } => cmd_build_install(&name, r#ref, retain, false, cli.json),
            BuildCmd::Update {
                name,
                r#ref,
                retain,
            } => cmd_build_install(&name, r#ref, retain, true, cli.json),
            BuildCmd::Rollback { name } => cmd_build_rollback(&cfg, &name, cli.json),
        },
        Cmd::Mem {
            model,
            ctx,
            kv,
            fit,
        } => {
            if fit {
                // The former `node fit` surface, folded under `mem --fit`.
                if ctx.is_some() {
                    anyhow::bail!(
                        "--ctx doesn't apply with --fit (it reports the MAX context \
                         per KV quant) - drop --ctx, or use `mem <model> --ctx N` to \
                         size a specific context"
                    );
                }
                cmd_node_fit(&cfg, model.as_deref(), kv.as_deref(), cli.json)
            } else {
                let model = model.ok_or_else(|| {
                    anyhow::anyhow!("pass a model name (or --fit to size the served model)")
                })?;
                cmd_mem(&cfg, &model, ctx, kv.as_deref().unwrap_or("f16"), cli.json)
            }
        }
        Cmd::Endpoint { cmd } => match cmd.unwrap_or(EndpointCmd::Show) {
            EndpointCmd::Show => cmd_endpoint(&cfg, cli.json),
            EndpointCmd::Expose { action } => cmd_node_expose(&cfg, action, cli.json),
            EndpointCmd::Auth { action } => cmd_node_auth(&cfg, action, cli.json),
            EndpointCmd::Identity { action } => cmd_node_identity(&cfg, action, cli.json),
            EndpointCmd::ApiKey { cmd } => cmd_node_api_key(&cfg, cmd, cli.json),
        },
        Cmd::Compare { filter } => cmd_compare(&cfg, filter, cli.json),
        Cmd::Proxy {
            host,
            port,
            swap_hold,
        } => proxy::serve(
            &cfg,
            &proxy::ProxyOpts {
                host,
                port,
                swap_hold: std::time::Duration::from_secs(swap_hold),
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_set_accepts_hyphen_leading_flags() {
        // Every llama flag starts with '-'; the --flags value must be accepted.
        let cli = Cli::try_parse_from([
            "llmtune",
            "profile",
            "set",
            "_default",
            "--flags",
            "-c 4096 --flash-attn on",
        ])
        .expect("hyphen-leading --flags value must parse");
        match cli.cmd {
            Some(Cmd::Profile {
                cmd: ProfileCmd::Set { id, flags, .. },
            }) => {
                assert_eq!(id, "_default");
                assert_eq!(flags.as_deref(), Some("-c 4096 --flash-attn on"));
            }
            _ => panic!("expected `profile set`"),
        }
    }

    #[test]
    fn endpoint_group_parses_show_expose_and_api_key() {
        // bare `endpoint` = show
        let c = Cli::try_parse_from(["llmtune", "endpoint"]).unwrap();
        assert!(matches!(c.cmd, Some(Cmd::Endpoint { cmd: None })));
        let c = Cli::try_parse_from(["llmtune", "endpoint", "show"]).unwrap();
        assert!(matches!(
            c.cmd,
            Some(Cmd::Endpoint {
                cmd: Some(EndpointCmd::Show)
            })
        ));
        let c = Cli::try_parse_from(["llmtune", "endpoint", "expose", "on"]).unwrap();
        assert!(matches!(
            c.cmd,
            Some(Cmd::Endpoint {
                cmd: Some(EndpointCmd::Expose { .. })
            })
        ));
        let c = Cli::try_parse_from(["llmtune", "endpoint", "api-key", "generate"]).unwrap();
        assert!(matches!(
            c.cmd,
            Some(Cmd::Endpoint {
                cmd: Some(EndpointCmd::ApiKey {
                    cmd: ApiKeyCmd::Generate
                })
            })
        ));
        // The intuitive auth toggle: on / off / status.
        for (arg, want) in [
            ("on", AuthAction::On),
            ("off", AuthAction::Off),
            ("status", AuthAction::Status),
        ] {
            let c = Cli::try_parse_from(["llmtune", "endpoint", "auth", arg]).unwrap();
            match c.cmd {
                Some(Cmd::Endpoint {
                    cmd: Some(EndpointCmd::Auth { action }),
                }) => assert!(
                    std::mem::discriminant(&action) == std::mem::discriminant(&want),
                    "auth {arg}"
                ),
                _ => panic!("endpoint auth {arg} did not parse"),
            }
        }
    }

    #[test]
    fn endpoint_identity_parses_harness_model_and_status() {
        // The identity toggle mirrors auth: harness / model / status.
        for (arg, want) in [
            ("harness", IdentityAction::Harness),
            ("model", IdentityAction::Model),
            ("status", IdentityAction::Status),
        ] {
            let c = Cli::try_parse_from(["llmtune", "endpoint", "identity", arg]).unwrap();
            match c.cmd {
                Some(Cmd::Endpoint {
                    cmd: Some(EndpointCmd::Identity { action }),
                }) => assert!(
                    std::mem::discriminant(&action) == std::mem::discriminant(&want),
                    "identity {arg}"
                ),
                _ => panic!("endpoint identity {arg} did not parse"),
            }
        }
    }

    #[test]
    fn hidden_aliases_still_parse_for_one_release() {
        // `node expose` / `node api-key` / `node fit` are hidden but functional.
        let c = Cli::try_parse_from(["llmtune", "node", "expose", "off"]).unwrap();
        assert!(matches!(
            c.cmd,
            Some(Cmd::Node {
                cmd: NodeCmd::Expose { .. },
                ..
            })
        ));
        let c = Cli::try_parse_from(["llmtune", "node", "api-key", "show"]).unwrap();
        assert!(matches!(
            c.cmd,
            Some(Cmd::Node {
                cmd: NodeCmd::ApiKey {
                    cmd: ApiKeyCmd::Show
                },
                ..
            })
        ));
        let c = Cli::try_parse_from(["llmtune", "node", "fit", "qwen", "--kv", "q8_0"]).unwrap();
        match c.cmd {
            Some(Cmd::Node {
                cmd: NodeCmd::Fit { model, kv },
                ..
            }) => {
                assert_eq!(model.as_deref(), Some("qwen"));
                assert_eq!(kv.as_deref(), Some("q8_0"));
            }
            _ => panic!("expected `node fit`"),
        }
    }

    #[test]
    fn mem_fit_folds_the_two_memory_surfaces() {
        // plain mem: model + default kv
        let c = Cli::try_parse_from(["llmtune", "mem", "qwen", "--ctx", "65536"]).unwrap();
        match c.cmd {
            Some(Cmd::Mem {
                model,
                ctx,
                kv,
                fit,
            }) => {
                assert_eq!(model.as_deref(), Some("qwen"));
                assert_eq!(ctx, Some(65536));
                assert!(kv.is_none() && !fit);
            }
            _ => panic!("expected `mem`"),
        }
        // mem --fit with no model (sizes the served model)
        let c = Cli::try_parse_from(["llmtune", "mem", "--fit"]).unwrap();
        assert!(matches!(
            c.cmd,
            Some(Cmd::Mem {
                model: None,
                fit: true,
                ..
            })
        ));
    }

    #[test]
    fn tui_guard_refuses_non_interactive_callers() {
        // --json can never launch the TUI, explicit or default.
        assert!(tui_refusal(true, false, true).is_some());
        assert!(tui_refusal(true, true, false).is_some());
        // no subcommand + piped stdout = an agent/script; refuse with a hint.
        let msg = tui_refusal(false, false, false).unwrap();
        assert!(msg.contains("llmtune tui"), "names the explicit escape");
        // interactive default and explicit `tui` on a real terminal both run.
        assert!(tui_refusal(false, false, true).is_none());
        assert!(tui_refusal(false, true, true).is_none());
        // explicit `llmtune tui` without --json is honored even piped
        // (ratatui itself reports the terminal problem).
        assert!(tui_refusal(false, true, false).is_none());
    }

    #[test]
    fn node_group_takes_a_target_selector() {
        // `node --node <name> <verb>`: drive a remote fleet node from here.
        let c = Cli::try_parse_from(["llmtune", "--json", "node", "--node", "bc250-a", "status"])
            .unwrap();
        match c.cmd {
            Some(Cmd::Node { node, cmd }) => {
                assert_eq!(node.as_deref(), Some("bc250-a"));
                assert!(matches!(cmd, NodeCmd::Status));
            }
            _ => panic!("expected `node --node bc250-a status`"),
        }
        // and without it, the selector is None (the local node).
        let c = Cli::try_parse_from(["llmtune", "node", "served"]).unwrap();
        assert!(matches!(
            c.cmd,
            Some(Cmd::Node {
                node: None,
                cmd: NodeCmd::Served
            })
        ));
    }

    #[test]
    fn node_transport_verbs_parse() {
        // The verbs the SSH transport reinvokes must exist on the CLI surface.
        let c = Cli::try_parse_from(["llmtune", "--json", "node", "endpoint"]).unwrap();
        assert!(c.json);
        assert!(matches!(
            c.cmd,
            Some(Cmd::Node {
                cmd: NodeCmd::Endpoint,
                ..
            })
        ));
        let c = Cli::try_parse_from([
            "llmtune",
            "--json",
            "node",
            "profile",
            "set-model",
            "m.gguf",
            "--flags",
            "-c 4096 -ngl 99",
        ])
        .unwrap();
        match c.cmd {
            Some(Cmd::Node {
                cmd:
                    NodeCmd::Profile {
                        cmd:
                            NodeProfileCmd::SetModel {
                                model,
                                flags,
                                reset,
                            },
                    },
                ..
            }) => {
                assert_eq!(model, "m.gguf");
                assert_eq!(flags.as_deref(), Some("-c 4096 -ngl 99"));
                assert!(!reset);
            }
            _ => panic!("expected `node profile set-model`"),
        }
    }
}
