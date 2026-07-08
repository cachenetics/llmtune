# llmtune - SPEC

Status: IMPLEMENTED (v0.1.0; scoped 2026-06-29, narrowed to inference engine 2026-07-02)
License: GPL-2.0-only
Family: the Cachenetics BC-250 tool family (`memtune` / `aputune` / `biostune` / `llmtune` / `wikitune`).

---

## 1. Purpose

`llmtune` is the public, self-hostable **inference engine for the AMD BC-250**. It lets a BC-250 owner
go from "I have a box and some GGUF files" to "I am serving and benchmarking models" without the trial-
and-error that the silicon currently demands. It discovers models, launches and hot-swaps them with
known-good per-architecture flags, benchmarks throughput, logs the history, and surfaces an
OpenAI-compatible endpoint the owner points their own agent/harness at.

llmtune does **one thing well: run models on the BC-250, easily.** It holds no knowledge of its own and
is not a chat app - the board's reference manual is a separate tool (`wikitune`, browsable by humans and
parseable by agents), and grounded Q&A / agent work belongs to whatever harness the owner points at the
served endpoint. llmtune is the runtime member of the BC-250 family (`memtune`/`aputune`/`biostune`
tune the hardware; `wikitune` is the manual; `llmtune` runs the models).

A single TUI controls **one BC-250 or many** - run it on the BC-250 itself, or on a separate host
orchestrating a rack. The single-node case is simply a one-node fleet.

### 1.1 Why this exists

Running LLM inference on the BC-250 (Cyan Skillfish / Oberon, gfx1013, RDNA1-class, 16 GB unified
GDDR6) is finicky: the right llama-server flags differ by model architecture (Vulkan vs the MTP draft
build, UMA memory split, context sizing, KV quant), and a wrong configuration silently underperforms or
fails to load. The hard-won knowledge of "which flags make which model fly" currently lives in one
operator's scripts. `llmtune` encodes it as data so the whole community inherits it.

---

## 2. Goals and non-goals

### 2.1 Goals

- **Make models run.** Detect the hardware, discover GGUFs, and launch them with known-good per-arch
  profiles. A new owner should reach a served model in one command.
- **Make swapping safe.** Hot-swap the served model; health-check; auto-revert on failure so a bad
  swap never leaves the box without a working model.
- **Make performance measurable.** Benchmark throughput (tok/s, TTFT, latency) with GPU telemetry, and
  log every run to a durable per-node, per-model history.
- **Make a fleet manageable.** One TUI/CLI over N nodes: per-node status, batch operations, a
  cross-node benchmark leaderboard.
- **Pool boxes for big models.** Combine several BC-250s into a cluster (llama.cpp RPC) to serve a model
  too large for any single box's 16 GiB. A first-class option, not a bolt-on.
- **Stay public and portable.** No private paths, no single-host assumptions, no mandatory daemon to
  install. Configurable models dir / port / service unit. GPL-2.0-only, community-ownable.
- **Be thorough, not rushed.** Correctness, durability, and a clean architecture over a fast MVP.

### 2.2 Non-goals (initial)

- Not a model trainer or fine-tuner.
- Not a general multi-GPU/multi-vendor inference manager - BC-250 first; portability is a bonus, not a
  charter.
- Not a replacement for llama-server - `llmtune` orchestrates it, it does not reimplement inference.
- **Not a knowledge brain and not a chat app.** llmtune serves an OpenAI-compatible endpoint; grounded
  Q&A, RAG, personas and agent behaviour belong to the harness the owner points at it. The BC-250
  reference corpus lives in the sibling tool `wikitune` (agent-parseable via its CLI/JSONL export).

---

## 3. The pillars

| Pillar | Responsibility | Ported from |
|---|---|---|
| **doctor** | Preflight: detect BC-250, Vulkan/rusticl, UMA split, amdgpu, models dir, llama build + build-deps. | `memtune doctor` + setup knowledge |
| **build** | Own the llama.cpp toolchain: install/update/rollback versioned, pinned Vulkan/MTP/RPC builds for gfx1013. | new (`builds.toml` as data) |
| **runtime** | Discover GGUFs (arch/params/quant/ctx), launch/swap via per-arch profiles, health-poll, auto-revert, pre-warm. | a prior model-select script |
| **bench** | Perf: tok/s prompt+gen, TTFT, latency + GPU telemetry; crash-safe history; quant/build comparison. | `memtune` bench + llama-server `/props` |
| **endpoint** | Surface the OpenAI-compatible URL + connection snippets; swap-on-demand proxy so one box serves a whole library. | new, thin over llama-server |

**Explicitly out of scope** (decision, binding 2026-07-02): no in-app chat client, no knowledge brain, no
local RAG / embed / rerank / FTS. llmtune's output is a fast model at a URL; the owner's own harness
does the conversing and grounding. The BC-250 reference corpus is the separate `wikitune` tool.

---

## 4. Architecture

### 4.1 One binary, role by invocation

```
llmtune                       # default -> controller TUI over the configured fleet (localhost if none)
llmtune tui                   # explicit TUI
llmtune doctor                # preflight on the local node
llmtune node <subcommand>     # node-local op, emits JSON; ALSO what the ssh transport runs remotely
llmtune fleet <subcommand>    # controller CLI (headless / scriptable / cron-able)
llmtune agent                 # node daemon (Phase: enables the Agent transport)
```

`node` subcommands (the canonical node-op surface; every transport ultimately calls these):

```
llmtune node list             # discovered models (+ which is served)
llmtune node served           # name of currently-served model
llmtune node load <name>      # swap to model (substring match), with profile + auto-revert
llmtune node bench [--spec]   # run a perf benchmark, append to history
llmtune node telemetry        # one telemetry sample (gfx/mem clocks, die temp, power)
llmtune node history          # this node's benchmark history
llmtune node doctor           # preflight
```

`fleet` subcommands (controller-side, fan out over the configured nodes):

```
llmtune fleet status          # one row per node: served model, health, last bench, temp
llmtune fleet bench-all       # benchmark every node, append to history, print leaderboard
llmtune fleet swap-all <name> # swap every node to a model (where present)
llmtune fleet leaderboard     # cross-node tok/s comparison from history
```

### 4.2 Transport abstraction (the keystone)

Node operations are defined once as a `NodeTransport` trait. The controller is identical regardless of
how bytes reach a node - local call, SSH, or a daemon API.

```rust
trait NodeTransport {
    fn discover(&self) -> Result<Vec<Model>>;
    fn served(&self) -> Result<Option<String>>;
    fn load(&self, model: &Model, profile: &Profile) -> Result<SwapOutcome>;
    fn bench(&self, spec: &BenchSpec) -> Result<PerfBench>;
    fn telemetry(&self) -> Result<Telemetry>;
    fn history(&self) -> Result<Vec<Record>>;
    fn doctor(&self) -> Result<DoctorReport>;
}
```

Three implementations, sequenced:

| Impl | Mechanism | Security | Phase | Use |
|---|---|---|---|---|
| `Local` | In-process node ops. | n/a (local root) | 1 | Single BC-250 running it on itself. |
| `Ssh` | Controller runs `llmtune node <op> --json` over SSH, parses JSON. | SSH keys | 1 | Fleet over LAN / Tailscale / VPN. Zero daemon to install. |
| `Agent` | HTTP/JSON to `llmtune agent` (systemd service on each node). | bearer token + optional TLS | later | Live-streaming telemetry, low-latency rack dashboard. |

Design decision (locked): **all three are designed into the trait from day one; `Local` + `Ssh` are
implemented first.** That covers single-node and the whole fleet for every owner with nothing to
install or secure beyond SSH. The `Agent` daemon is added when a live wall-of-nodes dashboard justifies
the added security surface.

Rationale for the sequence: the node-op surface (`llmtune node <op> --json`) is the SAME contract the
SSH transport invokes and the Agent daemon will wrap. Building `node` subcommands first means the SSH
transport is nearly free, and the Agent daemon later is "serve the existing node ops over HTTP" rather
than a new surface.

### 4.3 Fleet model and config

A fleet is a list of nodes. `localhost` is implicit when no fleet file exists - a single BC-250 never
sees a "fleet" concept.

`~/.config/llmtune/fleet.toml`:

```toml
[[node]]
name       = "bc250-1"
 host       = "192.0.2.10"     # omit or "localhost" for the local node
transport  = "ssh"              # local | ssh | agent
ssh_user   = "user"
ssh_key    = "~/.ssh/id_ed25519"
models_dir = "/home/user/models"
llama_unit = "llama-server.service"
llama_url  = "http://127.0.0.1:8080"

[[node]]
name       = "bc250-2"
host       = "192.0.2.11"
transport  = "ssh"
ssh_user   = "user"
# ... per-node overrides; unspecified fields inherit [defaults]

[defaults]
models_dir = "/home/user/models"
llama_unit = "llama-server.service"
llama_url  = "http://127.0.0.1:8080"

# A cluster pools several nodes to serve one model too big for a single BC-250,
# via llama.cpp RPC (see section 6.1). The head runs llama-server --rpc <workers>;
# each worker runs rpc-server. Members must run a GGML_RPC-enabled llama.cpp build.
[[cluster]]
name      = "big"
head      = "bc250-1"
workers   = ["bc250-2", "bc250-3"]
rpc_port  = 50052
rpc_bin   = "/path/to/llama.cpp/build/bin/rpc-server"  # optional override
```

### 4.4 Controller (TUI + CLI over the same engine)

CLI-first (locked): the engine is fully usable headless - `fleet`/`node` subcommands emit text or
`--json`, scriptable and cron-able. The TUI is a view over the same engine, never a separate code path.

---

## 5. Data model

### 5.1 Model descriptor

```rust
struct Model {
    path:       PathBuf,
    name:       String,        // basename
    arch:       String,        // general.architecture from GGUF header
    params:     Option<u64>,   // parameter count if derivable
    quant:      Option<String>,// e.g. "IQ2_XXS", "Q4_K_M"
    size_bytes: u64,
    ctx_max:    Option<u32>,   // training context from header
    profile:    ProfileId,     // resolved per-arch profile
}
```

GGUF header is parsed directly (scalars only, arrays skipped) for `general.architecture` and sizing -
no weights loaded.

### 5.2 Per-arch profile (the crown jewel)

```rust
struct Profile {
    id:         ProfileId,
    arch_match: Vec<String>,   // family prefixes, e.g. ["qwen35moe"]
    bin:        PathBuf,       // llama-server binary (build varies: Vulkan vs MTP-cross)
    ld_path:    Option<PathBuf>,
    env:        BTreeMap<String, String>,
    flags:      String,        // the known-good launch flags
}
```

Profiles ship as DATA (a bundled `profiles.toml`, overridable by the user), seeded from a known-good
starter set. Initial set: `gemma`, `lfm2`, `qwen35moe`, `qwen35`, `_default`. Resolution is by
architecture-family prefix; `_default` is the fallback (and flags a "no known profile" warning so the
owner knows they are on generic settings).

Example (seed, qwen35moe - the MTP speculative-draft build):

```toml
[[profile]]
id = "qwen35moe"
arch_match = ["qwen35moe"]
bin = "/path/to/llama-mtp-cross/llama-server"
ld_path = "/path/to/llama-mtp-cross"
env = { GGML_VULKAN_DEVICE = "0", GGML_VK_PREFER_HOST_MEMORY = "1" }
flags = "-c 32768 --parallel 1 --flash-attn on --no-mmap --spec-type draft-mtp --spec-draft-n-max 1 --reasoning off --jinja -ngl 99 -t 4 -ctk q4_0 -ctv q4_0 --cache-ram 0 --ctx-checkpoints 4"
```

Note: the seed `bin`/`ld_path` are host-specific paths; the SPEC treats these as DEFAULTS to be made
configurable (discover the llama build, or take a path), so the profiles are portable beyond the seed
host.

### 5.3 Perf benchmark result

```rust
struct PerfBench {
    prompt_tok_s:   f64,
    gen_tok_s:      f64,
    ttft_ms:        f64,       // time to first token
    total_ms:       f64,
    n_prompt:       u32,
    n_gen:          u32,
    telemetry:      TelemetrySummary, // min/max/avg gfx & mem clock, peak die temp, peak power
}
```

Measured by driving llama-server's completion endpoint with a fixed `BenchSpec` (prompt tokens, gen
tokens, repeat count) and timing the stream; tok/s from server timings where available, wall-clock
otherwise. GPU telemetry sampled throughout the run.

### 5.4 Telemetry

Read from `/sys/class/drm/*/device/gpu_metrics` (v2.2 struct): gfx clock, mem clock, die temp, power.
Ported from `memtune metrics.rs`. Sampled on idle ticks in the TUI and throughout a bench run.

### 5.5 History record

```rust
struct Record {
    ts:       u64,             // unix seconds
    node:     String,
    model:    String,
    arch:     String,
    quant:    Option<String>,
    ctx:      u32,
    profile:  ProfileId,
    perf:     PerfBench,
    build:    Option<String>,  // llama.cpp build slug that served this run
    notes:    String,
}
```

### 5.7 Cluster descriptor

```rust
struct Cluster {
    name:     String,
    head:     String,        // node name that runs llama-server --rpc ...
    workers:  Vec<String>,   // node names that run rpc-server
    rpc_port: u16,
    rpc_bin:  Option<String>,// rpc-server path override (else discovered)
}
```

A cluster references nodes by name from the fleet. Serving a model on a cluster pools the members'
unified memory via llama.cpp RPC (section 6.1). A bench `Record` taken on a cluster carries the member
list and realized aggregate tok/s, so the memory-vs-throughput trade is visible in history.

---

## 6. Runtime: swap mechanism

Swap is the highest-risk operation and gets the most care. Generalized from a prior model-select script,
hardened.

Sequence:

1. **Guard.** Refuse if a benchmark is running on the node (lock file). Refuse ambiguous substring
   matches; require a unique model match.
2. **Resolve.** Parse the target's GGUF arch; resolve its `Profile`.
3. **Stage.** Write a systemd drop-in for the llama unit with the profile's `ExecStart` (bin + model +
   host/port + flags), `Environment=` lines, and `LD_LIBRARY_PATH`. Back up the prior drop-in.
4. **Apply.** `systemctl daemon-reload` + `systemctl restart <llama_unit>`.
5. **Health-poll.** Poll `<llama_url>/health` for "ok" for up to ~2 min.
6. **Auto-revert.** On timeout/failure: restore the prior drop-in, daemon-reload, restart, and report
   the failure. The box is never left without a working model.
7. **Pre-warm.** On success: warm the KV cache with a few short prompts (n = 8, 64, 256 tokens).
8. **Report.** Return `SwapOutcome { from, to, ok, reverted, elapsed }`.

Two actuation backends behind a trait: `SystemdDropin` (the community-standard, Phase 1) and, optionally
later, `ManagedProcess` (llmtune spawns/kills llama-server itself for hosts without systemd). Drop-in
first.

The served model is detected by GET `<llama_url>/props` -> `model_path` basename; used
to label benches and mark the current model in `list`.

### 6.1 Cluster mode - combining BC-250s for larger models

A single BC-250 has 16 GiB of unified GDDR6, which caps the model size it can serve. `llmtune`
supports **pooling several BC-250s to serve one model too big for any single box**, via llama.cpp's RPC
backend. This is a first-class option, not an afterthought: a `cluster` is a named group of fleet nodes
that jointly serve one model.

Mechanism (llama.cpp RPC):

- Each **worker** node runs `rpc-server` (from a `GGML_RPC`-enabled llama.cpp build) bound to an RPC
  port, advertising its GPU/UMA as a remote backend.
- The **head** node runs `llama-server` with `--rpc <w1_host>:<port>,<w2_host>:<port>,…` and `-ngl 99`.
  llama.cpp shards the model's layers across the head's local GPU plus every RPC backend, so the model's
  weights and KV are spread over the pooled memory of all the boxes.

Cluster swap is a superset of the single-node swap (section 6), orchestrated over the transport:

1. **Guard + resolve** as in 6 (on the head).
2. **Start workers.** On each worker (via its transport), launch `rpc-server` on the cluster's RPC port;
   health-check each worker's RPC endpoint is accepting.
3. **Launch head.** Stage the head's llama drop-in from the model's `Profile`, but inject
   `--rpc <worker-list>` into the flags (profile flags otherwise unchanged; `-ngl 99` stays - layers
   offload across backends). daemon-reload + restart.
4. **Health-poll** the head `<llama_url>/health`.
5. **Auto-revert (whole cluster).** On any failure - a worker that won't start, or the head that won't
   come healthy - tear the whole cluster down cleanly: stop the `rpc-server` on every worker, restore the
   head's prior drop-in, restart. The cluster never half-forms.
6. **Pre-warm + report** as in 6; `SwapOutcome` carries the per-worker status.

Notes / constraints:

- RPC is bandwidth-sensitive; cluster throughput depends on the interconnect between boxes (a wired LAN
  is the baseline; the bench history captures the realized tok/s so an owner sees the cost). `doctor`
  warns if workers are only reachable over a slow/wireless path.
- All nodes must run a `GGML_RPC`-enabled llama.cpp build. `doctor` (run on each member) warns when the
  node's installed managed build lacks the `rpc-server` binary (a build predating `-DGGML_RPC=ON`), with
  the rebuild remedy.
- All nodes must run the SAME llama.cpp version - a mixed-version cluster aborts at the RPC handshake
  with a cryptic "malformed response". `cluster up` compares the profile's managed-build version (commit
  slug) across head + workers before touching any worker: a confirmed mismatch is refused
  (`--allow-version-skew` overrides); an identity that can't be determined (older remote llmtune,
  literal-bin profile) warns loudly and proceeds.
- Only dense-transformer models pool. Recurrent/hybrid state-space architectures (lfm2, mamba, rwkv,
  jamba, …) cannot be split over llama.cpp RPC - the worker rejects the recurrent-state graph
  (`[create_node] invalid data ptr`) - so `cluster up` refuses them; they still serve fine single-node.
- A cluster reserves its member nodes - they cannot independently serve their own model while enrolled in
  an active cluster serve (the fleet view shows them as "in cluster `<name>`").
- Single-node serving and cluster serving share the same `Profile` data; cluster mode only adds the
  `--rpc` injection and the worker orchestration, so per-arch knowledge is not duplicated.
- **Security caveat.** llama.cpp's `rpc-server` is unauthenticated and, by design, lets the head drive
  memory and compute on the worker - treat an open RPC port as granting code-execution-equivalent trust.
  llmtune binds workers on `0.0.0.0` for LAN pooling, so **only run clusters on a trusted private
  network** (never expose an RPC port to the internet). A future option may bind workers to a specific
  interface; until then the operator owns the network boundary.

---

## 7. doctor (preflight)

One-shot health report so an owner knows what is and is not ready. Checks (extends `memtune doctor`):

- BC-250 detected (PCI 1002:13fe / CPU family 17h model 0x46).
- amdgpu loaded; DRM render node present.
- Vulkan device present (and/or rusticl ICD) for the inference backend.
- UMA memory split sane (enough VRAM carved for the target model).
- A llama-server build present and runnable (Vulkan and/or MTP-cross).
- `models_dir` exists and holds GGUFs.
- `<llama_url>` reachable (if a server is up).
- For fleet use: SSH reachability of each configured node, and `llmtune` present on the remote.

Output: a `[ok]` / `[fail]` / `[warn]` line per check with a one-line remedy on failure.

---

## 8. TUI design

ratatui + crossterm, structured on the `memtune` skeleton (event loop, `Mode` enum, draft/confirm,
async-task polling, crash-safe persistence, centered overlays, telemetry-on-idle, pulse animation).

### 8.1 Views

- **Fleet view** (default when >1 node): a table - one row per node (name, served model, health, last
  tok/s, die temp, status). Enter drills into a node.
- **Node view**: model list (names) + a full-height model card (served state, latest bench, arch/params/
  quant, profile, llama.cpp flags with an in-card editor). This is the single-node default.
- **Endpoint overlay** (key `o`): the OpenAI-compatible URL, served id, health + copy-paste snippets.

### 8.2 Modes and overlays

`Mode`: `Normal | Search | Confirm | Bench | Chat`. Overlays: confirm-swap (shows from/to + warning),
bench-progress (pulse + live telemetry + partial tok/s), doctor report, model detail.

### 8.3 Key bindings (Node view)

```
Up/Down     move selection         Tab     toggle list <-> history focus
Enter / l   load (swap to) model   b       benchmark served model
/           search models          d       doctor
r           refresh discovery      h       history pane
c           chat with served       q/Esc   back / quit
```

Long operations (load, bench) run on a background thread; the loop polls completion each tick (90 ms
during a run, 300 ms idle), exactly as `memtune` polls its bench/memtest threads.

---

## 9. Persistence

Crash-safe JSON, the `memtune tune.rs` pattern (temp file -> fsync -> atomic rename -> fsync dir).

- Per-node history: `/var/lib/llmtune/<node>/history.json` (append-only, newest-first), an array of
  `Record`. On a remote node the history lives on that node; the controller aggregates via the
  transport's `history()`.
- Drop-in backups: `/var/lib/llmtune/backups/` (prior llama drop-in before each swap).
- Locks: `/run/llmtune/bench.lock`, `/run/llmtune/swap.lock` (prevent concurrent bench/swap and
  swap-during-bench).
- User config: `~/.config/llmtune/fleet.toml`, `~/.config/llmtune/profiles.toml` (override of the
  bundled seed).

---

## 10. Security model

- **Local**: root on the box (needed for `/dev`, systemd, sysfs). No network surface.
- **Ssh**: trust is SSH keys; nothing is exposed beyond sshd. The controller runs `llmtune node` on
  the remote; no llmtune-specific port is opened. Default and recommended for fleets.
- **Agent** (later): bearer token (required) + optional TLS; never bind `0.0.0.0` without a token;
  `doctor` warns on an unauthenticated bind. The Agent transport is opt-in precisely because it adds a
  network surface the SSH path avoids.

---

## 11. Crate / module layout (Rust)

```
llmtune/
  Cargo.toml          # ratatui 0.29, crossterm 0.28, serde/serde_json, toml, clap 4, anyhow, libc
  Makefile            # build/release/test/check/install (memtune-style)
  README.md  SPEC.md  LICENSE(GPL-2.0-only)  NOTICE
  profiles.toml       # bundled per-arch seed
  src/
    main.rs           # CLI dispatch: tui | node | fleet | agent | doctor
    cli.rs            # clap defs
    config.rs         # fleet.toml + profiles.toml load/save; NodeDescriptor; defaults inheritance
    model.rs          # Model descriptor + GGUF header parse
    profile.rs        # Profile + arch-family resolution
    serve.rs          # llama-server client: /props, /health, /completion
    swap.rs           # drop-in write, daemon-reload, restart, health-poll, auto-revert, pre-warm
    cluster.rs        # llama.cpp RPC cluster: rpc-server orchestration + --rpc head launch (M5)
    bench.rs          # perf bench engine (async thread + result)
    telemetry.rs      # gpu_metrics sysfs reader (port memtune)
    history.rs        # crash-safe JSON store
    doctor.rs         # preflight checks
    transport/
      mod.rs          # NodeTransport trait
      local.rs        # in-process
      ssh.rs          # run `llmtune node <op> --json` over ssh
      agent.rs        # client + `llmtune agent` server (later phase)
    ui/
      mod.rs          # event loop, App, Mode
      fleet.rs        # fleet table view
      node.rs         # node view (model list | model card + editor)
      bench.rs        # bench overlay
      widgets.rs      # shared widgets ported from memtune (rounded block, pulse, centered, key-line)
```

---

## 12. Phased build order

| Milestone | Scope | Gives |
|---|---|---|
| **M0** | Scaffold, `config.rs`, `model.rs` (GGUF parse), CLI skeleton, `doctor`. | `llmtune doctor`, `llmtune node list`. |
| **M1** | `profile.rs` + `serve.rs` + `swap.rs` (Local). | `llmtune node load <name>` with auto-revert. |
| **M2** | `bench.rs` + `telemetry.rs` + `history.rs` (Local node). | `llmtune node bench`, durable history. |
| **M3** | TUI single-node (Node view, bench overlay, history). | Interactive single BC-250. |
| **M4** | `transport/ssh.rs` + Fleet view + `fleet` CLI + leaderboard. | Multi-node fleet from one TUI/CLI. |
| **M5** | `cluster.rs`: llama.cpp RPC cluster mode (rpc-server orchestration + `--rpc` head launch + whole-cluster auto-revert); cluster config + doctor checks; serve/bench a cluster. | Pool BC-250s to run models too big for one box. |
| **M6** | `build.rs` build manager + memory-math card + per-model flag overrides. | Own the toolchain; show the UMA arithmetic. |
| **M7** | `endpoint.rs` endpoint surface + `proxy.rs` swap-on-demand OpenAI proxy + LAN expose + api-key. | Point any harness at the box. |
| **M8** | `transport/agent.rs` (daemon). | Live-streaming fleet telemetry. *(deferred)* |

M0-M7 is the shippable engine: single node, through full fleet, through pooled-box clusters, with a
managed toolchain and a harness-ready endpoint. (Retired from scope 2026-07-02: an in-TUI chat pane and a
local knowledge "brain" - llmtune serves models; `wikitune` is the reference corpus and the owner's own
harness does the conversing.)

---

## 13. Testing strategy

- GGUF header parse: fixture files (a few real-ish headers), arch/quant/size extraction.
- Profile resolution: arch-family prefix matching, `_default` fallback flagging.
- History: crash-safe write roundtrip; concurrent-append safety; newest-first ordering.
- Swap: drop-in render is a pure function -> snapshot tests; health-poll + auto-revert with a mock
  llama (a fake `/health` that fails -> assert revert).
- Bench: timing/parse from canned `/completion` streams.
- Transport: a mock `NodeTransport` so the controller/TUI test without real nodes; `ssh` transport
  tested against a localhost loopback running the real `node` subcommands.
- TUI render: small-terminal render-without-panic tests (80x24, 40x12, 20x8), the `memtune` pattern.

---

## 14. Open questions

- Profile portability: how to discover the local llama-server build(s) so seed `bin`/`ld_path` are not
  required (probe known locations? a `doctor`-found path written into config?).
- Agent transport timing: ship it in Phase 1 if a live rack dashboard is wanted sooner, or hold to M8.
- Cluster (M5): how to discover/manage the `rpc-server` build per node (same probe problem as profile
  `bin`); whether layer-split ratios across heterogeneous boxes need tuning or llama.cpp's auto-split is
  enough; how the bench leaderboard should present cluster vs single-node runs side by side; whether a
  worker should auto-start `rpc-server` as a managed systemd unit vs an on-demand process per serve.
- Known-good `vulkan` build ref cadence: who/what advances the pinned ref in `builds.toml`, and how
  "known-good for gfx1013" is certified (a manual bless after a bench passes, or an automated
  `doctor` + tok/s-floor smoke gating a new pin).
- Naming of the served-model identity sync (some harnesses sync a config + system-prompt on swap; llmtune's
  equivalent, if any, for a generic owner).

---

## 15. References

- Sibling Cachenetics BC-250 tools: `memtune` - memory-timing TUI/bench/history; `biostune` -
  BIOS-settings navigation/catalog/search; `aputune` - GPU/CPU/CU tuner; `wikitune` - the BC-250
  reference manual (human + agent views).
- Served-model detection uses llama-server's `/props` (`model_path` basename).
