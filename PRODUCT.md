# llmtune - Product Design

Status: ACTIVE (product layer). Most of the arc is implemented.
Companion to [`SPEC.md`](SPEC.md). SPEC is the architecture and build order (the engine);
this doc is the product - the journey that makes a BC-250 owner *want* to use it, and the
pillars that close it.

Pillar status legend: **[SHIPPED]** implemented · **[DEFERRED]** parked, not built · **[AS-IS]**
intentionally left minimal.

---

## 1. The bar, and what it does and does not mean

The goal is "a tool BC-250 owners love, the way people love LM Studio." What makes those
tools loved is not their inference - it is the **friction they remove**. On the BC-250 the
friction is far worse than mainstream: getting a working Vulkan/MTP `llama.cpp` is harder
than getting models, "16 GiB" does not tell you what fits because it is UMA shared between
CPU and GPU, and the known-good launch flags are folklore scattered across ad-hoc scripts.

So "loved like LM Studio" here means:

- **Zero hand-edited TOML to first served model.** Detect, build, configure, run.
- **The platform's hard parts handled or made honest** - the toolchain is managed; the
  memory math is shown, not guessed.
- **The community's hard-won knowledge shipped as data** - flags, build refs, all
  inheritable.

It explicitly does **not** mean (locked design decisions):

- **Not a GUI.** TUI + CLI only for now. The audience is headless homelab/rack boxes over
  SSH; the terminal is the right surface, and the single-binary fleet/cluster story is the
  differentiator a GUI would not improve.
- **Not a chat app, and holds no knowledge.** Owners point their own harness/agent at the
  box; llmtune serves a fast model at a URL and nothing more. There is no in-app chat and
  no local "brain" - the BC-250 reference corpus is the sibling `wikitune` tool, and grounded
  Q&A belongs to whatever agent the owner connects. All that effort goes to the endpoint
  surface instead.
- **Not a "fits / recommended" oracle.** UMA + context + KV quant + the BIOS memory carve
  decide what runs; a green checkmark would lie. llmtune shows the arithmetic and the
  levers; the owner decides.

---

## 2. The journey (what a new owner walks through)

1. **Onboard** - `llmtune setup` on a bare box: detect the BC-250, build the llama.cpp
   engine if absent, locate or create `models_dir`, generate the systemd unit, write config.
2. **Get a model** - search Hugging Face from the TUI, pick a quant, download with progress.
3. **Understand the fit** - the memory-math card: weights + KV at this ctx/quant vs. the UMA
   budget, and the headroom left over. No verdict.
4. **Run it well** - swap to it with the resolved per-arch profile (the crown jewel),
   auto-reverting on failure.
5. **Point a harness at it** - the endpoint card: the OpenAI-compatible URL, the served
   model id, copy-paste snippets.
6. **Know it is fast** - bench / history / telemetry, surfaced (auto-offer a bench after the
   first load; compare quants; inform rollback).

Steps 1-3 are the new headline. SPEC treated them as "M9 polish" and "open questions"; they
are the actual product. Inference orchestration (SPEC M0-M5) is the engine under step 4.

**As shipped:** steps 1, 3, 4, 5, 6 are live (`setup`, the memory-math card, build-id swap,
the `endpoint` surface, and `compare`/auto-bench). Step 2 (Hugging Face search/download) is
**deferred** - the model still comes from the on-disk `models_dir`. The swap-on-demand proxy
(`llmtune proxy`) shipped as an extra: one OpenAI port that loads the requested model on
demand, so a single box looks like it hosts the whole library.

---

## 3. Pillars

### 3.1 llama.cpp build manager (the new move 1) - [SHIPPED]

*Shipped: `llmtune build list/install/update/rollback`, `builds.toml` seed + `Builder`
seam, versioned atomic install, doctor build-deps check. The shipped seed tracks `ref =
"master"` (every install still pins to a concrete commit) pending the bless process in §7.*

"Pull, build if not found, update, roll back" means llmtune **owns the llama.cpp
toolchain** for the BC-250. This is the single biggest friction on the platform and the
precondition for anyone-but-the-author to use the tool. It also subsumes the profile
portability fix: profiles stop pointing at host paths like `/path/to/llama-mtp-cross/...`
and instead reference a **build id** this manager resolves.

MTP merged into upstream `llama.cpp` on 2026-05-16 (PR #22673), so there is **no fork**: a
single upstream Vulkan build off a recent `main` has MTP (`--spec-type draft-mtp`), Vulkan, and RPC
(clusters) all in mainline. The build manager is therefore simpler than "manage a fork"
ever was: one remote, pinned refs, atomic versioned install.

**`builds.toml` (shipped, known-good; user-overridable)** - builds as data, same philosophy
as `profiles.toml`:

```toml
[[build]]
name       = "vulkan"
git_url    = "https://github.com/ggml-org/llama.cpp"
ref        = "<pinned-known-good-sha-or-tag>"   # known-good for BC-250 / gfx1013
cmake_flags = "-DGGML_VULKAN=ON -DGGML_RPC=ON -DLLAMA_CURL=OFF"
out_bins   = ["llama-server", "llama-cli", "rpc-server"]
```

The community inherits "which ref + which flags build a working BC-250 llama," not just
"which flags run a model." One default build covers single-node serving, MTP draft, and RPC
clusters.

**Versioned artifacts, atomic install:**

```
/var/lib/llmtune/builds/<name>/<ref>/{llama-server,llama-cli,rpc-server}
/var/lib/llmtune/builds/<name>/current -> <ref>/      # symlink
```

A profile's `build = "vulkan"` resolves through `current` to the binary. Build into a temp
dir; flip `current` only on success - an interrupted compile never corrupts the live build
(same discipline as the crash-safe history writes).

**Lifecycle (CLI; mirrored in the TUI):**

```
llmtune build list                 # installed builds, current ref, available update
llmtune build install <name>       # clone/fetch -> configure -> compile -> install -> flip current
llmtune build update  <name>       # fetch next known-good ref -> rebuild -> flip current (old kept)
llmtune build rollback <name>      # flip current back to the prior version (instant, no recompile)
```

- **Background + crash-safe.** Compiling on the BC-250 (Zen2 4c/8t) is minutes-long; reuse
  the bench/swap background-thread + progress-overlay pattern. Keep the last N versions.
- **`update` tracks pinned known-good refs by default** (decision locked), with
  `--ref <sha>` as an explicit escape hatch to chase `main`. Upstream is a fast-moving
  target again post-MTP, and a new `main` can regress the gfx1013 Vulkan path; pinned-by-
  default keeps "update" safe and keeps **bench-informed rollback** meaningful: "v-new is
  12 percent slower than v-old, roll back?"
- **System build-deps: detect and instruct, do not drive the package manager** (decision
  locked). A Vulkan build needs cmake, a C++ toolchain, Vulkan headers/loader, and
  `glslc`/shaderc. `doctor` checks each and prints the exact install command for the
  detected distro. llmtune owns everything *inside* the build dir; it never runs
  apt/pacman on the user's system.

**Verify, do not assume:** MTP lives in the sampling/spec layer and should be
backend-agnostic, but whether the draft-head path runs clean on the Vulkan/gfx1013 backend
(vs. silently CPU-falling-back and tanking the acceptance win) is exactly what the first
bench should confirm. This is a concrete `doctor` + first-bench check and a real reason the
bench corpus exists.

### 3.2 First-run setup wizard - [SHIPPED]

*Shipped: `llmtune setup` (`--print`/`--yes`/`--user`/`--models-dir`) - preflight, generate
the base systemd unit, ensure the models dir, write a starter `fleet.toml`, point at `build
install vulkan`.*

`llmtune setup`, auto-triggered when no config exists on first launch. Guided, no TOML by
hand:

1. Detect the BC-250 + GPU stack (runs `doctor`).
2. No llama build found -> offer `build install vulkan` (3.1), with the deps check first.
3. Locate `models_dir`, or create one.
4. **Generate the systemd unit** for the llama service (today the unit is *assumed* to
   exist - this is a real gap; the wizard writes it).
5. Write `~/.config/llmtune/fleet.toml` (implicit localhost node) and seed
   `profiles.toml`.

Target: from a bare box to a served model with no file hand-edited.

### 3.3 Hugging Face model search + download - [DEFERRED]

*Not built (a deliberate call): heaviest remaining piece (live HF API, download
management, and the UMA "fits" honesty problem) with no clear pull yet. The model comes from
the on-disk `models_dir` for now. The design below stands for when it's picked up.*

HF exposes a public, no-auth HTTP API; search-and-pull in the TUI is straightforward and
replaces a curated catalog entirely (users decide, no false "fits"
claims):

- **Search:** `GET https://huggingface.co/api/models?search=<q>&filter=gguf&limit=N`
- **List quant files with real byte sizes:**
  `GET https://huggingface.co/api/models/{repo}/tree/main` -> entries `{path, size}`
  (LFS blobs carry size), so every quant (Q4_K_M, IQ2_XXS, ...) is shown with its size.
- **Download:** `GET https://huggingface.co/{repo}/resolve/main/{file}` with a progress bar
  (HTTP range/resume for interrupted pulls).
- **Gated repos:** optional `hf_token` in config -> `Authorization: Bearer <token>`. Public
  first.

After download the profile auto-resolves by the GGUF arch, so a pulled model is immediately
loadable. TUI flow: search -> pick repo -> pick quant (sizes shown) -> download -> appears
in the model list with its profile resolved.

### 3.4 Memory-math card (the signature feature) - [SHIPPED]

*Shipped: `src/mem.rs` (KV/GQA/quant-aware estimate + UMA budget from sysfs + signed
headroom), `llmtune mem <model> --ctx --kv`, and the memory section in the TUI model card,
computed node-side so it crosses SSH. The pre-download range-GET variant ships with HF search
(3.3, deferred); on-disk math is live.*

The platform's worst weakness - "16 GiB does not mean a 16 GiB model runs" - becomes
llmtune's signature honesty. We already parse the GGUF header, so the card shows the
arithmetic and names the levers, with **no verdict**:

```
weights      8.2 GiB
KV @32k q4_0 1.1 GiB        (levers: ctx, KV quant -ctk/-ctv)
working set  9.3 GiB
---------------------------
UMA budget  ~13.4 GiB       (VRAM 8.0 + GTT 5.4, from sysfs)
headroom    ~4.1 GiB        <- how much is left over
```

- **KV estimate:** `bytes = 2 * n_layers * n_kv_heads * head_dim * ctx * bytes_per_elem`,
  with `head_dim = embedding_length / head_count`, `n_kv_heads = attention.head_count_kv`,
  and `bytes_per_elem` from the KV quant in the resolved profile (f16 = 2.0, q8_0 ~ 1.06,
  q4_0 ~ 0.56). All fields come from the GGUF metadata KV store. Re-renders live as the
  profile's `-c` / `-ctk` / `-ctv` change.
- **UMA budget from real sysfs**, not a "16 GiB" guess:
  `/sys/class/drm/card*/device/mem_info_vram_total` (the BIOS UMA carve) and
  `mem_info_gtt_total` (the GTT spill llama.cpp Vulkan can also use); `*_used` for current
  pressure. The BIOS carve is `memtune`/`biostune` territory - the card names that lever
  too.
- **Before download, too.** For an HF search result, **range-GET the first ~512 KB** of the
  `resolve/main/{file}` URL - the GGUF metadata KV store is at the file head, so arch +
  `block_count` + `head_count` + `head_count_kv` + `embedding_length` + `context_length`
  parse without pulling the multi-GB body. Same parser, on-disk and pre-download.

This is the feature a community burned by silent OOM and mystery slowdowns will love: the
memory math the platform makes confusing, shown plainly, with the knobs labeled.

### 3.5 Profiles: portability + flag migration - [SHIPPED]

*Shipped: `Profile.build` + `Profile::launch()` resolve a `build = "vulkan"` reference through
the build manager; the seed migrated to build-ids with host paths removed and MTP via
`--spec-type draft-mtp` (verified on a BC-250).*

- **Build-id resolution** (3.1) removes host-specific `bin`/`ld_path` from `profiles.toml`;
  a profile names a `build` and the manager resolves the binary. This is the portability
  fix from SPEC section 14, done properly.
- **Flag for mainline MTP.** Current upstream enables MTP with `--spec-type draft-mtp
  --spec-draft-n-max 1` (verified on a BC-250) - the same spelling the seed
  always used. (A bare `--mtp` existed only in the initial #22673 merge and was folded into
  the `--spec-type` framework by #23269; a brief mis-migration to `--mtp` was reverted.)
  Resolution (arch-family prefix, `_default` fallback with a "no known profile" warning) is
  unchanged.

### 3.6 Endpoint surface - [SHIPPED]

*Shipped: `llmtune endpoint` (+ `--json`) and a TUI endpoint overlay (key `o`).*

The product's real output is "a fast model at a URL your tools can hit." Make that
excellent. After a load, surface prominently:

- the OpenAI-compatible base URL (`<llama_url>/v1`), the served model id, health;
- **copy-paste snippets** - curl, the OpenAI SDK, "point your agent here" - so connecting an
  external harness is a 10-second job.

### 3.6a Swap-on-demand proxy - [SHIPPED]

*Was parked; shipped 2026-06-30 as `llmtune proxy`.* A stable llmtune port that swaps
llama-server to the requested model then proxies - making a one-model-at-a-time 16 GiB box
look like it hosts a whole library. `GET /v1/models` lists the whole `models_dir`; an
inference call swaps-then-proxies by the request's `model` field; responses stream (SSE
passes through). Sequential, binds `127.0.0.1` by default (no auth in v1 - warns on an
external bind).

### 3.7 Chat - [REMOVED]

The in-TUI chat pane was cut (2026-07-02) when llmtune's scope narrowed to a pure inference
engine. Even a thin sanity pane invites the system-prompt / sampling / persona / history
creep that belongs to a real harness, and it muddied the product's one job. Verifying "does
it talk" is what the endpoint surface (3.6) and a one-line `curl` snippet are for; the owner
converses through their own agent pointed at the served URL. No `node ask`, no chat view.

### 3.8 Bench / history / telemetry, surfaced - [SHIPPED]

*Shipped: `llmtune compare [filter]` (quant/throughput families), the auto-bench nudge after
a fresh load, build-tagged bench records (`Record.build` via `build::current_version`), and a
bench-informed `build rollback` perf delta.*

The engine is strong; make it visible.

- **Auto-offer a bench** right after the first successful load of a model ("benchmark it
  now? [b]") so history fills without ceremony.
- **Quant comparison** - when history holds multiple quants of the same repo, show the
  tok/s-vs-size trade so "is IQ2 worth it over Q4" is answerable from data.
- **Rollback is bench-informed** (3.1): a build update that regresses tok/s surfaces the
  delta and offers the one-key undo.

---

## 4. Reordered roadmap (vs. SPEC milestones)

SPEC's M0-M5 (config, GGUF parse, profiles, swap, bench, TUI, fleet, cluster) stays the
engine and is largely built. This doc reprioritizes the **product** layer that sits on top:

| Phase | Scope | Status |
|---|---|---|
| **P1 - Usable by anyone** | Build manager (3.1) + setup wizard (3.2) + profile build-id resolution and flag migration (3.5). | **DONE** |
| **P2 - Loved** | HF search + download (3.3) + memory-math card (3.4) + endpoint card (3.6). | **PARTIAL** - memory card + endpoint DONE; HF search **deferred** |
| **P3 - Sharp** | Bench surfacing (3.8) + bench-informed build rollback. | **DONE** |
| **Extra** | Swap-on-demand proxy (3.6a). | **DONE** |
| **Deferred** | HF search (3.3). | not committed |
| **Removed (2026-07-02)** | In-TUI chat pane (3.7); any local knowledge "brain" / RAG; optional web UI. | out of scope |

P1 is the precondition for the word "people" in the brief. P2 is the headline. P3 makes the
unique asset (the bench history) visible. Everything but HF search is on `main`; the chat pane
and the brain were cut on 2026-07-02 when the scope narrowed to a pure inference engine
(knowledge is the sibling `wikitune` tool; conversing is the owner's own harness).

---

## 5. Data formats and resolution chain

- `~/.config/llmtune/builds.toml` - managed builds (3.1); overrides the shipped seed.
- `~/.config/llmtune/profiles.toml` - per-arch launch profiles; `build = "<name>"` +
  flags; overrides the shipped seed.
- `~/.config/llmtune/fleet.toml` - nodes/clusters (SPEC 4.3); optional `hf_token`.
- `/var/lib/llmtune/builds/<name>/<ref>/` + `current` symlink - versioned binaries.
- `/var/lib/llmtune/<node>/history.json` - bench history (SPEC 9).

Resolution: model -> GGUF arch -> `Profile` (flags + `build` id) -> build manager `current`
-> binary. The systemd drop-in is rendered from that resolved binary + flags + the model
path (SPEC 6).

---

## 6. Decisions locked

- TUI + CLI only; no GUI for now.
- No in-app chat and no knowledge brain (2026-07-02) - llmtune is a pure inference engine;
  `wikitune` is the reference corpus and the owner's own harness does the conversing.
- No "fits/recommended" verdict; show the memory math and the levers.
- HF live search + download (no curated catalog).
- llmtune manages the llama.cpp build lifecycle (install / update / rollback).
- Single upstream build; no fork (MTP is mainline since 2026-05-16, PR #22673).
- `update` tracks pinned known-good refs by default; `--ref <sha>` escape hatch.
- System build-deps: detect + instruct; never drive the user's package manager.
- Raw llama-server endpoint shipped; swap-on-demand proxy shipped too (was parked).
- HF search deferred (2026-06-30): no clear pull yet; revisit before committing the effort.

## 7. Open questions

- **Known-good ref cadence.** Who/what advances the pinned `vulkan` ref in `builds.toml`,
  and how is "known-good for gfx1013" certified - a manual bless after a bench passes, or an
  automated smoke (`doctor` + a short bench above a tok/s floor) gating a new pin?
- **Build artifact size / GC.** *Resolved:* the build manager keeps `DEFAULT_RETAIN = 3` prior
  versions (plus `current`), reaping older ones via the durable install ledger; `--retain`
  overrides. Open sub-question: whether 3 is the right default given compile-output size.
- **HF download location + dedup.** Into `models_dir` directly; how to handle re-pulls and
  partial/resumed downloads cleanly.
- **MTP-on-Vulkan reality (gfx1013).** *Resolved (2026-06-30, verified on a live BC-250 with
  the actual upstream build `build-vk @ d14ce3dab` in a maintenance window):* the MTP draft
  head runs on the Vulkan backend (`Vulkan0 : AMD BC-250 (RADV GFX1013)`) - `creating MTP
  draft context` + `adding speculative implementation 'draft-mtp'`, **no CPU fallback**. Draft
  acceptance is workload-dependent: ~65% on a fresh creative prompt (39/60), up to ~100% on
  the production workload; ~31 tok/s gen at Q8 with the draft on. MTP stays on by default for
  MTP-tensor models. The verification also caught a flag bug: current upstream uses
  **`--spec-type draft-mtp --spec-draft-n-max 1`**, not the bare `--mtp` from the initial
  #22673 merge (folded into the `--spec-type` framework by the #23269 clean-up; current
  upstream rejects `--mtp`). The seed profiles were corrected.

## 8. References

- SPEC: [`SPEC.md`](SPEC.md) - architecture, data model, transports, cluster, build order.
- Upstream MTP: llama.cpp PR #22673 (merged 2026-05-16), refactored into the `--spec-type`
  framework by #23269; current flag `--spec-type draft-mtp --spec-draft-n-max 1` (the bare
  `--mtp` is rejected). Verified on a BC-250 (gfx1013): draft on Vulkan, ~100% acceptance.
- Sibling Cachenetics BC-250 tools: `memtune` (memory-timing TUI/bench template), `biostune`
  (UMA carve / BIOS settings - the memory-budget lever named in 3.4).
