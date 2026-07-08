# Agentic fleet bring-up (no TTY required)

llmtune has two surfaces over the same operations: the TUI for humans, and an
agentic CLI for scripts and LLM agents. This document is the CLI recipe: a
non-interactive caller (no terminal, no prompts) takes a control host and a
box of BC-250 boards to a serving, benchmarked fleet, `--json` end to end.

The read-side of this recipe is exercised by `scripts/agentic-smoke.sh`
(run it any time; it proves no command needs a TTY).

## The convention

- Every read/status/list command supports the global `--json` flag and prints
  exactly ONE JSON document on stdout (streaming commands print one JSON
  object per line - NDJSON). Progress/notes go to stderr in `--json` mode;
  stdout never mixes text and JSON.
- Every action that is destructive or would prompt has a non-interactive
  escape hatch: `--yes` (power actions, `setup`, `models rm`), `--apply`
  (preview-first pipelines: `netboot init`, `netboot image build`), `--force`
  (`models rm` served-guard), `--allow-version-skew` (`cluster up`),
  `--skip-check` (`netboot up` reachability). Prompts never block a non-TTY
  caller: they either refuse (exit 2, see below) or skip the step and say so.
- Exit codes:
  - `0` success.
  - `1` error (bad input, IO, unreachable node, failed build).
  - `2` refusal: a safety guard declined and state was left untouched -
    the `models rm` served-guard, the `netboot up` off-host reachability
    failure, `netboot boot` without a TTY or `--yes`, the cluster
    build-version-skew guard. The message names the override flag.
- On failure with `--json`, stderr carries ONE machine-parseable object:

  ```json
  {"error": "<message chain>", "refused": false}
  ```

  stdout stays data-only, so `llmtune --json ... | jq` is always safe.
- Two commands are inherently interactive/long-running and exempt from
  `--json` data shapes: the TUI (refused outright under `--json`, exit 2)
  and the servers (`proxy`, the hidden `netboot serve`, which systemd owns).

All commands below are runnable as-is with stdin closed (`< /dev/null`).

## 0. Configure `[netboot]`

The control host needs a `[netboot]` block in `~/.config/llmtune/fleet.toml`
(or a file passed via `--config`). Minimal working example (placeholder
values; substitute your own network and paths):

```toml
[netboot]
interface  = "enp6s0"            # LAN-facing NIC
server_ip  = "192.0.2.10"        # this host, as boards will reach it
subnet     = "192.0.2.0"
models_dir = "/srv/llmtune/models"          # the NFS-exported model library
flake      = "/path/to/your/netboot-flake"  # NixOS image flake (co-built triple)
mac_ouis   = ["58:11:22"]        # BC-250 NIC OUI(s), for board discovery
```

The image source is bring-your-own: `flake` points at a NixOS flake YOU
provide, whose outputs `.#netbootKernel`, `.#netbootRamdisk` and
`.#netbootIpxe` are co-built from one system closure (kernel, initrd, and the
iPXE script with its `init=` argument). Alternatively `iso =
"/path/to/cachyos.iso"` selects the legacy ISO pipeline. Without one of the
two, `netboot image build` refuses with a pointer to this config.

Every `netboot` command refuses with a clear error (exit 1) until this
exists - probe first:

```sh
llmtune --json netboot status < /dev/null
```

## 1. Build + stage the image

```sh
# preview (JSON plan; nothing executed)
llmtune --json netboot image build

# build: ONE nix invocation co-builds kernel + initrd + netboot.ipxe and
# seals them under a sha256 manifest; --stage activates it in the same step
llmtune --json netboot image build --apply --stage

# inspect what exists / what is active
llmtune --json netboot image list
```

`image stage <id>` verifies the manifest (initrd sha256 + `init=` pairing)
and refuses a mismatched triple - a board can never be handed an initrd that
does not match its kernel arguments.

### DEPLOYMENT NOTE - the image bakes a specific llmtune build

The netboot image contains the llmtune binary the flake saw at build time.
Shipping new llmtune features TO THE NODES (everything from fleet M1-M7:
the /run drop-in swap path, GPU-env preservation, `--json` shapes the SSH
transport parses) therefore requires an image rebuild + re-netboot:

```sh
llmtune --json netboot image build --apply --stage
llmtune --json netboot boot <node> --yes     # cold cycle into the new image
```

Until that is done, a board that netbooted an older image runs the older
baked binary; node-side behavior (swap paths, remote `--json` parsing) is
only current after the rebuild + reboot.

## 2. Stand the boot server up

```sh
# writes dnsmasq/exports/unit configs (previewed without --apply)
llmtune --json netboot init --apply

# start HTTP + NFS (+ dnsmasq when enabled), open LAN-scoped firewall
# ports, then PROVE reachability from off-host (a localhost curl proves
# nothing about an interfering firewall)
llmtune --json netboot up
```

`up --json` prints one summary object: services started/enabled, whether the
exports drop-in was added, firewall backend + rules added, and the
reachability report. If the off-host check fails (an interfering firewall,
a service not listening) `up` exits 2 with `{"refused": true}` on stderr -
the stack stays up for diagnosis, `netboot down` reverses exactly what `up`
changed, `--skip-check` accepts an unproven stack deliberately.

```sh
llmtune --json netboot status    # re-check any time (includes reachability)
llmtune --json netboot down      # reverse everything up did
```

## 3. Stock the model library

```sh
llmtune --json models add /path/to/Qwen3.5-9B-IQ4_NL.gguf
llmtune --json models add https://host/path/model.gguf   # plain http(s) too
llmtune --json models list
```

Adds are GGUF-validated and committed atomically, so a node's NFS automount
never sees a half-written file. `models rm <name>` confirms interactively
(non-interactive callers pass `--yes`) and refuses (exit 2) while any fleet
node is SERVING that file - a llama-server holds it mmapped over NFS;
`--force` overrides both for offline surgery.

## 4. Boot a board

```sh
# one-shot arm: EFI BootNext -> the node's iPXE entry, verified by re-read
# (never BootOrder, never a BIOS/SMM variable)
llmtune --json netboot arm bc250-a

# arm + power. REALITY CHECK: a netboot needs a COLD power cycle - a warm
# reboot does not reset the RTL8168 PHY, so iPXE gets no DHCP link. With a
# per-node power_cmd (smart-plug hook) in fleet.toml the cycle is automated;
# --yes is required for any non-TTY caller (without it: refusal, exit 2).
llmtune --json netboot boot bc250-a --yes
```

Without a `power_cmd`, `boot --json` reports `"power": "manual"` (exit 0)
and prints the cold-cycle instruction on stderr - the agent must tell a
human to pull power.

```sh
# watch it boot without a serial cable (netconsole UDP receiver).
# --json = NDJSON: {"ts_unix_ms":..., "kernel_ts":..., "line":"..."} per line
llmtune --json netboot console bc250-a --duration-secs 120

# discover booted boards and fold them into fleet.toml as ssh nodes
llmtune --json netboot nodes --register
```

## 5. Serve and bench

`node` commands default to the local node; `--node <name>` drives any
configured fleet node from the control host (the CLI equivalent of drilling
into a node in the TUI):

```sh
llmtune --json node --node bc250-a list          # models it sees (NFS library)
llmtune --json node --node bc250-a load qwen3.5  # hot-swap, auto-revert on fail
llmtune --json node --node bc250-a status        # served / health / last bench
llmtune --json node --node bc250-a gpu           # clock + temperature
llmtune --json node --node bc250-a endpoint      # its OpenAI-compatible URL
llmtune --json node --node bc250-a bench         # bench the served model

llmtune --json fleet status                      # one row per node
llmtune --json fleet bench-all                   # leaderboard across the fleet
```

A failed `node load` (model did not take; auto-reverted) exits non-zero in
both modes; the JSON payload carries `ok`/`reverted`/`detail`.

## The end-to-end script

```sh
#!/bin/sh -e
# from zero to a benched fleet, no TTY anywhere
llmtune --json netboot image build --apply --stage
llmtune --json netboot init --apply
llmtune --json netboot up
llmtune --json models add /srv/staging/Qwen3.5-9B-IQ4_NL.gguf
llmtune --json netboot boot bc250-a --yes        # needs power_cmd configured
sleep 90                                          # or watch `netboot console`
llmtune --json netboot nodes --register
llmtune --json node --node bc250-a load qwen3.5
llmtune --json fleet bench-all > bench.json
```

## Command coverage summary

| group | reads (`--json`) | actions (non-interactive escape) |
|---|---|---|
| top level | `doctor`, `compare`, `mem`, `endpoint` | `setup --yes` |
| `node [--node N]` | `list`, `served`, `status`, `history`, `gpu`, `endpoint`, `server status` | `load`, `unload`, `bench`, `server start/stop/restart` (none prompt) |
| `fleet` | `status`, `leaderboard` | `bench-all`, `swap-all` (none prompt) |
| `models` | `list` | `add`; `rm` (served-guard refusal, `--force`) |
| `netboot` | `status`, `nodes`, `image list` | `init --apply`, `up` (`--skip-check`), `down`, `image build --apply [--stage]`, `image stage`, `arm`, `boot --yes`, `console` (NDJSON stream) |
| `cluster` | `list`, `status` | `up` (skew refusal, `--allow-version-skew`), `down` |
| `profile` / `build` | `list`, `show` | `set`, `set-model`, `install`, `update`, `rollback` (none prompt) |

Node-local settings surfaces (`node expose/api-key/boot-restore/profile
set-model/fit/build-version`) run on the box they configure; `--node <remote>`
rejects them with a pointer to run them on the node.
