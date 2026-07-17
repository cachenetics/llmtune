# llmtune BC-250 fleet inference - design of record

Status: implementation brief. Context: this captures a full hands-on bring-up
that took the BC-250 from bricked firmware -> a working PXE-netbooted,
NFS-model-library, GPU-serving inference node driven from llmtune. Every
"fact" below was verified on real hardware on 2026-07-07. The working reference
implementation lives in a companion NixOS image repo (referred to below as
`bc250-nixos`). This brief tells llmtune what to absorb so a user never has to
re-derive any of it.

## 1. The product goal

llmtune is the self-hostable control plane for BC-250 (Cyan Skillfish / gfx1013)
inference. TWO deployment modes, ONE tool, extremely friendly TUI + agentic CLI:

- **Local mode**: `llmtune` installed on a single BC-250 (its own disk) drives
  inference on that box's GPU. `llmtune` (TUI) or `llmtune node ...` (CLI) just
  works: detect GPU, serve a model on the GPU, swap models, bench.
- **Fleet / netboot mode**: `llmtune` on a control host (the PXE/NFS boot server)
  PXE-netboots BC-250s as **diskless** nodes, serves the whole model library over
  **NFS**, and drives every node from one place (serve/swap/bench/telemetry).

"Extremely user friendly": the TUI is the default surface; the CLI mirrors it 1:1
and emits `--json` everywhere for agentic use (an agent should be able to stand up
a fleet, boot a board, and serve a model with a handful of non-interactive calls).

## 2. The proven pipeline (what "works" means, concretely)

Control host = the boot server (example address 192.0.2.10 below). Node = a
BC-250 board. End state that WORKS today:
- Node PXE-netboots a diskless NixOS image -> tmpfs root, GPU tuned on boot.
- Node NFS4-automounts the library `<control-host>:/srv/llmtune/models` (ro) at
  `/var/lib/llmtune/models` -> all models available, **not copied into the 14 GB
  RAM** (~7 GB used serving one model vs ~11 GB with a tmpfs copy).
- Node serves gemma-4-E4B on the **BC-250 GPU** at ~66 tok/s over the network on
  `:8080` (OpenAI-compatible), benched from the control host via
  `llmtune fleet bench-all` (67.9 gen t/s).
- `bc250-swap <name>` on the node swaps to any library model (frees the previous
  model's GPU+RAM, streams the new one from NFS).

## 3. Hard-won facts to BAKE IN (do not let users re-discover these)

### 3.1 gfx1013 GPU serving needs mesa >= 26
- nixpkgs-24.11 mesa 24.2.8 RADV does NOT recognise the BC-250 chip:
  `amdgpu: unknown (family_id, chip_external_rev): (143,132)` -> llama.cpp prints
  "no usable GPU found" and silently falls back to CPU (15 tok/s).
- Unstable **mesa 26.1.4** RADV enumerates it as `AMD BC-250 (RADV GFX1013)`
  (16.9 GiB). Both the vulkan llama.cpp and mesa26 are PREBUILT in cache.nixos.org
  (no compile): `llama-cpp.override { vulkanSupport = true; }` + `mesa` from
  nixos-unstable.
- Serve env (force mesa26 per-service, NOT as the system GL stack -> avoids ABI
  clash with the 24.11 base):
  - `VK_DRIVER_FILES=<mesa26>/share/vulkan/icd.d/radeon_icd.x86_64.json`
  - `LD_LIBRARY_PATH=<llama>/lib:<mesa26>/lib`
  - `VK_LOADER_LAYERS_DISABLE=*` (the 24.11 device-select layer breaks enumeration)
  - `-ngl 99`
- Sanity check: `llama-cli --list-devices` must print `Vulkan0: AMD BC-250 (RADV
  GFX1013)`. If it only lists CPU, the ICD/env is wrong.
- gemma needs `--cache-ram 1024 --cache-reuse 256 --no-mmap` or it OOMs the UMA.
  (These are already the arch-profile flags in llmtune's `profiles.toml` seed.)

### 3.2 Serving on the diskless image must be DECLARATIVE (read-only /etc)
- On NixOS, `llmtune setup` / `llmtune node load` FAIL: they write units into
  `/etc/systemd/system`, which is read-only. Error: "No such file or directory".
- Working pattern (see `bc250-nixos/modules/llama-vulkan.nix`): a declarative
  `systemd.services.llama-server` unit with the GPU env above, and a `bc250-swap`
  helper that writes a drop-in to **`/run/systemd/system/llama-server.service.d/`
  (writable)** and `systemctl restart`s. llmtune still drives+benches it because it
  probes `:8080` and the unit is named `llama-server.service`.
- IMPLICATION FOR llmtune: `llmtune node load`/swap must detect a read-only /etc
  (NixOS) and fall back to a `/run` drop-in instead of failing. This makes the ONE
  code path work on both a normal install (writes /etc) and the netboot image.
  (Shipped: MR !66. Field note: the netboot image DOES have sudo at
  `/run/wrappers/bin/sudo`, but the run-direct-when-euid==0 path is still correct
  for root service contexts.)

### 3.3 NFS model library (the RAM win)
- Mount `server:/export` ro at the models dir; llama reads weights straight into
  GPU/unified memory - no duplicate file copy in RAM (~1x vs 2x). This is what lets
  the 27B/35B (10-11 GB) models fit in 14 GB.
- Node NFS client: `nfs-utils` in systemPackages + `boot.kernelModules = [ "nfs" ]`
  + a `fileSystems` entry (`nfs4`, `ro`, `nofail`, `_netdev`, `x-systemd.automount`).
  DO NOT use `boot.supportedFilesystems = [ "nfs" ]` as a LIST - and be careful:
  declaring the fileSystems entry already sets `supportedFilesystems.nfs=true`;
  keep it out of the initrd (`boot.initrd.supportedFilesystems` must stay
  overlay/squashfs/tmpfs).
- Server side (control host): `nfs-utils`, `/etc/exports`
  `<dir> <lan>/24(ro,sync,no_subtree_check,root_squash)`, `exportfs -ra`,
  `systemctl enable --now nfs-server`, open ufw 2049+111.
- Swap cost: reading a model over gigabit NFS ~110 MB/s (~40 s for 5 GB, ~90 s for
  10 GB). One model resident at a time; freed on swap.

### 3.4 Netboot mechanics + the ONE bug that dominated the whole session
- iPXE chainload from the node ESP (`\EFI\ipxe\ipxe.efi`, a `dhcp||exit; chain
  http://server:8090/boot.ipxe ||exit` build), armed with a LONE one-shot
  `efibootmgr -n <num>` (NEVER a BIOS/SMM write - an SMM NVAR-append bricked the box
  to no-POST earlier; recovered only by external SPI reflash).
- Boot server: HTTP (`vmlinuz` + `initrd` + `boot.ipxe`). The initrd embeds the
  squashfs (RAM boot). Model delivered via NFS (or HTTP for local mode).
- **THE BUG (cost ~3 hours, many power-cycles): init/initrd MISMATCH.** After a
  rebuild, `boot.ipxe`'s `init=/nix/store/<system>/init` MUST come from the SAME
  build as the served `initrd`. Serving an init= from build A with build B's initrd
  -> stage-1 mounts fine, then can't find that init in the squashfs -> screen shows
  **"stage 2 init script not found"** (reads like a stage-1 hang). Correctly-paired
  images boot first try, every time.
  - MANDATE for llmtune: whatever stages/serves images MUST treat (kernel, initrd,
    boot.ipxe/init=) as an ATOMIC co-built triple. Take all three from ONE build's
    `.#netbootKernel .#netbootRamdisk .#netbootIpxe` output; record + verify the
    initrd sha256 and the init= path together; never hand-copy one without the
    others. Expose `llmtune netboot image ...` such that a mismatch is impossible.
- Netboot TRIGGER needs a COLD power-cycle: a warm `systemctl reboot` doesn't
  reliably reset the RTL8168 PHY, so iPXE often can't get DHCP link. Cold power-off
  resets it. There's no BMC on the BC-250 -> recommend a networked smart-plug for
  remote power (llmtune could integrate a power-control hook: `power_cmd` per node).
- Headless boot debug WITHOUT serial: `netconsole` + journald `ForwardToKMsg=yes`
  ships kernel+systemd log to the control host over UDP. See
  `bc250-nixos/modules/netconsole-debug.nix`. llmtune's netboot should offer
  `llmtune netboot console <node>` (stand up the receiver + show the live stream).

### 3.5 Boot reliability / gotchas
- ufw on the control host silently dropped :8080/:8090 from the LAN (SYN dropped
  pre-accept, zero HTTP log). `llmtune netboot up` must open the firewall for its
  HTTP/NFS/DHCP ports and self-check reachability FROM another host, not just local
  curl.
- Kernel pinned to cachyos-bore 7.0.9 (7.0.11+ breaks BC-250 SDMA). arieltune tunes
  the GPU on boot (pins gfx 1500 MHz). Both already in the reference image.

## 4. What llmtune already has vs what to add

Already present (repo): `llmtune netboot` (init/up/down/status/nodes/image),
`llmtune node` (serve/load/bench/gpu/swap/endpoint), `llmtune fleet`, `profiles.toml`
(per-arch flags incl. the gemma cache-ram flags), `builds.toml` (managed llama
builds), a ratatui TUI, `fleet.toml` (ssh nodes). So the SCAFFOLD exists; this
session proved the real-hardware path and found the exact working config. Wire that
in:

1. **Image pipeline (`llmtune netboot image build`)**: produce the diskless image
   with the reference config baked in - mesa26 RADV + vulkan llama.cpp + the
   declarative GPU `llama-server` + NFS automount + `bc250-swap` + netconsole +
   arieltune tune + the pinned kernel. Port `bc250-nixos/modules/*` and `flake.nix`
   into an llmtune-owned image definition (or have llmtune drive that flake). Output
   the co-built (kernel, initrd, init=) triple and stage them ATOMICALLY (3.4).
2. **Boot server (`llmtune netboot up/down/status`)**: HTTP artifact server + NFS
   export of the model library + optional dnsmasq proxyDHCP; open + verify firewall;
   the served `boot.ipxe` is generated from the current image's co-built init=.
3. **Node lifecycle**: `llmtune netboot nodes` lists booted boards (from DHCP leases
   / discovery), `--register` adds them to fleet.toml; `llmtune netboot arm <node>`
   sets the one-shot BootNext over SSH; `llmtune netboot boot <node>` arms + triggers
   power (via the node's `power_cmd`/smart-plug hook, else prompts for a manual cold
   cycle); `llmtune netboot console <node>` = netconsole receiver + live stream.
4. **Serving that works on NixOS**: make `llmtune node load`/swap detect read-only
   /etc and use a `/run` drop-in (3.2). Fold the `bc250-swap` logic into
   `llmtune node load` so there is ONE swap path (local + netboot). Ensure the GPU
   env (3.1) is applied whether serving locally or on a node.
5. **Model library management**: `llmtune models` (list/add/rm on the NFS export) so
   the library is managed from the control host; nodes see it read-only over NFS.
6. **TUI**: a Netboot/Fleet view - boards booting, their model + GPU temp/power +
   tok/s, one-key serve/swap/bench per node, a console pane (netconsole), a "boot a
   board" flow. Local mode: a single-node dashboard.
7. **CLI/agentic**: `--json` on every read; non-interactive `--yes` on every action;
   an agent can do `llmtune netboot up`, `... boot bc250-x`, `... node load qwen3.5`,
   `... fleet bench-all --json` without a TTY.

## 5. Acceptance criteria (must verify on a live node)

- Local: on a BC-250, `llmtune node serve <gemma>` serves on the GPU (verify
  `llama-cli --list-devices` shows RADV GFX1013; tok/s >> CPU 15). `llmtune node load
  <qwen>` swaps, freeing the prior model.
- Fleet: `llmtune netboot up` stands up HTTP+NFS (+firewall). `llmtune netboot image
  build` yields a co-built triple; staging can NOT mismatch init/initrd (add a test
  that a mismatched pair is rejected). A board arms + boots (cold cycle) + auto-
  serves on GPU + appears in `llmtune fleet status`. `llmtune fleet bench-all --json`
  returns tok/s. `llmtune netboot console <node>` streams the boot log.
- Swap works on the netboot image (read-only /etc) via a /run drop-in.
- No step requires the user to know any of section 3 - it's all handled.

## 6. Constraints / house rules
- Names llmtune/arieltune/Cachenetics are trademarks (NOTICE); public forks rename.
- GPL-2.0-only. Commits/MRs are authored by Cachenetics. No emojis in records.
- Route changes through the project's merge-request flow. Keep the ratatui TUI
  style (headers colored, plain body). CLI is agentic-first (`--json`).
- Reference image + working modules: the companion `bc250-nixos` image repo.
  Acceptance runs need a live netbooted node.
