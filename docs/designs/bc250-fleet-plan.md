# BC-250 fleet inference - milestone plan

Breakdown of `bc250-fleet-inference.md` section 4 into shippable milestones.
Each is one focused MR. Reference implementation: the companion `bc250-nixos`
image repo. Acceptance runs use a live netbooted test node (netboot image, NFS
library, declarative llama-server).

## M1 - Unified NixOS-safe serving path (spec 3.1, 3.2, 3.4-swap)

The one serving/swap code path works on BOTH a normal install (writes
/etc) and the read-only-/etc netboot image (writes /run). Folds the
reference `bc250-swap` helper's behavior into `llmtune node load`.

- Detect a read-only /etc/systemd/system (NixOS marker, /nix/store
  symlink, or ro mount) and write the model drop-in to
  `/run/systemd/system/<unit>.d/` instead of failing.
- Drop-in scan/winning-name/clear operate over the union of systemd
  drop-in dirs (/etc, /run, vendor), so a swap cannot be shadowed by a
  drop-in in another root (e.g. the image's `bc250-swap` override.conf).
- GPU env preservation: the drop-in's `Environment=` reset must not
  strip the base unit's gfx1013 Vulkan env (VK_DRIVER_FILES -> mesa26
  RADV ICD, VK_LOADER_LAYERS_DISABLE=*, LD_LIBRARY_PATH). Keys the
  profile does not set are carried over from the unit's own definition.
- Privileged ops run directly when already root (the netboot image has
  no sudo); sudo otherwise.
- `setup`/`doctor` follow the same root detection.

Acceptance: unit tests for root detection, unit-env parsing, GPU-env
merge, drop-in scan across roots; `make check` green; on the test node a
`node load` swap succeeds via a /run drop-in and the server stays on
the GPU (non-destructive verify: swap to another library model and
back).

## M2 - Image pipeline: atomic co-built triple (spec 3.4, 4.1)

`llmtune netboot image build` produces the diskless image with the
reference config baked in (mesa26 RADV + vulkan llama.cpp +
declarative GPU llama-server + NFS automount + netconsole + arieltune
tune + pinned kernel), by driving the bc250-nixos flake (or an
llmtune-owned copy of it).

- Output (kernel, initrd, boot.ipxe/init=) as ONE build artifact set;
  record initrd sha256 + init= path together in the image manifest.
- Staging refuses a mismatched pair (unit test: mismatch is rejected).

Acceptance: `image build` yields a manifest-verified triple; a
hand-mismatched triple fails to stage; served boot.ipxe always carries
the init= of the initrd it serves.

## M3 - Boot server control plane (spec 3.3 server-side, 3.5, 4.2)

`llmtune netboot up/down/status`: HTTP artifact server + NFS export of
the model library + optional dnsmasq proxyDHCP.

- `up` opens the firewall (HTTP/NFS/DHCP ports) and self-checks
  reachability from off-host (not just local curl).
- boot.ipxe generated from the CURRENT image's co-built init=.
- NFS export management (exports entry, exportfs, nfs-server).

Acceptance: `netboot up --json` reports every service + a remote
reachability check; `down` reverts; a cold-booted board reaches
stage 2 first try.

## M4 - Node lifecycle: arm/boot/console (spec 3.4, 4.3)

- `netboot arm <node>`: one-shot `efibootmgr -n` over SSH (never a
  BIOS/SMM write).
- `netboot boot <node>`: arm + trigger power via per-node `power_cmd`
  (smart-plug hook), else prompt for a manual COLD cycle (warm reboot
  does not reset the RTL8168 PHY).
- `netboot console <node>`: netconsole UDP receiver + live stream
  (port the netconsole-debug.nix receiver side).
- `netboot nodes --register` folds booted boards into fleet.toml.

Acceptance: on the test node, arm + cold cycle boots into the image and the
board auto-serves; console streams the boot log; node appears in
`fleet status`.

## M5 - Model library management (spec 4.5)

`llmtune models list/add/rm` on the control host's NFS export; nodes
see it read-only. Guard rm against the currently-served model on any
known node.

Acceptance: add/rm reflect on the node's automount without a remount;
`models list --json` matches node-side discovery.

## M6 - TUI: fleet/netboot dashboard (spec 4.6)

Extend the cockpit netboot view: boards booting, served model, GPU
temp/power, tok/s, one-key serve/swap/bench per node, console pane, a
"boot a board" flow. Local mode: single-node dashboard. Style: colored
section headers, plain body.

Acceptance: drive a full boot->serve->swap->bench from the TUI against
the test node without dropping to the CLI.

## M7 - Agentic surface audit (spec 4.7)

Sweep: every read has `--json`, every action has `--yes`; document the
non-interactive fleet bring-up recipe end to end.

Acceptance: scripted (no TTY) bring-up + serve + bench-all passes.

Order: M1 (this MR) -> M2 -> M3 -> M4 (hardware loop closed) -> M5 ->
M6 -> M7. M2-M4 need the live board for their acceptance runs; M5-M7
mostly do not.
