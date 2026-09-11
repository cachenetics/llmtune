# Netboot / fleet mode

`llmtune` runs in two shapes. **One-box**: installed on a BC-250 itself,
driving that board's own GPU - no config file needed. **Fleet**: `llmtune`
on a spare control host PXE-boots a rack of *diskless* BC-250s and drives
every one of them - serve, swap, bench, telemetry - from a single TUI/CLI.
This doc covers the fleet shape: how netboot works, how to stand it up, and
the gotchas that aren't obvious from the command list in the README.

Reference implementation note: the netboot control plane (this doc) is
llmtune's own code. The *image* a board boots is typically built from a
companion NixOS flake (declarative, reproducible); see
[`docs/agentic-bringup.md`](agentic-bringup.md) for the non-interactive
bring-up recipe and `docs/designs/` for the full design rationale this doc
summarizes.

## Architecture: the boot chain

```
board firmware (PXE)
  -> dnsmasq, proxyDHCP mode
       answers ONLY the PXE options (client-arch, iPXE user-class); your
       LAN's existing DHCP server still assigns the IP. Safe to run on a
       shared network - it will not fight your router.
  -> hands the board ipxe.efi over dnsmasq's built-in TFTP
  -> iPXE re-requests, tagged by its user-class this time
  -> dnsmasq points it at llmtune's embedded HTTP server: /boot.ipxe
  -> llmtune serves /boot.ipxe (kernel + NFS-root cmdline + OC profile),
     then streams /vmlinuz and /initramfs-nfs.img over plain HTTP
  -> board boots a diskless image, root mounted read-only over NFS
  -> the booted board is discovered from the dnsmasq lease file and
     auto-registers into fleet.toml as an `ssh` node
  -> it now flows into `llmtune fleet` / `llmtune node` like any other node
```

Everything except the LAN's own DHCP is llmtune's: it orchestrates a real
`dnsmasq` and a real NFS export rather than reinventing either, but it owns
the HTTP artifact/iPXE server itself (`tiny_http`, already a dependency),
so it controls exactly what each board boots.

The config generators (dnsmasq.conf, the NFS export line, the iPXE script,
the systemd unit) are pure functions - `llmtune netboot init` always
*previews* them; nothing is written until you pass `--apply`.

## Prerequisites

* A spare host on the same LAN segment as the boards, with the interface
  they'll PXE from.
* `[netboot]` in `fleet.toml`: `interface` and `server_ip` auto-detect from
  the host's default route if omitted (`subnet` then derives from
  `server_ip`) - but if llmtune can't see a default route at `init` time
  (multiple NICs, no gateway configured yet, some VM/container setups),
  detection fails with a clear error telling you to set them by hand.
  That's expected behavior, not a sign something's broken. Everything else
  - ports, paths, the OC profile, lease-file location for discovery - has a
  sane default; see the field docs in `src/config.rs` or
  `llmtune netboot init --help`.
* A diskless image (`llmtune netboot image build`, below) and a model
  library the control host will NFS-export (`llmtune models add ...`).
* Boards that can PXE off EFI (client-arch 7/9) and reach the control host
  over the LAN.

## Walkthrough

```sh
llmtune netboot init --apply         # write dnsmasq.d/NFS-export/systemd-unit config, reload
llmtune netboot image build --apply  # build the diskless image (see "Image build" below)
llmtune netboot up                   # start the boot stack, open the firewall, self-check reachability
llmtune netboot boot <node>          # arm + power-cycle a specific board (see "Node lifecycle")
llmtune netboot nodes --register     # fold newly-booted boards into fleet.toml as ssh nodes
llmtune fleet status                 # now drive the rack like any other fleet nodes
llmtune fleet bench-all
llmtune netboot down                 # tear down exactly what `up` started
```

`llmtune netboot status` gives one view of the whole stack at any point:
per-service state, firewall port state, the NFS export, the staged image,
and a live reachability check - reach for it before re-running `up` blind.

### Image build

`llmtune netboot image build --apply` produces the diskless image as an
**atomic (kernel, initrd, boot.ipxe) triple**, sealed under a sha256
manifest. This is deliberate: the one bug that costs the most real time on
this hardware is serving an `init=` path from one build alongside an
`initrd` from a *different* build - it boots stage 1 fine, then fails to
find that init inside the squashfs, and the board just shows "stage 2 init
script not found" (reads like a hang, isn't one). llmtune's manifest makes
that mismatch impossible to stage: the three artifacts are always taken
from one build and verified together, never hand-copied individually.

### Node lifecycle: arm, boot, console

* **`llmtune netboot arm <node>`** sets the board's EFI `BootNext` to its
  iPXE entry over SSH (discovered from the node's own `efibootmgr` output)
  and verifies by reading it back. This is a **one-shot boot override,
  never a BIOS/SMM write** - it never touches `BootOrder` or a setup
  variable. `--disarm` clears it if you need to cancel before it's
  consumed.
* **`llmtune netboot boot <node>`** arms, then triggers power: a
  configured `power_cmd` (smart-plug hook) if you have one, otherwise it
  prints the manual instruction. This has to be a **cold** power cycle - a
  warm `systemctl reboot` does not reliably reset these boards' RTL8168
  PHY, so iPXE never gets a DHCP link and the boot just stalls silently.
* **`llmtune netboot console <node>`** receives the board's `netconsole`
  stream (kernel + journald-via-kmsg over UDP, default port 6666) and
  prints it live - useful for watching a boot with no serial cable
  attached. `--since`, `--file`, and `--duration-secs` filter/capture it.

### Model library

Models live in one NFS-exported library on the control host
(`llmtune models list/add/rm`); nodes mount it **read-only**. This is what
lets a 10-11 GB model fit on a 14 GiB board: llama.cpp reads weights
straight off NFS into GPU/unified memory, so there's no second copy sitting
in RAM the way a tmpfs-embedded model would need. The cost is swap latency
(roughly gigabit-NFS-bound - tens of seconds for a multi-GB model), not
memory.

## Security notes

* The NFS export CIDR **is** the trust boundary: every host inside it can
  read the entire diskless root image. Keep it scoped to the actual fleet
  subnet, not a wide LAN range.
* `llmtune netboot up` opens the host firewall for exactly the ports it
  needs (HTTP artifact server, NFS, and DHCP/TFTP when dnsmasq is enabled),
  additively, and self-checks reachability from **off-host** - a
  `curl localhost` success does not prove another machine on the LAN can
  actually reach the boot server (a silently-dropping firewall rule
  upstream of llmtune's own has bitten this before).
* Everything else llmtune serves (the TUI/CLI's own OpenAI-compatible
  endpoint) stays loopback-only by default; exposing it to the LAN is a
  separate, explicit `llmtune endpoint expose on`.

## Known rough edges

* The boot server's default HTTP port (8090) can collide with a local
  inference server if you're running one on the same box - `up` currently
  reports the resulting `EADDRINUSE` as "a service is down" rather than
  "something else already has this port." If `up`/`status` looks wrong,
  check `ss -ltnp` on that port before assuming the boot stack itself is
  broken.
* `netboot image build` on a rootless `nix-portable` setup currently needs
  a couple of manual unblocks (root-owned system home, sudo-less nix
  store) rather than working end to end unattended.

Both are tracked; if you hit either, it's not something you're doing
wrong.

## See also

* [`docs/agentic-bringup.md`](agentic-bringup.md) - the non-interactive,
  no-TTY fleet bring-up recipe (what an agent/script runs end to end).
* [`docs/designs/bc250-fleet-inference.md`](designs/bc250-fleet-inference.md)
  and [`docs/designs/bc250-fleet-plan.md`](designs/bc250-fleet-plan.md) -
  the original design brief and milestone breakdown, including the
  hands-on-hardware detail (exact GPU driver versions, kernel pin, the
  full story behind each gotcha above) that this doc distills.
