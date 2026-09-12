# BC-250 complete bring-up: fresh CachyOS -> liberated board -> serving models

End-to-end guide from a bare BC-250 with a fresh CachyOS install to a fully liberated (40 CU,
8-core) board serving an LLM over the network. Covers [arieltune](https://github.com/cachenetics/project-ariel)
(hardware liberation) then [llmtune](https://github.com/cachenetics/llmtune) (model serving). Written
from a real live bring-up (2026-09-12) - every gotcha below actually happened to someone doing this for
the first time, not theoretical.

Each tool's own README is the authoritative reference for flags/options; this guide is the path through
both, start to finish, with the traps called out inline.

## 0. Before you install anything

Sync package mirrors first. A live CachyOS ISO's package database can be stale enough that a normal
`pacman -S` 404s on a package that's since moved on the mirror:

```sh
sudo pacman -Syyu
```

## 1. Install arieltune

```sh
git clone https://github.com/cachenetics/project-ariel.git
cd project-ariel
./install.sh
```

Installs `arieltune` plus `aputune`/`memtune`/`biostune`/`wikitune` compat symlinks to
`/usr/local/bin`. Confirm it's there: `aputune --help`.

## 2. BIOS: set the UMA carve

**UMA Frame Buffer Size = 512M**, nothing else. This is hard-gated in code (`aputune build`
preflight-checks it and refuses to proceed on any other value) because a different carve breaks IP
discovery and PSP firmware loads in ways that look exactly like a broken patch, wasting a full build
cycle before you find out. 256M (the stock default) is just as wrong as 2G - it's an exact match
requirement, not a minimum.

## 3. Pin the kernel source

`aputune build` needs the `linux-cachyos-bore` PKGBUILD - **not** plain `linux-cachyos` (that variant's
prebuilt fan-driver module won't match; see the known-gap section below) - at exactly version 7.0.9-1.
Newer versions (7.0.11+) regress BC-250 SDMA.

```sh
git clone https://github.com/CachyOS/linux-cachyos.git
cd linux-cachyos
git checkout 791fb8ea6d3cf7c85e596678c25c56fa140591be   # tagged 7.0.9-1 upstream
```

## 4. Build dependencies

```sh
sudo pacman -S --needed gcc15 bc base-devel clang llvm lld pahole rust rust-bindgen rust-src
```

## 5. GRUB kernel command line

```sh
sudo sed -i 's|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT="nowatchdog nvme_load=YES splash loglevel=3 amdgpu.ppfeaturemask=0xfff77ef7 amdgpu.noretry=0 amdgpu.gpu_recovery=1 amdgpu.sched_hw_submission=2 ttm.pages_limit=3588867 ttm.page_pool_size=3588867 iommu=pt amd_iommu=on amdgpu.bc250_flush_by_runlist=1 amdgpu.bc250_sdma_fw=navi12"|' /etc/default/grub
```

If your host already carries other boot flags (disk encryption, `resume=`, etc.), merge them into this
line by hand instead of clobbering it. Drop `mitigations=off` from the flag list above unless this is a
dedicated inference box - it's a throughput trade, not a liberation requirement.

Also set a sane default now, so you're not stuck hand-picking a kernel at every boot later:

```sh
sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT=saved/' /etc/default/grub
echo 'GRUB_SAVEDEFAULT=true' | sudo tee -a /etc/default/grub
sudo grub-mkconfig -o /boot/grub/grub.cfg
```

With `saved` mode, whatever kernel you actually boot becomes the new default automatically - no
`grub-set-default` bookkeeping needed.

## 6. Build and install (~30-40 min)

```sh
sudo aputune build --pkgbuild ~/linux-cachyos/linux-cachyos-bore          # preview
sudo aputune build --pkgbuild ~/linux-cachyos/linux-cachyos-bore --run    # go
```

Installs the patched kernel, arms the 40-CU modprobe.d drop-in, rebuilds the initramfs. Does **not**
run `grub-mkconfig` again after installing the new kernel package - if this is your first liberation
build, re-run `sudo grub-mkconfig -o /boot/grub/grub.cfg` once more after this step so GRUB picks up the
new kernel entry.

## 7. Reboot and verify

```sh
sudo reboot
```

If more than one kernel shows in the GRUB menu (plain `linux-cachyos` and/or `linux-cachyos-lts` from
the original ISO install commonly coexist alongside the new `-bore` one, as separate packages pacman
doesn't touch) - pick `linux-cachyos-bore` explicitly this first time. With `GRUB_DEFAULT=saved` from
step 5, that becomes the new default from here on.

```sh
sudo arieltune apu doctor --verify
```

Expect `liberation series: 27/27 live` and `CUs active: 40/40`.

## 8. Keep the pin from getting silently overwritten

Nothing currently stops a routine `pacman -Syu` from pulling in a newer `linux-cachyos-bore` build and
clobbering your patched kernel. Pin it:

```sh
sudo sed -i '/^#IgnorePkg/a IgnorePkg = linux-cachyos-bore' /etc/pacman.conf
```

## 9. 8-core CPU unlock (optional, separate feature)

The BC-250's CPU ships with 2 of 8 physical cores fused off at the factory (6C/12T stock). This is a
different SMU mailbox path than the GPU liberation above - a separate step.

In the `arieltune` TUI, `Tab` to the **Core Map** panel (make sure keyboard focus is actually on that
panel - key presses go nowhere if focus is elsewhere) and press `[u]` to unlock. Read the status line it
shows.

**Critical:** it needs a **warm** reboot specifically (`sudo systemctl reboot`, not a power-button
cycle) to take effect - a cold boot reverts the mask back to stock every time. Verify after:

```sh
nproc                          # expect 16, not 12
sudo arieltune apu cores verify
```

`[F]` (force-unlock) is a deliberate no-op on the normal stock mask (0x77) - it only ever activates for
a genuinely abnormal mask. If you're on stock, `[u]` is always the right key, not `[F]`.

## 10. Known gap: fan/hwmon control

The public repo's embedded fan-telemetry kernel module (`nct6687`) currently targets the plain
`linux-cachyos` kernel variant, not `-bore` - so on a `-bore` liberation (the correct choice for
everything else above), fan RPM readout and manual fan-curve control won't come up; you'll stay on the
stock read-only sensors and the board's automatic EC fan curve. Doesn't block anything else. Tracked
upstream - check `arieltune`'s issue tracker for current status before assuming it's still open.

## 11. Install llmtune and serve a model

```sh
git clone https://github.com/cachenetics/llmtune.git
cd llmtune
./install.sh --setup
```

Drop `.gguf` models directly into `/var/lib/llmtune/models`, then:

```sh
llmtune node list            # discovered models
llmtune node load <name>     # substring match, no need for the full filename
llmtune node status          # confirm it's actually serving, not just discovered
```

Serves on `127.0.0.1:8080`, OpenAI-compatible at `/v1` - loopback only by default.

## 12. Reach it from another device or a harness

```sh
llmtune endpoint auth on      # generate + apply a key (or `auth off` to stay keyless)
llmtune endpoint expose on    # bind 0.0.0.0:8080 + open the firewall
llmtune endpoint              # prints the live key + ready-to-paste connection snippets
```

Always copy the key straight out of `llmtune endpoint`'s own output rather than retyping it -
`auth on` regenerates a new key every time it's run, so anything copied earlier can go stale. `auth`/
`expose` toggles apply immediately (they force a full model reload internally) - no separate manual
reload needed.

If you see `Failed to reset failed state of unit ... not loaded` at any point during a load/reload,
that's expected on a first-ever load and harmless - it's a defensive pre-step llmtune runs before every
restart and explicitly ignores the result of.

Point any OpenAI-compatible client (a chat UI, a coding harness, anything) at
`http://<this-box's-LAN-IP>:8080/v1` with `Authorization: Bearer <key>` (or no header at all if you left
it keyless). Keyless + exposed means anything on your LAN can reach it unauthenticated - fine on a
trusted home network, worth knowing.

---

Questions or something doesn't match what's here: aibc250 Discord, `#ariel-chat`.
