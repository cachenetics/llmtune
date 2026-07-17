# llmtune codebase audit - 2026-07-06 (7-track)

Whole-codebase audit (~14.2k LOC, 36 files) run as 7 parallel per-subsystem
passes. Axes: security, correctness, robustness (panics/hangs/leaks), and
public-readiness. Verdict: the core is solid (memory math verified against
llama.cpp block formats, the primary SSH path is argv-safe and well-tested, the
TUI panic surface is clean). The weak areas cluster into auth/key handling, a
GGUF-parser DoS, lock/state-machine safety, and a set of config/UX foot-guns.

Tally: 0 critical, 6 high, ~26 medium, ~27 low, ~18 info.

Severity = impact x reachability. "Self-inflicted" = reachable only via the
operator's own config (foot-gun, not a remote-attacker path), but still relevant
because a public user pastes configs.

---

## P0 - auth/key (fix before any network exposure)

- **P0-A [HIGH] proxy has no inbound auth and does not forward the key** -
  `proxy.rs:98-145,200-205`. `proxy_to_backend` sets only Content-Type, never the
  Bearer key, so when llama-server is launched with `--api-key` every forwarded
  request 401s (proxy non-functional). Conversely the proxy enforces no auth on
  inbound clients, so anyone who reaches the proxy port reaches the model with no
  key - a bypass of the key that protects the backend. Fix: require + constant-time
  compare an inbound key, and forward it to the backend.
- **P0-B [HIGH/MED] api_key leaks (argv, world-readable drop-in, backup, 0644)** -
  `nodeops.rs:142-144`, `swap.rs:103-110,419-423`, `settings.rs:75-80` (root cause
  `paths.rs:134-153`). The key is interpolated into `ExecStart=... --api-key {k}`
  (visible in `ps` and in the world-readable systemd drop-in and its `.prev`
  backup), and `settings.toml` is created with umask-default `0644` on first write.
  Fix: pass the key via `EnvironmentFile=` (or systemd `LoadCredential`) at 0600,
  never argv; chmod drop-in + backup + settings 0600.
- **P0-C [HIGH] cluster rpc-server binds 0.0.0.0 unauthenticated** -
  `cluster.rs:296-313`. llama.cpp `rpc-server -H 0.0.0.0` has no auth and is an
  arbitrary-memory/RCE surface upstream documents as unsafe on untrusted networks;
  the bind is hardcoded and unwarned, exposed for the cluster's lifetime. Fix:
  make the bind a `[[cluster]]` field defaulting to the fleet-facing interface (or
  loopback), and warn on `cluster up`.

## P1 - DoS on untrusted model files

- **P1-D [HIGH/MED] GGUF parser hang + stack overflow on a crafted .gguf** -
  `model.rs:97-104`. Unknown array element type makes `rd_val` consume zero bytes
  and return `Ok(None)`, so `for _ in 0..cnt` (cnt up to u64::MAX) spins forever
  without hitting EOF; nested arrays recurse unboundedly -> stack-overflow abort. A
  ~30-byte file in the models dir hangs/crashes every `node list`/swap/TUI path;
  GGUFs are downloaded from HF (untrusted). Fix: bail on unknown types, require
  forward progress, add a recursion-depth cap.

## P1 - state-machine & concurrency safety

- **P1-E [MED] swap strands the node on a systemctl error after stage** -
  `swap.rs:212-213`. After `stage()` removes the prior drop-in and writes the new
  one, a failing `reload_restart` propagates with no rollback; auto-revert covers
  only the health-poll path. Fix: rollback+reload on any early error in the
  stage->reload scope.
- **P1-F [MED] locks are check-then-act, non-exclusive, and Drop deletes a peer's
  lock** - `lock.rs:80-107`, callers `nodeops.rs:121-124,209-212`,
  `build.rs:309-312,407-410`. `*_running()` freshness probe then `File::create`
  (not `O_EXCL`) - two concurrent swaps both "acquire"; both hold the same path so
  the first to drop unlinks the shared lock. Also `build.rs` ledger: a corrupt
  `installs.json` -> gc keep-set `{current}` -> every rollback version deleted. Fix:
  `create_new(true)`/`flock(LOCK_EX|LOCK_NB)` with stale-steal; never remove a lock
  you did not create; on unparseable ledger, error/rebuild instead of gc.

## P1 - TUI blocks the event loop (public users will hit these)

- **P1-G [HIGH] expose/auth toggles run a full model reload inline on the UI
  thread** - `ui/mod.rs:574,657`. `set_exposure`/`restage_served` stop+reload the
  served model (tens of seconds to minutes for a 30B) synchronously in the key
  handler, no busy overlay, no abort. Fix: worker-thread + SwapState like
  `start_swap`.
- **P1-H [HIGH] fleet bench-all: minutes-long inline loop, no confirm, errors
  swallowed** - `ui/mod.rs:165-176`. One Shift+B freezes the UI for N x minutes
  unabortably and always reports success (`let _ =`). Fix: confirm gate + worker
  thread + surface per-node failure.
- **P1-I [MED] more inline-on-event-loop blocking** - `do_swap_all`
  (`ui/mod.rs:714-740`), `FleetApp::refresh` per-node SSH (`127-146`, up to ~8s per
  down node), `draw_confirm_server` spawns `systemctl is-active` **every frame**
  (`overlays.rs:640-644`), remote `refresh`/`do_unload`/`o`/`y` inline
  (`417-423,499-511,1101,1138`). Fix: background-worker pattern (as
  `poll_remote_telemetry`); probe server-active once on entering the mode.
- **P1-J [MED] auth-off is one unconfirmed keypress while exposed** -
  `ui/mod.rs:911-917`. Clears the key + restages, leaving a 0.0.0.0 endpoint
  unauthenticated with no confirm (expose-on has a confirm; disabling auth while
  exposed is the same risk). Fix: confirm when `settings::exposed()`.

## P2 - config foot-guns

- **[MED] explicit `--config <missing>` silently ignored** - `config.rs:114-127` /
  `cli.rs:24-25`. Falls through to synthesized localhost; every subsequent command
  actuates the wrong target. Fix: error if an explicitly-passed path is not a
  readable file.
- **[MED] unknown `transport` value silently parses as Local** - `config.rs:16-24`.
  A typo of `ssh` makes a remote node the local box - swaps/systemd land on the
  wrong machine. Fix: reject unknown transport at load.
- **[MED] llama_unit interpolated into root path writes unvalidated** -
  `setup.rs:157,200-217`. A unit containing `/` or `..` turns setup into an
  arbitrary root file write via `sudo tee`. Fix: validate
  `^[A-Za-z0-9:_@.\-]+\.service$`.
- **[MED] cluster down actuates the LOCAL head when the head is remote** -
  `cmds/cluster.rs:272-302`. `up` guards `head==Local`; `down` does not, so it
  strips the local machine's drop-in for that unit name. Fix: mirror the up-path
  guard.
- **[MED] under `sudo llmtune`, config files in the invoker's ~/.config are created
  root-owned** - `paths.rs:82-98`, `setup.rs:226-230`, `settings.rs:60-97`. The
  documented non-sudo configure flow then EACCESes. Fix: chown to SUDO_UID:GID
  after root writes under a sudo-invoker home.

## P2 - injection foot-guns (self-inflicted via own config)

- **[MED] cluster run_on second SSH path is not shell-quoted** - `cluster.rs:274-293`.
  `rpc_bin` (from fleet.toml) reaches a remote login shell unquoted; `;`/`$()`/
  backticks execute as the ssh user. The primary path uses `sh_quote`
  (`transport.rs:249-258`); this one bypasses it. Fix: route through `sh_quote`.
- **[MED] ssh option-argument injection via host/user beginning with `-`** -
  `transport.rs:274-279`, `cluster.rs:265-271`. No `--` separator before the ssh
  target. Fix: insert `--`, or reject leading-`-` host/user.
- **[LOW] git ref/url and sudo mkdir/chown args parsed as options (leading `-`)** -
  `build.rs:567,606-616`, `setup.rs:184-186`. `--upload-pack=` on `git ls-remote`
  is command execution; `chown` without `-h` follows a symlink at models_dir. Fix:
  reject leading `-`, insert `--`, require absolute non-symlink path.
- **[LOW] no explicit SSH StrictHostKeyChecking; behavior inherited from env** -
  `transport.rs:263-285`. Fix: set `-o StrictHostKeyChecking=accept-new`.

## P2 - robustness

- **[MED] no timeouts on spawned git/cmake/bench/ssh** - `build.rs:696-731`,
  `bench.rs:315`, `transport.rs:288-303`. A hung child blocks the CLI/fleet
  indefinitely (and a mid-bench GPU wedge leaves the node serverless). Fix: wait
  with a deadline + kill; add ssh `ServerAliveInterval`.
- **[LOW] RMW lost-updates + corrupt-file field-wipe** - `history.rs:87-95`,
  `profile.rs:105-134`, `paths.rs:175-185` (`edit_toml`). Concurrent writers lose
  records/overrides; a corrupt `settings.toml` is silently replaced by defaults,
  wiping `api_key`/exposure. Fix: flock around the RMW; on parse failure rename
  `.bad` and warn.
- **[LOW] unbounded history growth, O(n) rewrite per append** - `history.rs:87-94`.
  Fix: cap/rotate at N records.
- **[LOW] GGUF length truncated before the 64MB guard on 32-bit** - `model.rs:66`
  (nil impact on x86_64/BC-250 targets). `[LOW]` bench sampler thread leaked when
  spawn fails - `bench.rs:315-321`. `[LOW]` `/proc/meminfo` `?` on a colon-less
  line zeroes sys memory and misreports fit - `mem.rs:155`.

## Public-readiness

- **[INFO] `aln-` api-key prefix is an "arieltune" leftover** - `settings.rs:110-112`.
  Ends up in users' client configs. Fix: `llt-`/`lmt-`.
- **[INFO] em dashes pervasive in help/TUI strings** - repo-wide. Known; scrub
  before public (structural em dashes in embedded manuals excepted).
- Otherwise clean: no hardcoded LAN IPs (test fixtures use RFC-5737 TEST-NET), no
  secrets in source, no AI-authorship tells, no live TODO/FIXME.

## Deliberate designs (flagged, not defects)

- `mem.rs:133-141` credits all `gtt_used` as freeable (single-tenant assumption).
- `profile.rs:40-53` bare `llama-server` PATH fallback (makes a missing build fail
  visibly via health-revert).
- `endpoint`/CLI/TUI print the key in cleartext (operator-facing copy-paste). A
  `--show-key`/masking gate is the cheap improvement.

---

## Method

7 parallel per-subsystem review passes (build/exec, serving/swap/proxy,
ssh/cluster, config/state, bench/telemetry, cli, tui); each pass read its files
in full and reported findings with file:line + confidence. Read-only; no code
changed during the audit.
