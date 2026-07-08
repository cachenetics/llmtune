#!/bin/sh
# agentic-smoke: prove the read-side of docs/agentic-bringup.md works with
# NO TTY - every read/status/list command runs with stdin closed, exits 0,
# and prints exactly one parseable JSON document on stdout; the error and
# refusal paths honor the exit-code convention (1 = error, 2 = refusal)
# and emit the {"error", "refused"} object on stderr under --json.
#
# Safe to run against a live setup: reads only (no loads, no boots, no
# service changes). Commands whose subsystem is unconfigured on this host
# (e.g. `netboot status` without a [netboot] block) are accepted as a
# CLEAN JSON error (exit 1 + parseable stderr object) - that path is part
# of the contract too.
#
# Usage: scripts/agentic-smoke.sh [path-to-llmtune-binary]

set -u

BIN="${1:-${LLMTUNE_BIN:-target/release/llmtune}}"
if [ ! -x "$BIN" ]; then
    BIN="target/debug/llmtune"
fi
if [ ! -x "$BIN" ]; then
    echo "no llmtune binary (build first, or pass a path)" >&2
    exit 1
fi

command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 1; }

pass=0
fail=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# run_read <name> <args...>: exit 0 + stdout parses as JSON, OR a clean
# JSON error (nonzero exit + stderr parses as the {"error"} object).
run_read() {
    name="$1"; shift
    out="$tmp/out"; err="$tmp/err"
    # shellcheck disable=SC2086
    "$BIN" --json "$@" >"$out" 2>"$err" </dev/null
    code=$?
    if [ "$code" -eq 0 ]; then
        if jq -e . "$out" >/dev/null 2>&1; then
            echo "[ok]   $name (exit 0, stdout is JSON)"
            pass=$((pass + 1))
        else
            echo "[fail] $name: exit 0 but stdout is not one JSON document:"
            sed 's/^/       /' "$out" | head -5
            fail=$((fail + 1))
        fi
    else
        if jq -e '.error' "$err" >/dev/null 2>&1 \
           && ! grep -q . "$out"; then
            echo "[ok]   $name (unavailable here: exit $code, clean JSON error)"
            pass=$((pass + 1))
        else
            echo "[fail] $name: exit $code without a parseable stderr error object:"
            sed 's/^/       /' "$err" | head -5
            fail=$((fail + 1))
        fi
    fi
}

# expect_code <want> <name> <args...>: exact exit code + stderr error object
# with the right "refused" flag ("2" -> true, else false), stdout empty.
expect_code() {
    want="$1"; name="$2"; shift 2
    out="$tmp/out"; err="$tmp/err"
    "$BIN" --json "$@" >"$out" 2>"$err" </dev/null
    code=$?
    want_refused="false"
    [ "$want" -eq 2 ] && want_refused="true"
    if [ "$code" -ne "$want" ]; then
        echo "[fail] $name: exit $code, wanted $want"
        sed 's/^/       /' "$err" | head -3
        fail=$((fail + 1))
    elif ! jq -e ".refused == $want_refused and (.error | type == \"string\")" \
             "$err" >/dev/null 2>&1; then
        echo "[fail] $name: stderr is not {\"error\", \"refused\": $want_refused}:"
        sed 's/^/       /' "$err" | head -3
        fail=$((fail + 1))
    elif grep -q . "$out"; then
        echo "[fail] $name: failure leaked non-JSON data onto stdout"
        fail=$((fail + 1))
    else
        echo "[ok]   $name (exit $want, {\"refused\": $want_refused} on stderr)"
        pass=$((pass + 1))
    fi
}

echo "== agentic smoke: $BIN =="

# --- reads: every one must run without a TTY and speak JSON ---------------
run_read "doctor"               doctor
run_read "node list"            node list
run_read "node served"          node served
run_read "node status"          node status
run_read "node history"         node history
run_read "node gpu"             node gpu
run_read "node endpoint"        node endpoint
run_read "node server status"   node server status
run_read "fleet status"         fleet status
run_read "fleet leaderboard"    fleet leaderboard
run_read "models list"          models list
run_read "profile list"         profile list
run_read "build list"           build list
run_read "compare"              compare
run_read "cluster list"         cluster list
run_read "netboot status"       netboot status
run_read "netboot nodes"        netboot nodes
run_read "netboot image list"   netboot image list

# --- the exit-code convention ---------------------------------------------
# bare `llmtune --json` would be the interactive TUI: refusal, exit 2.
expect_code 2 "bare --json refuses the TUI"
# nonexistent model: plain error, exit 1.
expect_code 1 "mem on a nonexistent model errors" mem no-such-model-zzz

echo "== $pass passed, $fail failed =="
[ "$fail" -eq 0 ]
