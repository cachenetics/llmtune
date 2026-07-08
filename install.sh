#!/usr/bin/env bash
# Build and install llmtune -- the BC-250 inference engine and fleet control plane.
#
#   ./install.sh            build (release) + install to /usr/local/bin
#   ./install.sh --setup    also run first-run setup (models dir + systemd service)
#   ./install.sh --check    run the preflight (doctor) after installing
#
# Installs one binary (`llmtune`) plus its bundled architecture profiles.
#
# Needs: a Rust toolchain (cargo) to build, sudo to install.
set -euo pipefail
cd "$(dirname "$0")"

PREFIX="${PREFIX:-/usr/local}"
DESTDIR="${DESTDIR:-}"
BIN="$DESTDIR$PREFIX/bin/llmtune"
DO_SETUP=0
DO_CHECK=0
for a in "$@"; do
  case "$a" in
    --setup) DO_SETUP=1 ;;
    --check) DO_CHECK=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

command -v cargo >/dev/null 2>&1 || {
  echo "error: cargo (Rust toolchain) not found -- install Rust from https://rustup.rs" >&2
  exit 1
}

echo ">> building release binary..."
cargo build --release

echo ">> installing $BIN (sudo)..."
sudo install -Dm755 target/release/llmtune "$BIN"
sudo install -Dm644 profiles.toml "$DESTDIR$PREFIX/share/llmtune/profiles.toml"

echo ">> installed: $("$BIN" --version)"

if [ "$DO_CHECK" = 1 ]; then
  echo ">> preflight (doctor):"
  "$BIN" doctor || true
fi
if [ "$DO_SETUP" = 1 ]; then
  echo ">> first-run setup:"
  sudo "$BIN" setup --yes
fi

echo
echo "Done. Launch the TUI:      llmtune"
echo "  preflight the box:       llmtune doctor"
echo "  first-run setup:         sudo llmtune setup"
echo "  add and serve a model:   llmtune models add <url> && llmtune node load <name>"
