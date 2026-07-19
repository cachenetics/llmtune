#!/usr/bin/env bash
# build-ternary-bc250.sh - build the PrismML llama.cpp fork (ternary Q2_0 kernels)
# for the AMD BC-250 (gfx1013), NATIVELY on the board, with the Vulkan backend.
#
# Why native: the ternary Q2_0_g128 format needs PrismML's custom kernels (stock
# llama.cpp cannot load it). Building ON the BC-250 also guarantees a matching CPU
# ISA (Zen2 = x86-64-v3, no AVX512) - cross-building on a newer host can bake in a
# too-high ISA marker that makes the binary refuse to start here.
#
# Usage:   ./build-ternary-bc250.sh [install_dir]
# Default install_dir: ~/ternary-llama
set -euo pipefail

REPO="https://github.com/PrismML-Eng/llama.cpp"
COMMIT="9fcaed763ccda38ea81068ad9d7f991aaddca451"   # BC-250-tested; Q1_0/Q2_0 kernels
DEST="${1:-$HOME/ternary-llama}"
SRC="$DEST/src"
JOBS="$(nproc)"

say() { printf '\n== %s\n' "$*"; }

say "0/5  preflight: BC-250 + Vulkan + toolchain"
command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary 2>/dev/null | grep -qi "GFX1013\|BC-250" \
  && echo "  [ok] BC-250 (gfx1013) Vulkan device present" \
  || echo "  [warn] could not confirm a gfx1013 Vulkan device - continuing, but this build targets the BC-250"
miss=""
for t in git cmake gcc g++ glslc; do command -v "$t" >/dev/null 2>&1 || miss="$miss $t"; done
if [ -n "$miss" ]; then
  echo "  [fail] missing tools:$miss"
  echo "  install them, e.g.:"
  echo "    Arch/CachyOS: sudo pacman -S --needed base-devel cmake git shaderc vulkan-headers vulkan-radeon"
  echo "    Debian/Ubuntu: sudo apt install build-essential cmake git glslc libvulkan-dev mesa-vulkan-drivers"
  exit 1
fi
echo "  [ok] toolchain present ($(cmake --version | head -1))"

say "1/5  fetch fork @ ${COMMIT:0:10}  ->  $SRC"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch --depth 1 origin "$COMMIT" && git -C "$SRC" checkout -q "$COMMIT"
else
  mkdir -p "$DEST"
  git clone "$REPO" "$SRC"
  git -C "$SRC" checkout -q "$COMMIT"
fi

say "2/5  configure (Vulkan on, CUDA off, native ISA off for portability)"
cmake -S "$SRC" -B "$SRC/build" \
  -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release

say "3/5  build (this can take 20-40 min on a BC-250 - the Vulkan shaders are the slow part)"
cmake --build "$SRC/build" --target llama-server llama-cli llama-bench -j "$JOBS"

say "4/5  stage binaries + libs -> $DEST/bin"
mkdir -p "$DEST/bin"
# copy binaries + all shared libs, preserving soname symlinks
rsync -a "$SRC/build/bin/" "$DEST/bin/" 2>/dev/null || cp -a "$SRC"/build/bin/. "$DEST/bin/"
# native crt => correct ISA marker; strip a stray one only if present (harmless no-op otherwise)
for b in llama-server llama-cli llama-bench; do
  objcopy --remove-section .note.gnu.property "$DEST/bin/$b" 2>/dev/null || true
done

say "5/5  done"
cat <<EOF

  binaries: $DEST/bin/{llama-server,llama-cli,llama-bench}

  Grab a ternary GGUF (the ~1.71-bit Q2_0 is the one to run):
    huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf -> Ternary-Bonsai-27B-Q2_0.gguf (~7 GB)

  Serve it (all layers on the GPU, ~17 tok/s decode on a BC-250):
    LD_LIBRARY_PATH=$DEST/bin $DEST/bin/llama-server \\
      -m /path/to/Ternary-Bonsai-27B-Q2_0.gguf -ngl 99 -fa on -c 4096 \\
      --host 0.0.0.0 --port 8080

  Bench it:
    LD_LIBRARY_PATH=$DEST/bin $DEST/bin/llama-bench \\
      -m /path/to/Ternary-Bonsai-27B-Q2_0.gguf -ngl 99 -fa 1 -p 256 -n 64

  Note: the DSpark speculative drafter uses a CUDA-only resample path, so on the
  BC-250 (Vulkan) run the plain Q2_0 model without a drafter.
EOF
