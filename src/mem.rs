// SPDX-License-Identifier: GPL-2.0-only
//! Memory-fit arithmetic - the honest answer to "will this model run?" on a
//! BC-250, where "16 GiB" lies because it is UMA shared between CPU and GPU.
//!
//! No verdict, just the numbers and the levers: model weights + the KV cache at a
//! given context and KV-quant, versus the real UMA budget read from sysfs, and
//! the headroom left over. The KV estimate comes from the GGUF header dims
//! (parsed by [`crate::model::read_dims`]); the budget from the amdgpu DRM
//! `mem_info_*` counters (the BIOS VRAM carve plus the GTT spill llama.cpp's
//! Vulkan backend can also draw on).

use serde::{Deserialize, Serialize};
use std::path::Path;

/// The header dimensions needed to size a KV cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dims {
    pub n_layers: u64,
    pub n_head: u64,
    /// KV heads (== n_head for MHA; smaller for GQA/MQA).
    pub n_head_kv: u64,
    /// Per-head dimension (key_length if the header gives it, else n_embd/n_head).
    pub head_dim: u64,
    /// Training context from the header, if present.
    pub ctx_train: Option<u64>,
}

/// Bytes per KV element for a llama.cpp cache type. The quantized types store a
/// block of 32 elements plus scale(s); the per-element average is what matters
/// for a cache-size estimate. Unknown / unrecognized -> f16 (the safe upper bound).
pub fn bytes_per_elem(kv_quant: &str) -> f64 {
    match kv_quant.to_lowercase().as_str() {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        // block-of-32 quants: (data bits/8 * 32 + scale bytes) / 32
        "q8_0" => 34.0 / 32.0, // 32 int8 + 1 f16 scale
        "q8_1" => 36.0 / 32.0, // 32 int8 + 2 f16 (scale+min)
        "q5_1" => 24.0 / 32.0, // 20 + 4
        "q5_0" => 22.0 / 32.0, // 20 + 2
        "q4_1" => 20.0 / 32.0, // 16 + 4
        "q4_0" => 18.0 / 32.0, // 16 + 2
        _ => 2.0,
    }
}

/// KV cache size in bytes: 2 (key+value) * layers * kv_heads * head_dim * ctx *
/// bytes_per_elem. Saturating so an absurd ctx can't overflow.
pub fn kv_bytes(dims: &Dims, ctx: u64, per_elem: f64) -> u64 {
    let elems = 2u128
        .saturating_mul(dims.n_layers as u128)
        .saturating_mul(dims.n_head_kv as u128)
        .saturating_mul(dims.head_dim as u128)
        .saturating_mul(ctx as u128);
    (elems as f64 * per_elem) as u64
}

/// Runtime memory the `llama-server` process needs beyond weights + KV - Vulkan
/// staging buffers, per-token activations, context checkpoints, plus breathing
/// room so the box doesn't OOM at the edge. Held back from the usable budget.
pub const RUNTIME_RESERVE: u64 = 1 << 30; // 1 GiB

/// KV-cache bytes per token: 2 (key+value) * layers * kv_heads * head_dim *
/// bytes_per_elem. The per-token cost that a context window multiplies.
pub fn kv_bytes_per_token(dims: &Dims, per_elem: f64) -> f64 {
    2.0 * dims.n_layers as f64 * dims.n_head_kv as f64 * dims.head_dim as f64 * per_elem
}

/// The largest context that fits: given the memory left for the KV cache after
/// weights (`kv_budget_bytes`), how many tokens of cache a given KV quant buys.
/// Capped at the model's trained context (running past it degrades quality with
/// no benefit). Returns 0 if weights already overflow the budget.
pub fn max_ctx(dims: &Dims, kv_budget_bytes: i64, kv_quant: &str) -> u64 {
    if kv_budget_bytes <= 0 {
        return 0;
    }
    let per_token = kv_bytes_per_token(dims, bytes_per_elem(kv_quant));
    if per_token <= 0.0 {
        return 0;
    }
    let ctx = (kv_budget_bytes as f64 / per_token) as u64;
    match dims.ctx_train {
        Some(t) if t > 0 => ctx.min(t),
        _ => ctx,
    }
}

/// The UMA memory pool the Vulkan backend can draw on. The amdgpu sysfs counters
/// (`vram_*`/`gtt_*`) describe the GPU's *addressing* view, but on a UMA APU both
/// draw from the same physical DRAM shared with the OS - so `vram_total +
/// gtt_total` (16.5 GiB here: a 0.5 GiB BIOS carve + a 16 GiB GTT ceiling that
/// already spans ~all of RAM) double-counts and overstates what a model can use.
/// The honest budget comes from `/proc/meminfo`: what the kernel can actually
/// hand out right now.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmaBudget {
    /// BIOS-carved VRAM (the dedicated UMA frame buffer), bytes.
    pub vram_total: u64,
    /// GTT - system RAM the GPU can map, bytes.
    pub gtt_total: u64,
    pub vram_used: u64,
    pub gtt_used: u64,
    /// `/proc/meminfo` MemTotal - physical system RAM (the carve is excluded), bytes.
    #[serde(default)]
    pub sys_total: u64,
    /// `/proc/meminfo` MemAvailable - RAM allocatable now without swapping, bytes.
    #[serde(default)]
    pub sys_available: u64,
}

impl UmaBudget {
    /// The GPU's addressable pool (VRAM carve + GTT). Retained for the fallback
    /// path when `/proc/meminfo` is unreadable (tests / non-Linux); NOT the honest
    /// physical ceiling - it double-counts the shared DRAM.
    pub fn total(&self) -> u64 {
        self.vram_total.saturating_add(self.gtt_total)
    }

    /// Physical unified memory: system RAM + the BIOS VRAM carve (which is
    /// excluded from MemTotal). ~16 GiB on the BC-250 - the real hardware size,
    /// not the 16.5 GiB the amdgpu counters imply.
    pub fn physical_total(&self) -> u64 {
        if self.sys_total > 0 {
            self.sys_total.saturating_add(self.vram_total)
        } else {
            self.total() // no meminfo - degrade to the GPU pool
        }
    }

    /// Memory a model can actually take right now. MemAvailable already nets out
    /// the OS + other processes; we add back the GTT the currently-served model
    /// holds (a swap stops the server and frees it first), then hold back the
    /// runtime reserve. This is a live figure - it moves with real load.
    pub fn usable(&self) -> u64 {
        if self.sys_total > 0 {
            self.sys_available
                .saturating_add(self.gtt_used)
                .saturating_sub(RUNTIME_RESERVE)
        } else {
            self.total() // no meminfo - degrade to the GPU pool
        }
    }
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Read `MemTotal` and `MemAvailable` (bytes) from `/proc/meminfo`. Values there
/// are in kB. Returns None off Linux or if either line is missing.
fn read_meminfo_at(path: &Path) -> Option<(u64, u64)> {
    let s = std::fs::read_to_string(path).ok()?;
    let mut total = None;
    let mut avail = None;
    for line in s.lines() {
        // Skip a malformed (colon-less) line rather than aborting the whole
        // parse: `?` here zeroed BOTH totals on one bad line, and an inflated
        // "everything fits" verdict greenlights an OOM load.
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let kb: Option<u64> = rest.split_whitespace().next().and_then(|v| v.parse().ok());
        match key {
            "MemTotal" => total = kb.map(|k| k * 1024),
            "MemAvailable" => avail = kb.map(|k| k * 1024),
            _ => {}
        }
    }
    Some((total?, avail?))
}

/// Read the UMA budget from a DRM device dir (e.g. `/sys/class/drm/card0/device`).
/// Returns None unless at least the VRAM total is readable. System-RAM fields are
/// filled by [`read_uma`] (meminfo is host-wide, not per-device).
fn read_uma_at(dev: &Path) -> Option<UmaBudget> {
    let vram_total = read_u64(&dev.join("mem_info_vram_total"))?;
    Some(UmaBudget {
        vram_total,
        gtt_total: read_u64(&dev.join("mem_info_gtt_total")).unwrap_or(0),
        vram_used: read_u64(&dev.join("mem_info_vram_used")).unwrap_or(0),
        gtt_used: read_u64(&dev.join("mem_info_gtt_used")).unwrap_or(0),
        sys_total: 0,
        sys_available: 0,
    })
}

/// Read the UMA budget from the first amdgpu DRM card that exposes the counters.
/// None when not on an amdgpu box (so callers degrade to weights+KV only).
pub fn read_uma() -> Option<UmaBudget> {
    let rd = std::fs::read_dir("/sys/class/drm").ok()?;
    let mut cards: Vec<std::path::PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("card") && !n.contains('-')) // cardN, not cardN-eDP
                .unwrap_or(false)
        })
        .collect();
    cards.sort();
    let mut budget = cards
        .into_iter()
        .find_map(|c| read_uma_at(&c.join("device")))?;
    // Fold in the honest physical-RAM view (host-wide, so read once here).
    if let Some((total, avail)) = read_meminfo_at(Path::new("/proc/meminfo")) {
        budget.sys_total = total;
        budget.sys_available = avail;
    }
    Some(budget)
}

/// A full memory-fit estimate for a model at a context + KV quant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemEstimate {
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    pub working_bytes: u64,
    pub ctx: u64,
    pub kv_quant: String,
    /// None when the header lacked the dims to size the KV cache.
    pub dims: Option<Dims>,
    /// None when not on an amdgpu box (no sysfs budget).
    pub budget: Option<UmaBudget>,
    /// usable - working_bytes (signed: negative means it won't fit). None when
    /// the budget is unknown.
    pub headroom_bytes: Option<i64>,
    /// Largest context that fits at `kv_quant` given weights + the usable budget,
    /// capped at the trained context. None without both dims and a budget.
    #[serde(default)]
    pub max_ctx: Option<u64>,
}

/// Build a memory estimate. `dims = None` (header missing the fields) yields a
/// weights-only estimate (kv = 0) so the card still renders something useful.
pub fn estimate(
    weights_bytes: u64,
    dims: Option<Dims>,
    ctx: u64,
    kv_quant: &str,
    budget: Option<UmaBudget>,
) -> MemEstimate {
    let kv = dims
        .as_ref()
        .map(|d| kv_bytes(d, ctx, bytes_per_elem(kv_quant)))
        .unwrap_or(0);
    let working = weights_bytes.saturating_add(kv);
    // Headroom against what a model can actually take now (usable), not the
    // inflated GPU addressing pool. Signed: negative means it won't fit.
    let headroom = budget.as_ref().map(|b| b.usable() as i64 - working as i64);
    // Largest context this KV quant buys: memory left for KV after weights,
    // divided by the per-token cost, capped at the trained context.
    let max_ctx = match (dims.as_ref(), budget.as_ref()) {
        (Some(d), Some(b)) => Some(max_ctx(
            d,
            b.usable() as i64 - weights_bytes as i64,
            kv_quant,
        )),
        _ => None,
    };
    MemEstimate {
        weights_bytes,
        kv_bytes: kv,
        working_bytes: working,
        ctx,
        kv_quant: kv_quant.to_string(),
        dims,
        budget,
        headroom_bytes: headroom,
        max_ctx,
    }
}

/// Extract the serving context (`-c N`) and KV quant (`-ctk TYPE`) from a flags
/// string, so a memory estimate reflects how the model is actually launched.
/// Either may be absent (the caller supplies a default).
pub fn ctx_kv_from_flags(flags: &str) -> (Option<u64>, Option<String>) {
    let toks: Vec<&str> = flags.split_whitespace().collect();
    let mut ctx = None;
    let mut kv = None;
    for (i, t) in toks.iter().enumerate() {
        match *t {
            "-c" | "--ctx-size" => ctx = toks.get(i + 1).and_then(|v| v.parse().ok()),
            // -ctk / --cache-type-k sets the K cache type; V usually matches.
            "-ctk" | "--cache-type-k" => kv = toks.get(i + 1).map(|v| v.to_string()),
            _ => {}
        }
    }
    (ctx, kv)
}

/// GiB rendering helper (binary, 1 decimal).
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Signed GiB (for headroom, which can be negative).
pub fn gib_i(bytes: i64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> Dims {
        // Qwen3-ish small: 28 layers, GQA 16/4 heads, head_dim 128.
        Dims {
            n_layers: 28,
            n_head: 16,
            n_head_kv: 4,
            head_dim: 128,
            ctx_train: Some(40960),
        }
    }

    #[test]
    fn bytes_per_elem_known_types() {
        assert!((bytes_per_elem("f16") - 2.0).abs() < 1e-9);
        assert!((bytes_per_elem("q8_0") - 1.0625).abs() < 1e-9);
        assert!((bytes_per_elem("q4_0") - 0.5625).abs() < 1e-9);
        // unknown falls back to f16 (conservative)
        assert!((bytes_per_elem("wat") - 2.0).abs() < 1e-9);
    }

    #[test]
    fn kv_scales_with_ctx_and_quant() {
        let d = dims();
        // 2 * 28 * 4 * 128 * ctx * bpe
        let at_f16 = kv_bytes(&d, 32768, bytes_per_elem("f16"));
        // hand calc: 2*28*4*128 = 28672 elems/token; *32768 = 939_524_096; *2 B
        assert_eq!(at_f16, 939_524_096 * 2);
        // q4_0 is ~3.55x smaller than f16 (0.5625 vs 2.0)
        let at_q4 = kv_bytes(&d, 32768, bytes_per_elem("q4_0"));
        assert!(at_q4 < at_f16);
        assert!((at_f16 as f64 / at_q4 as f64 - (2.0 / 0.5625)).abs() < 0.01);
        // ctx doubles -> KV doubles
        assert_eq!(kv_bytes(&d, 65536, 2.0), 2 * kv_bytes(&d, 32768, 2.0));
    }

    #[test]
    fn estimate_headroom_sign() {
        let budget = UmaBudget {
            vram_total: 512 * (1 << 20),
            gtt_total: 16 * (1 << 30),
            vram_used: 0,
            gtt_used: 0,
            sys_total: 16 * (1 << 30),
            sys_available: 13 * (1 << 30),
        };
        // usable = 13 avail + 0 gtt_used - 1 reserve = 12 GiB.
        // 9 GiB weights + small KV fits -> positive headroom
        let fits = estimate(
            9 * (1 << 30),
            Some(dims()),
            8192,
            "q4_0",
            Some(budget.clone()),
        );
        assert!(fits.headroom_bytes.unwrap() > 0);
        // 18 GiB weights does NOT fit -> negative headroom (but still computed)
        let over = estimate(18 * (1 << 30), Some(dims()), 32768, "f16", Some(budget));
        assert!(over.headroom_bytes.unwrap() < 0);
    }

    #[test]
    fn max_ctx_is_the_inverse_of_kv_bytes() {
        let d = dims();
        // Give exactly enough KV budget for 8192 tokens at q4_0, expect ~8192.
        let per_tok = kv_bytes_per_token(&d, bytes_per_elem("q4_0"));
        let budget = (per_tok * 8192.0) as i64;
        assert_eq!(max_ctx(&d, budget, "q4_0"), 8192);
        // A smaller KV quant (q4_0 vs f16) buys more context for the same budget.
        assert!(max_ctx(&d, budget, "q4_0") > max_ctx(&d, budget, "f16"));
        // Capped at the trained context (40960 here) even with a huge budget.
        assert_eq!(max_ctx(&d, 1 << 60, "q4_0"), 40960);
        // Weights already over budget -> 0 context.
        assert_eq!(max_ctx(&d, -1, "q4_0"), 0);
    }

    #[test]
    fn usable_and_physical_from_system_ram() {
        // 16 GiB box: 15.5 RAM + 0.5 carve; 13 available, an 11 GiB model in GTT.
        let b = UmaBudget {
            vram_total: 512 * (1 << 20),
            gtt_total: 16 * (1 << 30),
            vram_used: 0,
            gtt_used: 11 * (1 << 30),
            sys_total: 15 * (1 << 30) + 512 * (1 << 20),
            sys_available: 3 * (1 << 30),
        };
        // physical = sys_total + carve = 16 GiB, NOT the 16.5 the GPU pool implies.
        assert_eq!(b.physical_total(), 16 * (1 << 30));
        assert!(b.total() > b.physical_total()); // the double-count we're avoiding
                                                 // usable = avail (3) + gtt the swap frees (11) - reserve (1) = 13 GiB.
        assert_eq!(b.usable(), 13 * (1 << 30));
        // no meminfo -> fall back to the GPU pool
        let nomem = UmaBudget {
            sys_total: 0,
            sys_available: 0,
            ..b.clone()
        };
        assert_eq!(nomem.usable(), nomem.total());
        assert_eq!(nomem.physical_total(), nomem.total());
    }

    #[test]
    fn meminfo_parses_kb_to_bytes() {
        let dir = std::env::temp_dir().join(format!("llmtune-meminfo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("meminfo");
        std::fs::write(
            &p,
            "MemTotal:       16289140 kB\nMemFree:  1000 kB\nMemAvailable:   13980000 kB\n",
        )
        .unwrap();
        let (total, avail) = read_meminfo_at(&p).unwrap();
        assert_eq!(total, 16_289_140 * 1024);
        assert_eq!(avail, 13_980_000 * 1024);
        // One malformed (colon-less) line must not abort the parse.
        std::fs::write(
            &p,
            "MemTotal:       16289140 kB\ngarbage line without a colon\nMemAvailable:   13980000 kB\n",
        )
        .unwrap();
        let (total, avail) = read_meminfo_at(&p).unwrap();
        assert_eq!(total, 16_289_140 * 1024);
        assert_eq!(avail, 13_980_000 * 1024);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ctx_kv_parsed_from_flags() {
        let f = "-c 32768 --flash-attn on -ngl 99 -ctk q4_0 -ctv q4_0";
        let (ctx, kv) = ctx_kv_from_flags(f);
        assert_eq!(ctx, Some(32768));
        assert_eq!(kv.as_deref(), Some("q4_0"));
        // absent -> None (caller defaults)
        let (ctx2, kv2) = ctx_kv_from_flags("--flash-attn on -ngl 99");
        assert_eq!(ctx2, None);
        assert_eq!(kv2, None);
    }

    #[test]
    fn estimate_without_dims_is_weights_only() {
        let e = estimate(9 * (1 << 30), None, 32768, "q4_0", None);
        assert_eq!(e.kv_bytes, 0);
        assert_eq!(e.working_bytes, 9 * (1 << 30));
        assert!(e.headroom_bytes.is_none());
    }

    #[test]
    fn read_uma_from_fixture_dir() {
        let dir = std::env::temp_dir().join(format!("llmtune-uma-{}", std::process::id()));
        let dev = dir.join("device");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::write(dev.join("mem_info_vram_total"), "8589934592\n").unwrap();
        std::fs::write(dev.join("mem_info_gtt_total"), "5800000000").unwrap();
        let b = read_uma_at(&dev).unwrap();
        assert_eq!(b.vram_total, 8_589_934_592);
        assert_eq!(b.gtt_total, 5_800_000_000);
        assert_eq!(b.total(), 8_589_934_592 + 5_800_000_000);
        // missing vram_total -> None
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(read_uma_at(&empty).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
