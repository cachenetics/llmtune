// SPDX-License-Identifier: GPL-2.0-only
//! Throughput benchmark via llama.cpp's dedicated `llama-bench` tool.
//!
//! The old path drove a running llama-server's `/completion` and read its
//! `timings` - unreliable on this stack: speculative decode (MTP) mis-attributes
//! generation time, cache hits report ~0 ms, and the numbers were routinely
//! fantasy (1,000,000 tok/s). `llama-bench` is the ground truth: it loads the
//! model directly, runs separate prompt-processing (pp) and text-generation (tg)
//! passes with warmup + repetitions, and reports clean averages. Because every
//! BC-250 model fills the 16 GiB UMA, the caller (nodeops) stops the server to
//! free the GPU, runs this, then restarts it.

use crate::telemetry::{self, TelemetryAccum, TelemetrySummary};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Hard wall-clock cap on a single llama-bench run. A bench that exceeds this on
/// this hardware is wedged (GPU hang / stuck load); killing it frees the node
/// instead of blocking the caller (which has already stopped llama-server)
/// forever.
const BENCH_TIMEOUT: Duration = Duration::from_secs(600);

/// What to benchmark: prompt-processing depth, generation length, repetitions.
#[derive(Debug, Clone)]
pub struct BenchSpec {
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub repeats: u32,
}

impl Default for BenchSpec {
    fn default() -> Self {
        BenchSpec {
            prompt_tokens: 512,
            gen_tokens: 128,
            // llama-bench does its own warmup, and the per-run stddev on this
            // stack is ~0, so 2 reps is a stable average - 3 just spent ~13s more
            // per bench for no meaningful accuracy gain.
            repeats: 2,
        }
    }
}

/// Aggregated perf result (stored in history).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerfBench {
    pub prompt_tok_s: f64,
    pub gen_tok_s: f64,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub n_prompt: u32,
    pub n_gen: u32,
    #[serde(default)]
    pub telemetry: TelemetrySummary,
}

/// Live progress emitted around a run (so a UI can show a spinner + final tok/s).
/// `llama-bench` reports only at the end, so this fires once at start and once at
/// completion rather than ticking per token.
#[derive(Debug, Clone, Copy)]
pub struct BenchProgress {
    pub completed: u32,
    pub total: u32,
    pub gen_tok_s: f64,
}

/// A resolved `llama-bench` invocation: the binary, its library path + env (the
/// per-arch profile's), the model file, and the extra flags mirrored from the
/// serving config (so the bench reflects how the model actually runs).
#[derive(Debug, Clone)]
pub struct LlamaBench {
    pub bin: PathBuf,
    pub ld_path: Option<String>,
    pub env: BTreeMap<String, String>,
    pub model: PathBuf,
    pub extra: Vec<String>,
}

/// No real BC-250 model generates faster than this; anything above is a parse
/// artifact, not a measurement. Clamp to 0 rather than report a fantasy number.
const MAX_PLAUSIBLE_TOK_S: f64 = 50_000.0;

/// A real text-generation pass benches at least this many tokens (the default is
/// 128). Records with fewer are not throughput measurements: the pre-`llama-bench`
/// `/completion` path returned ~1 generated token on a cache hit or MTP
/// mis-attribution, and its tok/s was noise (routinely 475/1e6 t/s "fantasy").
const MIN_BENCH_GEN: u32 = 8;

impl PerfBench {
    /// Whether this record is a trustworthy throughput measurement. Guards the
    /// leaderboard / best-per-model / "last t/s" pickers - all of which take a
    /// `max`/`first` over history - against the fantasy records left by the old
    /// `/completion` path, so one bad row can't win a comparison. A record is
    /// implausible if it ran too few generated tokens, reports a non-finite or
    /// absurd rate, or claims decode faster than prompt-processing (physically
    /// impossible on the same model - pp is always >= tg).
    pub fn is_plausible(&self) -> bool {
        self.n_gen >= MIN_BENCH_GEN
            && self.gen_tok_s.is_finite()
            && self.gen_tok_s > 0.0
            && self.gen_tok_s <= MAX_PLAUSIBLE_TOK_S
            // Guard tg <= pp only when pp was measured, so a pp-less tg-only
            // record isn't falsely rejected.
            && (self.prompt_tok_s <= 0.0 || self.gen_tok_s <= self.prompt_tok_s)
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Last `n` lines of `s` (for surfacing a failing tool's tail without flooding).
fn tail(s: &str, n: usize) -> String {
    let mut lines: Vec<&str> = s.lines().rev().take(n).collect();
    lines.reverse();
    lines.join("\n")
}

/// Derive the `llama-bench` flags that matter for a faithful measurement from the
/// serving flag string. We deliberately carry only the knobs that change
/// throughput on this stack (GPU offload, flash-attn, KV cache dtype, threads)
/// and drop server-only flags (`--host`/`--port`/`-c`/`--spec-type`/…) that
/// `llama-bench` doesn't accept. GPU offload defaults to full (`-ngl 99`).
pub fn bench_flags_from(serving: &str) -> Vec<String> {
    let toks: Vec<&str> = serving.split_whitespace().collect();
    let mut ngl: Option<String> = None;
    let mut fa = false;
    let mut ctk: Option<String> = None;
    let mut ctv: Option<String> = None;
    let mut threads: Option<String> = None;
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        let next = toks.get(i + 1).copied();
        match t {
            "-ngl" | "--n-gpu-layers" | "--gpu-layers" => {
                if let Some(v) = next {
                    ngl = Some(v.into());
                    i += 1;
                }
            }
            "-ctk" | "--cache-type-k" => {
                if let Some(v) = next {
                    ctk = Some(v.into());
                    i += 1;
                }
            }
            "-ctv" | "--cache-type-v" => {
                if let Some(v) = next {
                    ctv = Some(v.into());
                    i += 1;
                }
            }
            "-t" | "--threads" => {
                if let Some(v) = next {
                    threads = Some(v.into());
                    i += 1;
                }
            }
            "-fa" | "--flash-attn" => match next {
                // -fa may be bare (means on) or take on/off/1/0.
                Some("on" | "1" | "true") => {
                    fa = true;
                    i += 1;
                }
                Some("off" | "0" | "false") => {
                    fa = false;
                    i += 1;
                }
                _ => fa = true,
            },
            _ => {}
        }
        i += 1;
    }
    let mut out = vec!["-ngl".to_string(), ngl.unwrap_or_else(|| "99".to_string())];
    if fa {
        out.push("-fa".into());
        out.push("1".into());
    }
    if let Some(k) = ctk {
        out.push("-ctk".into());
        out.push(k);
    }
    if let Some(v) = ctv {
        out.push("-ctv".into());
        out.push(v);
    }
    if let Some(t) = threads {
        out.push("-t".into());
        out.push(t);
    }
    out
}

/// Parse `llama-bench -o json` output: an array with one object per test. The
/// prompt-processing row has `n_gen == 0`; the text-generation row has
/// `n_prompt == 0`. `avg_ts` is tokens/sec, `avg_ns` the total run time.
fn parse_bench_json(stdout: &str) -> Result<PerfBench> {
    let v: Value = serde_json::from_str(stdout.trim()).with_context(|| {
        format!(
            "llama-bench did not emit valid JSON - tail:\n{}",
            tail(stdout, 12)
        )
    })?;
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow!("llama-bench JSON was not an array"))?;

    // (n_tokens, avg_ts, avg_ns)
    let mut pp: Option<(u64, f64, f64)> = None;
    let mut tg: Option<(u64, f64, f64)> = None;
    for o in arr {
        let np = o.get("n_prompt").and_then(|x| x.as_u64()).unwrap_or(0);
        let ng = o.get("n_gen").and_then(|x| x.as_u64()).unwrap_or(0);
        let ts = o.get("avg_ts").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let ns = o.get("avg_ns").and_then(|x| x.as_f64()).unwrap_or(0.0);
        if ng == 0 && np > 0 {
            pp = Some((np, ts, ns));
        } else if np == 0 && ng > 0 {
            tg = Some((ng, ts, ns));
        }
        // combined pg rows (both > 0) are ignored - we request separate tests.
    }
    if pp.is_none() && tg.is_none() {
        bail!("llama-bench JSON had no pp/tg test rows");
    }
    let clamp = |x: f64| {
        if x.is_finite() && (0.0..=MAX_PLAUSIBLE_TOK_S).contains(&x) {
            x
        } else {
            0.0
        }
    };
    let (n_prompt, pp_ts, pp_ns) = pp.unwrap_or((0, 0.0, 0.0));
    let (n_gen, tg_ts, tg_ns) = tg.unwrap_or((0, 0.0, 0.0));
    Ok(PerfBench {
        prompt_tok_s: round1(clamp(pp_ts)),
        gen_tok_s: round1(clamp(tg_ts)),
        ttft_ms: round1(pp_ns / 1e6),
        total_ms: round1((pp_ns + tg_ns) / 1e6),
        n_prompt: n_prompt as u32,
        n_gen: n_gen as u32,
        telemetry: TelemetrySummary::default(),
    })
}

/// Run `llama-bench` and return the parsed result, sampling GPU telemetry across
/// the run on a background thread. The caller must have freed the GPU (stopped
/// the server) first. Emits start/finish progress on `tx` (best-effort).
pub fn run_llama_bench(
    cfg: &LlamaBench,
    spec: &BenchSpec,
    tx: &Sender<BenchProgress>,
) -> Result<PerfBench> {
    if !cfg.bin.exists() {
        bail!(
            "llama-bench not found at {} - rebuild the engine (`llmtune build install`)",
            cfg.bin.display()
        );
    }
    let _ = tx.send(BenchProgress {
        completed: 0,
        total: 1,
        gen_tok_s: 0.0,
    });

    // Sample GPU clocks/temp every 500ms for the duration (llama-bench blocks, so
    // we can't sample inline; a peak/mean over the run is the useful signal).
    let stop = Arc::new(AtomicBool::new(false));
    let accum = Arc::new(Mutex::new(TelemetryAccum::default()));
    let sampler = {
        let stop = stop.clone();
        let accum = accum.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(t) = telemetry::read() {
                    if let Ok(mut a) = accum.lock() {
                        a.observe(t);
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        })
    };

    // NB: the bench does NOT control the GPU clock - aputune is the controller.
    // It runs at whatever clock aputune currently provides (its governor ramps to
    // high under this load; or the operator pins it with `arieltune apu gpu pin <mhz>`
    // for a rock-steady number). The observed gfxclk range/avg is recorded in the
    // telemetry summary below, so a floating-clock run is visible in the result.
    let mut args: Vec<String> = vec![
        "-o".into(),
        "json".into(),
        "-p".into(),
        spec.prompt_tokens.to_string(),
        "-n".into(),
        spec.gen_tokens.to_string(),
        "-r".into(),
        spec.repeats.max(1).to_string(),
        "-m".into(),
        cfg.model.display().to_string(),
    ];
    args.extend(cfg.extra.iter().cloned());

    let mut cmd = Command::new(&cfg.bin);
    if let Some(ld) = &cfg.ld_path {
        cmd.env("LD_LIBRARY_PATH", ld);
    }
    for (k, v) in &cfg.env {
        cmd.env(k, v);
    }
    // Spawn + wait with a deadline so a wedged bench can't block forever. On any
    // early return the sampler thread must be stopped + joined (else it loops
    // reading sysfs for the life of the process).
    let stop_sampler = || {
        stop.store(true, Ordering::Relaxed);
    };
    let child = cmd
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let child = match child {
        Ok(c) => c,
        Err(e) => {
            stop_sampler();
            let _ = sampler.join();
            return Err(e).with_context(|| format!("running {}", cfg.bin.display()));
        }
    };
    let pid = child.id();
    let (otx, orx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = otx.send(child.wait_with_output());
    });
    let out = match orx.recv_timeout(BENCH_TIMEOUT) {
        Ok(res) => {
            stop_sampler();
            let _ = sampler.join();
            res.with_context(|| format!("running {}", cfg.bin.display()))?
        }
        Err(_) => {
            // Timed out - kill the child, stop the sampler, and fail.
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
            stop_sampler();
            let _ = sampler.join();
            bail!(
                "llama-bench exceeded {}s and was killed (GPU wedge or stuck load?)",
                BENCH_TIMEOUT.as_secs()
            );
        }
    };
    let telemetry = Arc::try_unwrap(accum)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .map(|a| a.finish())
        .unwrap_or_default();

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!(
            "llama-bench exited {} - tail:\n{}",
            out.status,
            tail(&err, 12)
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut perf = parse_bench_json(&stdout)?;
    perf.telemetry = telemetry;
    let _ = tx.send(BenchProgress {
        completed: 1,
        total: 1,
        gen_tok_s: perf.gen_tok_s,
    });
    Ok(perf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_two_row_pp_tg() {
        let json = r#"[
          {"n_prompt":512,"n_gen":0,"avg_ns":1200000000.0,"stddev_ns":0.0,"avg_ts":426.6,"stddev_ts":1.0},
          {"n_prompt":0,"n_gen":128,"avg_ns":2500000000.0,"stddev_ns":0.0,"avg_ts":51.2,"stddev_ts":0.5}
        ]"#;
        let p = parse_bench_json(json).unwrap();
        assert!((p.prompt_tok_s - 426.6).abs() < 0.1);
        assert!((p.gen_tok_s - 51.2).abs() < 0.1);
        assert_eq!(p.n_prompt, 512);
        assert_eq!(p.n_gen, 128);
        assert!((p.ttft_ms - 1200.0).abs() < 0.1);
        assert!((p.total_ms - 3700.0).abs() < 0.1); // 1200 + 2500
    }

    #[test]
    fn parse_clamps_fantasy_rate() {
        // A parse artifact (e.g. avg_ns ~0) yielding an absurd rate is clamped to 0
        // rather than reported as a real number - no more 1,000,000 tok/s.
        let json = r#"[{"n_prompt":0,"n_gen":128,"avg_ns":0.0,"avg_ts":1000000.0}]"#;
        let p = parse_bench_json(json).unwrap();
        assert_eq!(p.gen_tok_s, 0.0);
    }

    #[test]
    fn parse_integer_ns_ok() {
        // avg_ns may serialize as an integer; as_f64 must still read it.
        let json = r#"[{"n_prompt":0,"n_gen":64,"avg_ns":2000000000,"avg_ts":32.0}]"#;
        let p = parse_bench_json(json).unwrap();
        assert!((p.gen_tok_s - 32.0).abs() < 0.1);
        assert_eq!(p.n_gen, 64);
    }

    #[test]
    fn parse_rejects_non_array() {
        assert!(parse_bench_json(r#"{"oops":1}"#).is_err());
        assert!(parse_bench_json("not json").is_err());
        assert!(parse_bench_json("[]").is_err()); // no test rows
    }

    fn perf(n_gen: u32, pp: f64, tg: f64) -> PerfBench {
        PerfBench {
            prompt_tok_s: pp,
            gen_tok_s: tg,
            n_gen,
            n_prompt: 512,
            ..Default::default()
        }
    }

    #[test]
    fn plausible_accepts_a_real_bench() {
        // A clean llama-bench record (128 gen tokens, pp >= tg).
        assert!(perf(128, 141.8, 23.2).is_plausible());
    }

    #[test]
    fn plausible_rejects_completion_fantasies() {
        // The exact shapes rejected by the plausibility guard: n_gen == 1 with tg > pp,
        // and the 1e6 t/s parse artifact.
        assert!(!perf(1, 108.0, 475.6).is_plausible()); // n_gen too low AND tg > pp
        assert!(!perf(1, 82.1, 1_000_000.0).is_plausible());
        assert!(!perf(1, 241.4, 45.4).is_plausible()); // tg <= pp here, but n_gen == 1
    }

    #[test]
    fn plausible_rejects_tg_exceeding_pp() {
        // Decode can't beat prefill on the same model even at full gen length.
        assert!(!perf(128, 100.0, 200.0).is_plausible());
    }

    #[test]
    fn plausible_allows_pp_less_tg_only_record() {
        // A tg-only record (pp not measured) is not falsely rejected.
        assert!(perf(128, 0.0, 40.0).is_plausible());
    }

    #[test]
    fn flags_default_full_offload() {
        // A serving string with no -ngl still benches at full GPU offload.
        let f = bench_flags_from("-c 32768 --host 127.0.0.1 --port 8080");
        assert_eq!(f, vec!["-ngl", "99"]);
    }

    #[test]
    fn flags_carry_offload_fa_and_kv() {
        let f =
            bench_flags_from("-ngl 40 -fa on -ctk q8_0 -ctv q8_0 -c 32768 --spec-type draft-mtp");
        // -ngl mirrored, -fa on -> "1", kv dtypes carried, server-only flags dropped
        assert!(f.windows(2).any(|w| w == ["-ngl", "40"]));
        assert!(f.windows(2).any(|w| w == ["-fa", "1"]));
        assert!(f.windows(2).any(|w| w == ["-ctk", "q8_0"]));
        assert!(f.windows(2).any(|w| w == ["-ctv", "q8_0"]));
        assert!(!f.iter().any(|x| x == "--spec-type" || x == "-c"));
    }

    #[test]
    fn flags_bare_fa_means_on() {
        let f = bench_flags_from("-ngl 99 -fa -c 4096");
        assert!(f.windows(2).any(|w| w == ["-fa", "1"]));
    }

    #[test]
    fn flags_fa_off_is_dropped() {
        let f = bench_flags_from("-ngl 99 -fa off");
        assert!(!f.iter().any(|x| x == "-fa"));
    }
}
