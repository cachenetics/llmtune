// SPDX-License-Identifier: GPL-2.0-only
//! Live GPU/memory telemetry from the amdgpu `gpu_metrics` sysfs blob.
//!
//! Ported from memtune. The BC-250 exposes an APU metrics table
//! (`gpu_metrics_v2_2`); we read the few fields reliable on this silicon: the GFX
//! core clock (boosts under load), the UMC/memory clock (locked ~450 MHz here),
//! and the GFX die temperature. Safe sysfs read, no SMN/ioctl.
//!
//! NOTE: reading this while a GPU compute job is in flight can race on the BC-250,
//! so we sample around a bench run, not in its hot path.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Serialized as the `node gpu --json` wire shape, so the SSH transport parses
/// exactly what the node-side CLI emits (one wire contract, no drift).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Telemetry {
    pub gfxclk_mhz: u16,
    pub uclk_mhz: u16,
    pub temp_c: f64,
    /// Socket power in watts (`average_socket_power`). None when the firmware
    /// leaves the field unpopulated (0 / 0xffff), and when parsing JSON from a
    /// node running an older llmtune (`serde(default)` keeps the wire
    /// back-compatible).
    #[serde(default)]
    pub power_w: Option<f64>,
    /// Total UMA pool in MiB (`/proc/meminfo` MemTotal). The BC-250 is a
    /// unified-memory board: system RAM and GPU GTT share one 16 GiB pool, so
    /// whole-system memory IS the GPU memory ceiling. None when /proc/meminfo
    /// is unreadable, and when parsing JSON from a node running an older
    /// llmtune (`serde(default)` keeps the wire back-compatible).
    #[serde(default)]
    pub mem_total_mib: Option<u32>,
    /// Used UMA in MiB: MemTotal - MemAvailable. Counts amdgpu TTM/GTT
    /// allocations (they reduce MemAvailable), which is exactly the
    /// llama-server-OOM pressure signal. Same None semantics as `mem_total_mib`.
    #[serde(default)]
    pub mem_used_mib: Option<u32>,
}

/// Read current telemetry, or None if the metrics blob isn't readable (e.g. not
/// a BC-250 / no amdgpu). UMA memory rides along when /proc/meminfo is
/// readable; a failed memory read never drops the GPU reading.
pub fn read() -> Option<Telemetry> {
    let mut t = parse(&fs::read(gpu_metrics_path()?).ok()?)?;
    if let Some((used, total)) = fs::read_to_string("/proc/meminfo")
        .ok()
        .as_deref()
        .and_then(parse_meminfo)
    {
        t.mem_used_mib = Some(used);
        t.mem_total_mib = Some(total);
    }
    Some(t)
}

/// Parse `/proc/meminfo` text -> (used MiB, total MiB). Pure (no IO) so it is
/// unit-testable against a fixture. used = MemTotal - MemAvailable: on the
/// UMA board this captures GTT allocations too, the actual OOM headroom.
fn parse_meminfo(text: &str) -> Option<(u32, u32)> {
    let kib = |key: &str| -> Option<u64> {
        text.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.trim() != key {
                return None;
            }
            v.split_whitespace().next()?.parse().ok()
        })
    };
    let total = kib("MemTotal")?;
    let avail = kib("MemAvailable")?;
    let used = total.saturating_sub(avail);
    Some(((used / 1024) as u32, (total / 1024) as u32))
}

/// Parse a `gpu_metrics` v2_x blob. Pure (no IO) so the byte-offset math is
/// unit-testable against a captured BC-250 blob.
fn parse(b: &[u8]) -> Option<Telemetry> {
    if b.len() < 82 {
        return None;
    }
    let u16at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let (fmt, content) = (b[2], b[3]);
    // temperature_gfx is at offset 4 in every v2_x; hundredths of a degree.
    let temp_c = u16at(4) as f64 / 100.0;
    // current_gfxclk / current_uclk land at 76 / 80 in v2_2.
    let (gfxclk_mhz, uclk_mhz) = if fmt == 2 && content >= 2 {
        (u16at(76), u16at(80))
    } else {
        (0, 0)
    };
    // average_socket_power sits at offset 40 in every v2_x. The kernel header
    // says watts, but the BC-250 (cyan skillfish) firmware writes MILLIWATTS:
    // verified against a live blob (0x93ad = 37805 at offset 40) whose hwmon
    // power1_average read 38.1 W at the same instant. 0/0xffff = unpopulated.
    let power_w = if fmt == 2 {
        match u16at(40) {
            0 | 0xffff => None,
            mw => Some(mw as f64 / 1000.0),
        }
    } else {
        None
    };
    Some(Telemetry {
        gfxclk_mhz,
        uclk_mhz,
        temp_c,
        power_w,
        // The blob is GPU-only; the UMA figures ride along in read().
        mem_total_mib: None,
        mem_used_mib: None,
    })
}

fn gpu_metrics_path() -> Option<PathBuf> {
    for entry in fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let dev = entry.path().join("device");
        let f = dev.join("gpu_metrics");
        if f.exists() && dev.join("pp_dpm_sclk").exists() {
            return Some(f);
        }
    }
    None
}

/// Accumulates telemetry samples over a bench run.
#[derive(Default)]
pub struct TelemetryAccum {
    n: u32,
    gfx_sum: u64,
    gfx_min: u16,
    gfx_max: u16,
    uclk: u16,
    temp_peak: f64,
    /// Socket-power accumulation, counted separately from `n`: the firmware
    /// leaves the field unpopulated on some reads, and a missing reading must
    /// not drag the average toward zero.
    pow_sum: f64,
    pow_n: u32,
}

impl TelemetryAccum {
    pub fn observe(&mut self, t: Telemetry) {
        if self.n == 0 {
            self.gfx_min = t.gfxclk_mhz;
            self.gfx_max = t.gfxclk_mhz;
        } else {
            self.gfx_min = self.gfx_min.min(t.gfxclk_mhz);
            self.gfx_max = self.gfx_max.max(t.gfxclk_mhz);
        }
        self.gfx_sum += t.gfxclk_mhz as u64;
        self.uclk = t.uclk_mhz.max(self.uclk);
        self.temp_peak = self.temp_peak.max(t.temp_c);
        if let Some(w) = t.power_w {
            if w.is_finite() {
                self.pow_sum += w;
                self.pow_n += 1;
            }
        }
        self.n += 1;
    }

    pub fn finish(&self) -> TelemetrySummary {
        let gfx_avg = if self.n > 0 {
            (self.gfx_sum / self.n as u64) as u16
        } else {
            0
        };
        let power_avg_w = if self.pow_n > 0 {
            Some(((self.pow_sum / self.pow_n as f64) * 10.0).round() / 10.0)
        } else {
            None
        };
        TelemetrySummary {
            samples: self.n,
            gfxclk_min: self.gfx_min,
            gfxclk_max: self.gfx_max,
            gfxclk_avg: gfx_avg,
            uclk_mhz: self.uclk,
            temp_peak_c: (self.temp_peak * 10.0).round() / 10.0,
            power_avg_w,
        }
    }
}

/// Aggregated telemetry over a bench run (stored in history).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub samples: u32,
    pub gfxclk_min: u16,
    pub gfxclk_max: u16,
    pub gfxclk_avg: u16,
    pub uclk_mhz: u16,
    pub temp_peak_c: f64,
    /// Mean socket watts over the run's populated power readings. None when no
    /// reading landed (older nodes / old history records: `serde(default)`
    /// keeps the wire and stored history back-compatible).
    #[serde(default)]
    pub power_avg_w: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first 120 bytes of a REAL BC-250 `gpu_metrics` blob (captured
    /// 2026-07-05 on a reference node, idle at the deep-idle tier), padded to the header's
    /// declared 128-byte structure_size. Ground truth at capture time: hwmon
    /// power1_average 38.1 W, gfx 350 MHz, uclk 450 MHz, temp 49.0 C.
    fn bc250_blob() -> Vec<u8> {
        let mut b: Vec<u8> = vec![
            0x80, 0x00, 0x02, 0x02, 0x24, 0x13, 0x43, 0x12, 0x94, 0x11, 0x11, 0x12, 0x7b, 0x11,
            0x2a, 0x12, 0x1e, 0x14, 0x7b, 0x11, 0xff, 0xff, 0xff, 0xff, 0xdf, 0x11, 0x5c, 0x12,
            0xff, 0xff, 0xff, 0xff, 0x91, 0x5d, 0x40, 0x3d, 0x67, 0x57, 0x00, 0x00, 0xad, 0x93,
            0xff, 0xff, 0xfd, 0x0d, 0x36, 0x23, 0xaa, 0x00, 0x33, 0x00, 0x32, 0x00, 0x90, 0x00,
            0x8c, 0x08, 0xd4, 0x00, 0xff, 0xff, 0xff, 0xff, 0x5e, 0x01, 0xe6, 0x04, 0xc2, 0x01,
            0xc2, 0x01, 0x00, 0x00, 0x57, 0x04, 0x5e, 0x01, 0xe6, 0x04, 0xc2, 0x01, 0xc2, 0x01,
            0x00, 0x00, 0x57, 0x04, 0xac, 0x0d, 0x13, 0x06, 0x13, 0x06, 0xac, 0x0d, 0xac, 0x0d,
            0xac, 0x0d, 0xff, 0xff, 0xff, 0xff, 0xac, 0x0d, 0xac, 0x0d, 0x00, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        b.resize(128, 0);
        b
    }

    #[test]
    fn parse_extracts_clocks_temp_and_power_from_a_real_blob() {
        let t = parse(&bc250_blob()).expect("real blob must parse");
        assert_eq!(t.gfxclk_mhz, 350);
        assert_eq!(t.uclk_mhz, 450);
        assert_eq!(t.temp_c, 49.0);
        // offset 40 = 0x93ad = 37805 mW; the firmware writes milliwatts (hwmon
        // read 38.1 W at capture), so the parsed value is watts.
        let w = t.power_w.expect("power populated on the BC-250");
        assert!((w - 37.805).abs() < 1e-9, "got {w}");
    }

    #[test]
    fn parse_treats_unpopulated_power_as_none() {
        // 0xffff (firmware n/a marker) and 0 both mean "no reading".
        for filler in [[0xff, 0xff], [0x00, 0x00]] {
            let mut b = bc250_blob();
            b[40] = filler[0];
            b[41] = filler[1];
            let t = parse(&b).unwrap();
            assert!(t.power_w.is_none());
        }
        // a short/garbage blob parses to None, not a panic
        assert!(parse(&[0u8; 10]).is_none());
        assert!(parse(&[]).is_none());
    }

    #[test]
    fn wire_shape_is_backward_compatible() {
        // JSON from a node running an older llmtune (no power_w field) must
        // still parse; power collapses to None.
        let t: Telemetry =
            serde_json::from_str(r#"{"gfxclk_mhz":1500,"uclk_mhz":450,"temp_c":61.5}"#).unwrap();
        assert!(t.power_w.is_none());
        // ...and so must JSON predating the UMA memory fields.
        assert!(t.mem_total_mib.is_none());
        assert!(t.mem_used_mib.is_none());
        // and the new shape round-trips
        let t2 = Telemetry {
            gfxclk_mhz: 2230,
            uclk_mhz: 450,
            temp_c: 61.0,
            power_w: Some(38.1),
            mem_total_mib: Some(15680),
            mem_used_mib: Some(10340),
        };
        let back: Telemetry = serde_json::from_str(&serde_json::to_string(&t2).unwrap()).unwrap();
        assert_eq!(back.power_w, Some(38.1));
        assert_eq!(back.mem_total_mib, Some(15680));
        assert_eq!(back.mem_used_mib, Some(10340));
    }

    #[test]
    fn parse_meminfo_computes_used_from_available() {
        // Shape of a real BC-250 /proc/meminfo (kB values). MemTotal 15.3 GiB,
        // MemAvailable 5.2 GiB -> used = total - available.
        let fixture = "MemTotal:       16087040 kB\n\
                       MemFree:         1204480 kB\n\
                       MemAvailable:    5455872 kB\n\
                       Buffers:          102400 kB\n\
                       Cached:          4194304 kB\n";
        let (used, total) = parse_meminfo(fixture).expect("fixture must parse");
        assert_eq!(total, 16087040 / 1024); // 15710 MiB
        assert_eq!(used, (16087040 - 5455872) / 1024); // 10382 MiB
    }

    #[test]
    fn parse_meminfo_missing_or_garbage_is_none() {
        // No MemAvailable (ancient kernel) -> None, not a bogus figure.
        assert!(parse_meminfo("MemTotal: 16087040 kB\nMemFree: 1 kB\n").is_none());
        assert!(parse_meminfo("MemAvailable: 5455872 kB\n").is_none());
        assert!(parse_meminfo("").is_none());
        assert!(parse_meminfo("MemTotal: garbage kB\nMemAvailable: 1 kB\n").is_none());
        // MemAvailable > MemTotal (can't happen, but don't underflow)
        let (used, _) = parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 2048 kB\n").unwrap();
        assert_eq!(used, 0);
    }

    #[test]
    fn accum_aggregates() {
        let mut a = TelemetryAccum::default();
        a.observe(Telemetry {
            gfxclk_mhz: 1000,
            uclk_mhz: 450,
            temp_c: 60.0,
            power_w: None,
            ..Default::default()
        });
        a.observe(Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: 72.5,
            power_w: Some(38.1),
            ..Default::default()
        });
        a.observe(Telemetry {
            gfxclk_mhz: 2000,
            uclk_mhz: 450,
            temp_c: 70.0,
            power_w: Some(55.0),
            ..Default::default()
        });
        let s = a.finish();
        assert_eq!(s.samples, 3);
        assert_eq!(s.gfxclk_min, 1000);
        assert_eq!(s.gfxclk_max, 2000);
        assert_eq!(s.gfxclk_avg, 1500);
        assert_eq!(s.uclk_mhz, 450);
        assert_eq!(s.temp_peak_c, 72.5);
        // avg power over the POPULATED readings only: (38.1 + 55.0) / 2, the
        // None sample must not drag it toward zero.
        assert_eq!(s.power_avg_w, Some(46.6));
    }

    #[test]
    fn empty_accum_is_zero() {
        let s = TelemetryAccum::default().finish();
        assert_eq!(s.samples, 0);
        assert_eq!(s.gfxclk_avg, 0);
        assert_eq!(s.power_avg_w, None);
    }

    #[test]
    fn accum_power_none_when_never_populated() {
        let mut a = TelemetryAccum::default();
        a.observe(Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: 60.0,
            power_w: None,
            ..Default::default()
        });
        assert_eq!(a.finish().power_avg_w, None);
        // non-finite readings are dropped, not averaged
        a.observe(Telemetry {
            gfxclk_mhz: 1500,
            uclk_mhz: 450,
            temp_c: 60.0,
            power_w: Some(f64::NAN),
            ..Default::default()
        });
        assert_eq!(a.finish().power_avg_w, None);
    }

    #[test]
    fn summary_wire_shape_is_backward_compatible() {
        // A summary from an old history record / old remote llmtune (no
        // power_avg_w) must still parse; avg power collapses to None.
        let s: TelemetrySummary = serde_json::from_str(
            r#"{"samples":9,"gfxclk_min":1000,"gfxclk_max":2230,"gfxclk_avg":2100,
                "uclk_mhz":450,"temp_peak_c":71.5}"#,
        )
        .unwrap();
        assert_eq!(s.power_avg_w, None);
        // and the new shape round-trips
        let s2 = TelemetrySummary {
            power_avg_w: Some(38.1),
            ..s
        };
        let back: TelemetrySummary =
            serde_json::from_str(&serde_json::to_string(&s2).unwrap()).unwrap();
        assert_eq!(back.power_avg_w, Some(38.1));
    }
}
