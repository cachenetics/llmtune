// SPDX-License-Identifier: GPL-2.0-only
//! Durable, crash-safe benchmark history. One JSON file per node, newest-first.
//! The write pattern (temp -> fsync -> atomic rename -> fsync dir) is ported from
//! memtune: a power loss mid-write yields either the old or the new file, never a
//! corrupt one.

use crate::bench::PerfBench;
use crate::paths::{state_dir, write_durable};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cap on records kept per node - history is append-only and consulted on every
/// leaderboard/compare, so it can't grow without bound. Newest are kept.
const MAX_RECORDS: usize = 1000;

/// A best-effort BLOCKING advisory lock on a sibling `.lock` file, held for the
/// duration of an append so a read-modify-write can't lose a record when two
/// writers (a CLI bench and the proxy's swap-on-demand) race. Released when the
/// fd closes (drop or process exit). If locking is unavailable it's a no-op.
struct AppendLock(#[allow(dead_code)] Option<File>);

impl AppendLock {
    fn acquire(path: &Path) -> AppendLock {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = path.with_extension("lock");
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .ok();
        if let Some(f) = &f {
            // Wait for a concurrent append to finish, but bounded: a peer that
            // crashed or wedged mid-hold must not block this append forever. On
            // timeout we proceed best-effort (same as when the lock is a no-op).
            if !crate::lock::flock_ex_timeout(f.as_raw_fd(), crate::lock::RMW_LOCK_TIMEOUT) {
                eprintln!("llmtune: history append lock timed out; proceeding unlocked");
            }
        }
        AppendLock(f)
    }
}

/// One benchmark run and the model/context it measured.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub ts: u64,
    pub node: String,
    pub model: String,
    pub arch: String,
    #[serde(default)]
    pub quant: Option<String>,
    pub ctx: u32,
    pub profile: String,
    pub perf: PerfBench,
    #[serde(default)]
    pub notes: String,
    /// The llama.cpp build (version slug) this run used, if a managed build served
    /// it - lets the comparison/leaderboard attribute throughput to a build.
    #[serde(default)]
    pub build: Option<String>,
    /// The cluster (llama.cpp RPC pool) serving when this run was measured -
    /// None for a single-node run. Distinguishes pooled from single-node
    /// throughput in history/leaderboards (SPEC 5.7). Back-compatible: old
    /// records without the field load as None/empty.
    #[serde(default)]
    pub cluster: Option<String>,
    /// The pooled worker nodes at bench time (empty for a single-node run).
    #[serde(default)]
    pub cluster_members: Vec<String>,
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A node's history file.
pub struct Store {
    path: PathBuf,
}

impl Store {
    pub fn for_node(node: &str) -> Store {
        Store {
            path: state_dir().join(node).join("history.json"),
        }
    }

    /// Explicit path (tests / non-default layouts).
    #[cfg(test)]
    pub fn at(path: PathBuf) -> Store {
        Store { path }
    }

    pub fn load(&self) -> Vec<Record> {
        fs::read_to_string(&self.path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// True if a history file exists on disk but does not parse as records.
    fn file_is_corrupt(&self) -> bool {
        match fs::read_to_string(&self.path) {
            Ok(s) => serde_json::from_str::<Vec<Record>>(&s).is_err(),
            Err(_) => false, // missing file is not corrupt - just empty history
        }
    }

    /// Prepend a record (newest first) and persist durably. If the existing file
    /// is present but unparseable, it is moved aside to `*.corrupt` first rather
    /// than silently overwritten, so the prior data stays recoverable.
    pub fn append(&self, rec: Record) -> Result<()> {
        // Serialize concurrent appends so the load->insert->write RMW can't lose a
        // record (held until this function returns).
        let _lk = AppendLock::acquire(&self.path);
        if self.file_is_corrupt() {
            // Unique suffix so a second corruption doesn't clobber the first aside.
            let aside = self.path.with_extension(format!("json.corrupt.{}", rec.ts));
            let _ = fs::rename(&self.path, &aside);
        }
        let mut v = self.load();
        v.insert(0, rec);
        v.truncate(MAX_RECORDS);
        write_durable(&self.path, serde_json::to_string_pretty(&v)?.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(model: &str, ts: u64) -> Record {
        Record {
            ts,
            node: "localhost".into(),
            model: model.into(),
            arch: "llama".into(),
            quant: Some("Q4_K_M".into()),
            ctx: 32768,
            profile: "_default".into(),
            perf: PerfBench {
                gen_tok_s: 42.0,
                ..Default::default()
            },
            notes: String::new(),
            build: None,
            cluster: None,
            cluster_members: vec![],
        }
    }

    #[test]
    fn old_records_without_cluster_fields_still_load() {
        // Back-compat: a pre-cluster-tagging history file must deserialize, with
        // the new fields defaulting to None/empty.
        let old = r#"[{
            "ts": 100, "node": "localhost", "model": "a.gguf", "arch": "llama",
            "ctx": 32768, "profile": "_default",
            "perf": { "prompt_tok_s": 100.0, "gen_tok_s": 42.0, "ttft_ms": 200.0,
                      "total_ms": 3000.0, "n_prompt": 512, "n_gen": 128 }
        }]"#;
        let v: Vec<Record> = serde_json::from_str(old).expect("old records must load");
        assert_eq!(v.len(), 1);
        assert!(v[0].cluster.is_none());
        assert!(v[0].cluster_members.is_empty());
    }

    #[test]
    fn cluster_tagged_record_roundtrips() {
        let mut r = rec("big-70B.gguf", 100);
        r.cluster = Some("big".into());
        r.cluster_members = vec!["bc250-2".into(), "bc250-3".into()];
        let s = serde_json::to_string(&r).unwrap();
        let back: Record = serde_json::from_str(&s).unwrap();
        assert_eq!(back.cluster.as_deref(), Some("big"));
        assert_eq!(back.cluster_members, vec!["bc250-2", "bc250-3"]);
    }

    #[test]
    fn append_load_roundtrip_newest_first() {
        let dir = std::env::temp_dir().join(format!("llmtune-hist-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::at(dir.join("history.json"));
        store.append(rec("a.gguf", 100)).unwrap();
        store.append(rec("b.gguf", 200)).unwrap();
        let v = store.load();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].model, "b.gguf"); // newest first
        assert_eq!(v[1].model, "a.gguf");
        assert!((v[0].perf.gen_tok_s - 42.0).abs() < 1e-9);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_loads_empty() {
        let store = Store::at(PathBuf::from("/nonexistent/llmtune/history.json"));
        assert!(store.load().is_empty());
    }

    #[test]
    fn corrupt_file_is_preserved_not_wiped() {
        let dir = std::env::temp_dir().join(format!("llmtune-hist-corrupt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");
        fs::write(&path, b"{ this is not valid history json ][").unwrap();
        let store = Store::at(path.clone());
        // append must succeed, move the bad file aside, and start a fresh history
        store.append(rec("a.gguf", 100)).unwrap();
        let v = store.load();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].model, "a.gguf");
        assert!(
            path.with_extension("json.corrupt.100").exists(),
            "corrupt file must be moved aside (uniquely named), not destroyed"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_is_capped_newest_kept() {
        let dir = std::env::temp_dir().join(format!("llmtune-hist-cap-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::at(dir.join("history.json"));
        for ts in 0..(MAX_RECORDS as u64 + 25) {
            store.append(rec("m.gguf", ts)).unwrap();
        }
        let v = store.load();
        assert_eq!(v.len(), MAX_RECORDS, "history is capped");
        // newest-first: ts of the newest append is at the front, oldest dropped.
        assert_eq!(v[0].ts, MAX_RECORDS as u64 + 24);
        assert_eq!(v[MAX_RECORDS - 1].ts, 25);
        let _ = fs::remove_dir_all(&dir);
    }
}
