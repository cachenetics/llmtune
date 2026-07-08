// SPDX-License-Identifier: GPL-2.0-only
//! Advisory locks so GPU-touching operations don't run at once.
//!
//! A single `flock(LOCK_EX|LOCK_NB)` per lock file gives real mutual exclusion:
//! the kernel releases it automatically when the holding process exits (even on a
//! crash), so there are no stale lock files and no TOCTOU window. `swap` and
//! `bench` share ONE lock (they both drive the one GPU, so they must be mutually
//! exclusive); `build` (a CPU compile) has its own, so it may run alongside.
//!
//! The lock dir is chosen so it is ALWAYS writable by the invoking user: root
//! uses `/run/llmtune`, a non-root user its `$XDG_RUNTIME_DIR/llmtune` (falling
//! back to a per-uid temp dir). If the dir genuinely can't be created, locking is
//! unavailable and the op proceeds best-effort (as before) rather than blocking.

use std::fs::{File, OpenOptions};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn euid() -> u32 {
    unsafe { libc::geteuid() }
}

/// Default ceiling for the best-effort blocking RMW/append locks (history append,
/// settings read-modify-write). Those critical sections are milliseconds, so real
/// contention clears at once; the ceiling only stops a peer that crashed or
/// wedged mid-hold from blocking the caller forever.
pub const RMW_LOCK_TIMEOUT: Duration = Duration::from_secs(5);

/// Take an advisory `flock(LOCK_EX)` on `fd` without ever blocking indefinitely:
/// poll `LOCK_NB` until acquired or `timeout` elapses. Returns `true` if the lock
/// was actually taken, `false` on timeout (the caller then proceeds best-effort,
/// exactly as these lock sites already do when the lock file can't be opened).
///
/// Polling `LOCK_NB` rather than a bare blocking `LOCK_EX` also rides over the
/// fork-window footgun: a concurrent subprocess spawn can momentarily inherit
/// this fd before `execve` fires `O_CLOEXEC`, which a single `LOCK_NB` would
/// mis-read as contention (see [`LockGuard::acquire`]).
pub fn flock_ex_timeout(fd: RawFd, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Directory for lock files - always writable by the current user.
fn lock_dir() -> PathBuf {
    // Honor the test/state override so isolated runs isolate their locks too.
    if let Ok(d) = std::env::var("LLMTUNE_STATE_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d).join("locks");
        }
    }
    if euid() == 0 {
        return PathBuf::from("/run/llmtune");
    }
    if let Ok(rt) = std::env::var("XDG_RUNTIME_DIR") {
        if !rt.is_empty() {
            return PathBuf::from(rt).join("llmtune");
        }
    }
    std::env::temp_dir().join(format!("llmtune-{}", euid()))
}

/// The GPU-serialization lock (shared by swap + bench).
fn gpu_lock() -> PathBuf {
    lock_dir().join("gpu.lock")
}

fn build_lock() -> PathBuf {
    lock_dir().join("build.lock")
}

/// Returned when another process already holds the lock.
#[derive(Debug)]
pub struct Busy;

/// A held advisory lock. The `flock` is released when the file descriptor
/// closes: on drop, or automatically if the process dies. The (empty) lock
/// file is left in place: deleting a flock'd path is racy (a fresh create+lock
/// wouldn't conflict with a waiter holding the old inode), so we never unlink it.
pub struct LockGuard {
    _file: Option<File>,
}

impl LockGuard {
    fn acquire(path: PathBuf) -> Result<LockGuard, Busy> {
        // If we can't even create the lock dir/file, locking is unavailable -
        // proceed best-effort (unlocked) rather than falsely reporting Busy.
        if let Some(parent) = path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return Ok(LockGuard { _file: None });
            }
        }
        let file = match OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)
        {
            Ok(f) => f,
            Err(_) => return Ok(LockGuard { _file: None }),
        };
        // Exclusive lock, non-blocking with a short bounded retry. A genuine
        // holder keeps the lock for its whole operation (seconds), so after the
        // retry window we still correctly report Busy for real contention. The
        // retry exists because EWOULDBLOCK can be *transient with no real
        // contention*: when another thread in this process spawns a subprocess
        // (fork+exec of ssh/systemctl/git/cmake), the child momentarily inherits
        // this fd during the window before execve fires O_CLOEXEC, so the flock
        // counts as held until the child execs. Without the retry, a swap/bench/
        // build could then spuriously fail as "busy" (and the test suite flakes,
        // since its parallel ssh-probing tests fork constantly). Riding out that
        // sub-millisecond window converts the false Busy into a real acquire.
        let deadline = Instant::now() + Duration::from_millis(500);
        loop {
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc == 0 {
                return Ok(LockGuard { _file: Some(file) });
            }
            if Instant::now() >= deadline {
                return Err(Busy);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// The GPU lock: swap and bench both take it, so they are mutually exclusive.
    pub fn gpu() -> Result<LockGuard, Busy> {
        Self::acquire(gpu_lock())
    }

    /// The build (compile) lock - independent of the GPU lock.
    pub fn build() -> Result<LockGuard, Busy> {
        Self::acquire(build_lock())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One serial test on ISOLATED lock paths (not the shared gpu/build locks, so
    // it can't race the other tests, which now take real flocks).
    #[test]
    fn flock_is_exclusive_releases_and_independent() {
        let base = std::env::temp_dir().join(format!("llmtune-locktest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let a = base.join("a.lock");
        let b = base.join("b.lock");

        let g = LockGuard::acquire(a.clone()).expect("first acquire succeeds");
        // flock is per open-file-description: a distinct open of the same path
        // must be denied while the first is held.
        assert!(
            LockGuard::acquire(a.clone()).is_err(),
            "second acquire is Busy"
        );
        // A different lock file is unaffected.
        assert!(
            LockGuard::acquire(b.clone()).is_ok(),
            "distinct lock is free"
        );

        drop(g);
        assert!(
            LockGuard::acquire(a).is_ok(),
            "released on drop -> reacquirable"
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
