// SPDX-License-Identifier: GPL-2.0-only
//! One persistence spine: directory resolution + crash-safe writes, defined
//! ONCE. Everything llmtune persists goes through here, so the state-dir
//! rules and the durable-write pattern can't drift between modules.
//!
//! Two state-dir flavours exist deliberately:
//!
//! * [`state_dir`] - per-run MUTABLE state (bench history, cluster markers).
//!   Chosen by PRIVILEGE, not existence: `/var/lib/llmtune` may exist
//!   root-owned (a sudo swap created it), which a non-root run cannot write -
//!   picking it then fails with EACCES. Root uses `/var/lib`; everyone else
//!   uses their XDG data dir, which is always writable.
//! * [`shared_state_dir`] - the SHARED system home for builds/models
//!   (`/var/lib/llmtune/{builds,src}`), where read-resolution must be
//!   identical for a user run and the internal sudo. If it exists, always use
//!   it; root may create it; a truly rootless box falls back to XDG.
//!
//! Both honour `LLMTUNE_STATE_DIR`. Config files live under
//! `~/.config/llmtune` ([`config_dir`]/[`config_file`]).

use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// XDG data dir fallback (`$XDG_DATA_HOME/llmtune`, else
/// `~/.local/share/llmtune`, else a temp dir).
fn xdg_data_dir() -> PathBuf {
    xdg_data_dir_from(
        std::env::var("XDG_DATA_HOME").ok().as_deref(),
        std::env::var("HOME").ok().as_deref(),
    )
}

/// Pure core of [`xdg_data_dir`] (env injected). Honors `$XDG_DATA_HOME`
/// first, per the basedir spec (an empty value counts as unset, also per
/// spec), then falls back to the literal `~/.local/share` default.
fn xdg_data_dir_from(xdg: Option<&str>, home: Option<&str>) -> PathBuf {
    if let Some(x) = xdg.filter(|x| !x.is_empty()) {
        return PathBuf::from(x).join("llmtune");
    }
    if let Some(h) = home {
        return PathBuf::from(h).join(".local/share/llmtune");
    }
    std::env::temp_dir().join("llmtune")
}

const SYSTEM_HOME: &str = "/var/lib/llmtune";

fn is_root() -> bool {
    unsafe { libc::geteuid() == 0 }
}

/// Per-run mutable state dir (history, markers). Privilege-based - see the
/// module docs for why this must NOT pick `/var/lib` by mere existence.
pub fn state_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LLMTUNE_STATE_DIR") {
        return PathBuf::from(d);
    }
    if is_root() {
        let v = PathBuf::from(SYSTEM_HOME);
        if v.is_dir() || fs::create_dir_all(&v).is_ok() {
            return v;
        }
    }
    xdg_data_dir()
}

/// The shared system home for builds/models - existence-based, so a user run
/// and the internal sudo resolve the SAME installed builds.
pub fn shared_state_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LLMTUNE_STATE_DIR") {
        return PathBuf::from(d);
    }
    let sys = PathBuf::from(SYSTEM_HOME);
    // If it exists, always use it (read/resolve is identical for user + root).
    if sys.is_dir() {
        return sys;
    }
    // Fresh box: root can create it (setup/build install self-elevate); a
    // truly rootless install falls back to the user's data dir.
    if is_root() && fs::create_dir_all(&sys).is_ok() {
        return sys;
    }
    xdg_data_dir()
}

/// llmtune's user config dir (`~/.config/llmtune`). None without $HOME.
///
/// The actuating TUI is run as `sudo llmtune` (staging systemd drop-ins needs
/// root), but the owner CONFIGURES it non-sudo (`llmtune profile set` writes
/// to their own `~/.config`). Under sudo `$HOME` is root's - so a naive read
/// would miss the user's profiles/settings and silently fall back to the
/// compiled seed (e.g. a profile pointing at an uninstalled build). Resolve the
/// INVOKING user's home when running under sudo so both paths see one config.
pub fn config_dir() -> Option<PathBuf> {
    if let Some(home) = sudo_invoker_home() {
        // The invoker's XDG_CONFIG_HOME does not survive into sudo's env, so
        // the invoker's ~/.config is the best resolvable location here.
        return Some(home.join(".config/llmtune"));
    }
    config_dir_from(
        std::env::var("XDG_CONFIG_HOME").ok().as_deref(),
        std::env::var("HOME").ok().as_deref(),
    )
}

/// Pure core of [`config_dir`] (env injected). `$XDG_CONFIG_HOME` first (empty
/// = unset, per the basedir spec), then `~/.config`.
fn config_dir_from(xdg: Option<&str>, home: Option<&str>) -> Option<PathBuf> {
    if let Some(x) = xdg.filter(|x| !x.is_empty()) {
        return Some(PathBuf::from(x).join("llmtune"));
    }
    Some(PathBuf::from(home?).join(".config/llmtune"))
}

/// The home dir of the user who invoked `sudo`, if any - looked up from
/// `/etc/passwd` (no libc pwd dependency). `None` when not under sudo, when the
/// invoker is root, or when the entry can't be found.
fn sudo_invoker_home() -> Option<PathBuf> {
    let user = std::env::var("SUDO_USER")
        .ok()
        .filter(|u| !u.is_empty() && u != "root")?;
    let passwd = fs::read_to_string("/etc/passwd").ok()?;
    home_from_passwd(&passwd, &user)
}

/// Extract `user`'s home dir (field 6) from `/etc/passwd` content. Pure.
fn home_from_passwd(passwd: &str, user: &str) -> Option<PathBuf> {
    for line in passwd.lines() {
        // name:passwd:uid:gid:gecos:home:shell
        let mut fields = line.split(':');
        if fields.next() == Some(user) {
            if let Some(home) = fields.nth(4).filter(|h| !h.is_empty()) {
                return Some(PathBuf::from(home));
            }
        }
    }
    None
}

/// A file inside the user config dir (`~/.config/llmtune/<name>`).
pub fn config_file(name: &str) -> Option<PathBuf> {
    config_dir().map(|d| d.join(name))
}

/// Per-process counter for unique tmp names (see [`write_durable`]).
static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Crash-safe write: temp -> fsync -> atomic rename -> fsync dir (the memtune
/// pattern). A power loss mid-write yields either the old or the new file,
/// never a corrupt one. Creates parent directories.
///
/// The temp name is unique per writer (`<path>.tmp.<pid>.<seq>`, same
/// directory so the rename stays same-filesystem): concurrent writers (the
/// proxy's set_served_model racing a TUI settings edit) each stage their own
/// tmp, so neither can truncate the other's mid-write - the outcome is
/// last-rename-wins, never a torn target. An existing target's permissions are
/// preserved across the rename-replace; for a file that must be owner-only
/// regardless of prior mode/umask (the api key), use [`write_durable_secret`].
pub fn write_durable(path: &Path, data: &[u8]) -> Result<()> {
    write_durable_impl(path, data, false)
}

/// Like [`write_durable`], but forces owner-only (0600) permissions on the
/// target regardless of the prior mode or the umask - for files that carry a
/// secret (the api key in settings.toml). No-op perms on non-unix.
pub fn write_durable_secret(path: &Path, data: &[u8]) -> Result<()> {
    write_durable_impl(path, data, true)
}

#[cfg(unix)]
fn set_owner_only(f: &fs::File) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    f.set_permissions(fs::Permissions::from_mode(0o600))
}
#[cfg(not(unix))]
fn set_owner_only(_f: &fs::File) -> std::io::Result<()> {
    Ok(())
}

fn write_durable_impl(path: &Path, data: &[u8], secret: bool) -> Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let tmp = PathBuf::from(tmp_name);
    let staged = (|| -> Result<()> {
        let mut f = fs::File::create(&tmp)?;
        if secret {
            // Force 0600 even if a prior file was created world-readable.
            set_owner_only(&f)?;
        } else if let Ok(meta) = fs::metadata(path) {
            // fchmod the tmp to the prior file's mode before it takes the name.
            f.set_permissions(meta.permissions())?;
        }
        f.write_all(data)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = staged.and_then(|()| fs::rename(&tmp, path).map_err(Into::into)) {
        let _ = fs::remove_file(&tmp); // don't leave a stray tmp on failure
        return Err(e);
    }
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// A blocking advisory lock on a sibling `.lock` file, held for a read-modify-
/// write so two writers can't lose an update. Released when the fd closes (drop
/// or process exit). Best-effort: a no-op if the lock file can't be opened.
pub(crate) struct RmwLock(#[allow(dead_code)] Option<fs::File>);

impl RmwLock {
    pub(crate) fn acquire(path: &Path) -> RmwLock {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = path.with_extension("lock");
        let f = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .ok();
        if let Some(f) = &f {
            use std::os::unix::io::AsRawFd;
            // Bounded wait: serialize concurrent writers so no update is lost, but
            // never hang on a peer that crashed or wedged mid-hold. On timeout we
            // proceed best-effort, the same fallback as when the lock is a no-op.
            if !crate::lock::flock_ex_timeout(f.as_raw_fd(), crate::lock::RMW_LOCK_TIMEOUT) {
                eprintln!("llmtune: config rmw lock timed out; proceeding unlocked");
            }
        }
        RmwLock(f)
    }
}

/// Serialize `value` as pretty TOML under `header` and write it atomically.
pub fn save_toml_atomic<T: serde::Serialize>(path: &Path, header: &str, value: &T) -> Result<()> {
    let txt =
        toml::to_string_pretty(value).with_context(|| format!("serializing {}", path.display()))?;
    write_durable(path, format!("{header}{txt}").as_bytes())
}

/// Locked, atomic read-modify-write persisting the result with owner-only (0600)
/// permissions - for a state file that carries a secret (settings.toml's api
/// key). The whole file is 0600 (the secret shares it with the other settings).
/// A blocking lock serializes concurrent writers so no update is lost; a missing
/// file defaults cleanly, while a present-but-unparseable file is preserved aside
/// (never silently reset to defaults, which would wipe the key).
pub fn edit_toml_secret<T>(path: &Path, header: &str, edit: impl FnOnce(&mut T)) -> Result<()>
where
    T: Default + serde::de::DeserializeOwned + serde::Serialize,
{
    // Serialize the whole read-modify-write against concurrent writers (the
    // proxy's set_served_model racing an operator's set_api_key) so a lost update
    // can't silently drop the api_key. Held until this function returns.
    let _lk = RmwLock::acquire(path);
    // A MISSING file defaults cleanly. A file that EXISTS but doesn't parse must
    // NOT be silently reset to defaults (that would wipe the api_key + exposure on
    // one corrupt byte); preserve it aside and warn before starting fresh.
    let mut v: T = match fs::read_to_string(path) {
        Ok(s) => match toml::from_str(&s) {
            Ok(v) => v,
            Err(e) => {
                // Unique suffix so a second corruption doesn't clobber the first
                // aside (which may hold the only recoverable api_key).
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let aside = path.with_extension(format!("bad.{stamp}"));
                // The aside exists to preserve the possibly-only copy of the
                // api_key; if it can't be made, BAIL rather than proceed and
                // overwrite the unparseable original.
                fs::rename(path, &aside).with_context(|| {
                    format!(
                        "{} did not parse ({e}) and could not be preserved aside as {} - \
                         refusing to overwrite it",
                        path.display(),
                        aside.display()
                    )
                })?;
                eprintln!(
                    "warning: {} did not parse ({e}); moved aside to {} and starting fresh",
                    path.display(),
                    aside.display()
                );
                T::default()
            }
        },
        Err(_) => T::default(),
    };
    edit(&mut v);
    let txt =
        toml::to_string_pretty(&v).with_context(|| format!("serializing {}", path.display()))?;
    write_durable_secret(path, format!("{header}{txt}").as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
    struct Demo {
        #[serde(default)]
        a: Option<String>,
        #[serde(default)]
        b: Option<String>,
    }

    #[test]
    fn write_durable_creates_dirs_and_replaces() {
        let dir = std::env::temp_dir().join(format!("llmtune-paths-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = dir.join("deep/nested/file.json");
        write_durable(&p, b"one").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"one");
        write_durable(&p, b"two").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"two");
        // no temp file left behind (of either the old fixed or the unique naming)
        let leftovers: Vec<_> = fs::read_dir(p.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name() != "file.json")
            .collect();
        assert!(leftovers.is_empty(), "stray files: {leftovers:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn write_durable_preserves_target_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("llmtune-mode-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = dir.join("settings.toml");
        write_durable(&p, b"api_key = \"secret\"").unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o600)).unwrap();
        // the rename-replace must not widen a user's 0600 back to umask default
        write_durable(&p, b"api_key = \"rotated\"").unwrap();
        let mode = fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode widened by write_durable");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn write_durable_secret_forces_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("llmtune-secret-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = dir.join("settings.toml");
        // First create: must be 0600 even though the umask is typically 022.
        write_durable_secret(&p, b"api_key = \"secret\"").unwrap();
        let mode = fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "secret write must create 0600");
        // A pre-existing world-readable file must be tightened, not preserved.
        fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();
        write_durable_secret(&p, b"api_key = \"rotated\"").unwrap();
        let mode = fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "secret write must force 0600 over a prior 0644"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_durable_concurrent_writers_never_tear() {
        // Two writers hammering the same path (the proxy vs the TUI on
        // settings.toml). With a shared fixed tmp name they truncated each
        // other; with unique tmps every observable file state is one writer's
        // COMPLETE payload.
        let dir = std::env::temp_dir().join(format!("llmtune-race-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = dir.join("settings.toml");
        write_durable(&p, &[b'A'; 512]).unwrap();
        let mk = |byte: u8| {
            let p = p.clone();
            std::thread::spawn(move || {
                for _ in 0..50 {
                    write_durable(&p, &[byte; 512]).unwrap();
                    let got = fs::read(&p).unwrap();
                    assert_eq!(got.len(), 512, "torn read: {} bytes", got.len());
                    assert!(
                        got.iter().all(|&b| b == got[0]),
                        "mixed payload observed - torn write"
                    );
                }
            })
        };
        let a = mk(b'A');
        let b = mk(b'B');
        a.join().unwrap();
        b.join().unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_toml_secret_read_modify_write_preserves_other_fields() {
        let dir = std::env::temp_dir().join(format!("llmtune-toml-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = dir.join("settings.toml");
        // first edit on a missing file starts from Default
        edit_toml_secret(&p, "# hdr\n\n", |d: &mut Demo| d.a = Some("x".into())).unwrap();
        // a second edit must keep the first field intact (RMW, not overwrite)
        edit_toml_secret(&p, "# hdr\n\n", |d: &mut Demo| d.b = Some("y".into())).unwrap();
        let txt = fs::read_to_string(&p).unwrap();
        assert!(txt.starts_with("# hdr"));
        let back: Demo = toml::from_str(&txt).unwrap();
        assert_eq!(back.a.as_deref(), Some("x"));
        assert_eq!(back.b.as_deref(), Some("y"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_file_lands_under_llmtune_config() {
        if let Some(p) = config_file("fleet.toml") {
            let s = p.display().to_string();
            // XDG_CONFIG_HOME may or may not be set in the test environment;
            // either way the file lives under a llmtune config dir.
            assert!(s.ends_with("llmtune/fleet.toml"), "{s}");
        }
    }

    #[test]
    fn xdg_dirs_honor_the_basedir_vars() {
        // XDG var wins; empty counts as unset (per the basedir spec).
        assert_eq!(
            xdg_data_dir_from(Some("/xdg/data"), Some("/home/u")),
            PathBuf::from("/xdg/data/llmtune")
        );
        assert_eq!(
            xdg_data_dir_from(Some(""), Some("/home/u")),
            PathBuf::from("/home/u/.local/share/llmtune")
        );
        assert_eq!(
            xdg_data_dir_from(None, Some("/home/u")),
            PathBuf::from("/home/u/.local/share/llmtune")
        );
        assert_eq!(
            config_dir_from(Some("/xdg/cfg"), Some("/home/u")),
            Some(PathBuf::from("/xdg/cfg/llmtune"))
        );
        assert_eq!(
            config_dir_from(Some(""), Some("/home/u")),
            Some(PathBuf::from("/home/u/.config/llmtune"))
        );
        assert_eq!(config_dir_from(None, None), None);
    }

    #[test]
    fn passwd_home_lookup_picks_field_six() {
        let passwd = "root:x:0:0:root:/root:/bin/bash\n\
                      user:x:1000:1000:User,,,:/home/user:/usr/bin/fish\n\
                      nobody:x:65534:65534:Nobody:/:/usr/sbin/nologin\n";
        assert_eq!(
            home_from_passwd(passwd, "user"),
            Some(PathBuf::from("/home/user"))
        );
        assert_eq!(
            home_from_passwd(passwd, "root"),
            Some(PathBuf::from("/root"))
        );
        // A user not present yields None (caller falls back to $HOME).
        assert_eq!(home_from_passwd(passwd, "ghost"), None);
        // A prefix of a real name must not false-match (exact field-1 compare).
        assert_eq!(home_from_passwd(passwd, "st"), None);
    }
}
