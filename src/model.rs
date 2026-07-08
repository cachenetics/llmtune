// SPDX-License-Identifier: GPL-2.0-only
//! Model discovery: enumerate GGUF files and read what we need from each
//! header (architecture, training context) without loading any weights.

use anyhow::{bail, Result};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A discovered model file.
#[derive(Debug, Clone)]
pub struct Model {
    pub path: PathBuf,
    pub name: String,
    /// GGUF `general.architecture`, or "?" if unreadable.
    pub arch: String,
    /// Parameter-size token parsed from the filename, e.g. "35B-A3B".
    pub params: Option<String>,
    /// Quantization token parsed from the filename, e.g. "IQ2_M".
    pub quant: Option<String>,
    pub size_bytes: u64,
    /// `{arch}.context_length` from the header, if present.
    pub ctx_max: Option<u64>,
    /// True if the GGUF carries MTP / multi-token-prediction layers (the
    /// `{arch}.nextn_predict_layers` header key). A model WITHOUT them cannot be
    /// served with `--spec-type draft-mtp` - llama-server aborts on load - so the
    /// swap path strips the MTP flags for these. See swap::adjust_flags.
    pub has_mtp: bool,
}

impl Model {
    pub fn size_gib(&self) -> f64 {
        self.size_bytes as f64 / (1u64 << 30) as f64
    }
}

/// A GGUF metadata scalar (arrays are skipped during parse). Some variants carry
/// values we parse for completeness but don't currently read.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum GVal {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
}

fn read_n<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn rd_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_n(r, 4)?.try_into().unwrap()))
}

fn rd_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_n(r, 8)?.try_into().unwrap()))
}

fn rd_str<R: Read>(r: &mut R) -> Result<String> {
    let n = rd_u64(r)? as usize;
    // Guard against a corrupt/huge length blowing up memory.
    if n > 64 * 1024 * 1024 {
        bail!("gguf string length absurd: {n}");
    }
    Ok(String::from_utf8_lossy(&read_n(r, n)?).into_owned())
}

/// Max GGUF array nesting we'll follow. Real GGUF has no nested arrays; a bound
/// stops a crafted file from recursing us into a stack overflow.
const MAX_GGUF_DEPTH: u32 = 8;

/// Read one typed value; returns None for arrays (consumed but not stored).
/// `depth` bounds array nesting. An unknown type errors (rather than silently
/// consuming zero bytes) so a malformed file can't spin `rd_val` forever inside
/// an array loop.
fn rd_val<R: Read>(r: &mut R, t: u32, depth: u32) -> Result<Option<GVal>> {
    Ok(match t {
        0 => Some(GVal::U(read_n(r, 1)?[0] as u64)),
        1 => Some(GVal::I(
            i8::from_le_bytes(read_n(r, 1)?.try_into().unwrap()) as i64,
        )),
        2 => Some(GVal::U(
            u16::from_le_bytes(read_n(r, 2)?.try_into().unwrap()) as u64,
        )),
        3 => Some(GVal::I(
            i16::from_le_bytes(read_n(r, 2)?.try_into().unwrap()) as i64,
        )),
        4 => Some(GVal::U(rd_u32(r)? as u64)),
        5 => Some(GVal::I(
            i32::from_le_bytes(read_n(r, 4)?.try_into().unwrap()) as i64,
        )),
        6 => Some(GVal::F(
            f32::from_le_bytes(read_n(r, 4)?.try_into().unwrap()) as f64,
        )),
        7 => Some(GVal::Bool(read_n(r, 1)?[0] != 0)),
        8 => Some(GVal::Str(rd_str(r)?)),
        9 => {
            // array: element type (u32), count (u64), then elements
            if depth >= MAX_GGUF_DEPTH {
                bail!("gguf array nesting too deep (>{MAX_GGUF_DEPTH})");
            }
            let et = rd_u32(r)?;
            let cnt = rd_u64(r)?;
            for _ in 0..cnt {
                rd_val(r, et, depth + 1)?;
            }
            None
        }
        10 => Some(GVal::U(rd_u64(r)?)),
        11 => Some(GVal::I(i64::from_le_bytes(
            read_n(r, 8)?.try_into().unwrap(),
        ))),
        12 => Some(GVal::F(f64::from_le_bytes(
            read_n(r, 8)?.try_into().unwrap(),
        ))),
        other => bail!("gguf unknown value type {other}"),
    })
}

/// Does the GGUF metadata advertise MTP layers? Arch-agnostic: any key whose
/// name contains `nextn` or `mtp` with a truthy value (the layer-count key is
/// `{arch}.nextn_predict_layers` today, but this holds for any future arch that
/// keeps the naming convention). Presence of a non-numeric such key also counts.
fn meta_indicates_mtp(meta: &BTreeMap<String, GVal>) -> bool {
    meta.iter().any(|(k, v)| {
        let k = k.to_lowercase();
        if !(k.contains("nextn") || k.contains("mtp")) {
            return false;
        }
        match v {
            GVal::U(n) => *n > 0,
            GVal::I(n) => *n > 0,
            GVal::Bool(b) => *b,
            _ => true,
        }
    })
}

/// Ground-truth MTP detection: scan the tensor-info section for an MTP layer
/// tensor (`nextn`/`mtp` in the name). Independent of the metadata key naming,
/// so it recognizes MTP in model families whose header conventions we don't know
/// yet. Best-effort - an unreadable/short header just returns false.
fn has_mtp_tensors(path: &Path) -> bool {
    let Ok(f) = File::open(path) else {
        return false;
    };
    let mut r = BufReader::new(f);
    if read_n(&mut r, 4).ok().as_deref() != Some(b"GGUF") {
        return false;
    }
    // header: version, tensor_count, kv_count
    if rd_u32(&mut r).is_err() {
        return false;
    }
    let (Ok(tensor_count), Ok(n_kv)) = (rd_u64(&mut r), rd_u64(&mut r)) else {
        return false;
    };
    // skip the metadata KV block to reach the tensor-info section
    for _ in 0..n_kv {
        if rd_str(&mut r).is_err() {
            return false;
        }
        match rd_u32(&mut r) {
            Ok(t) if rd_val(&mut r, t, 0).is_ok() => {}
            _ => return false,
        }
    }
    // tensor infos: name (str), n_dims (u32), dims (u64*n_dims), type (u32), offset (u64)
    for _ in 0..tensor_count {
        let Ok(name) = rd_str(&mut r) else {
            return false;
        };
        let n = name.to_lowercase();
        if n.contains("nextn") || n.contains("mtp") {
            return true;
        }
        let Ok(nd) = rd_u32(&mut r) else {
            return false;
        };
        for _ in 0..nd {
            if rd_u64(&mut r).is_err() {
                return false;
            }
        }
        if rd_u32(&mut r).is_err() || rd_u64(&mut r).is_err() {
            return false;
        }
    }
    false
}

/// Parse the GGUF metadata KV block (scalars only) from a file.
fn read_gguf_meta(path: &Path) -> Result<BTreeMap<String, GVal>> {
    let mut r = BufReader::new(File::open(path)?);
    if read_n(&mut r, 4)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let _version = rd_u32(&mut r)?;
    let _tensor_count = rd_u64(&mut r)?;
    let n_kv = rd_u64(&mut r)?;
    let mut map = BTreeMap::new();
    for _ in 0..n_kv {
        let k = rd_str(&mut r)?;
        let t = rd_u32(&mut r)?;
        if let Some(v) = rd_val(&mut r, t, 0)? {
            map.insert(k, v);
        }
    }
    Ok(map)
}

fn param_from_name(name: &str) -> Option<String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"(?i)(\d+x?\d*B(?:-A\d+B)?)").unwrap());
    re.captures(name).map(|c| c[1].to_uppercase())
}

fn quant_from_name(name: &str) -> Option<String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re =
        RE.get_or_init(|| regex::Regex::new(r"((?:UD-)?(?:IQ|Q)\d[\w]*|BF16|F16|F32)").unwrap());
    re.captures(name).map(|c| c[1].replace("UD-", ""))
}

/// True for a chat/generation model (we exclude embedding/rerank models).
pub fn is_chat_model(name: &str) -> bool {
    let n = name.to_lowercase();
    !n.contains("embedding") && !n.contains("embed") && !n.contains("rerank")
}

/// Architecture families with recurrent / hybrid state-space blocks. These
/// CANNOT be split over llama.cpp RPC: the worker rejects the recurrent-state
/// graph at load (`[create_node] invalid data ptr`) and the head aborts, so
/// `cluster up` refuses to pool them. Single-node serving is unaffected - they
/// run fine on one box.
///
/// Prefix-matched (lowercase) against GGUF `general.architecture`. Extend this
/// list as new recurrent/hybrid families appear; only dense transformers
/// (qwen*, gemma*, llama*, ...) pool.
const RECURRENT_ARCH_PREFIXES: &[&str] = &[
    "lfm2",          // Liquid LFM2/LFM2.5 (lfm2, lfm2moe: conv + attention hybrid)
    "mamba",         // Mamba / Mamba2 state-space
    "rwkv",          // RWKV (rwkv6, rwkv6qwen2, rwkv7)
    "arwkv",         // ARWKV7
    "jamba",         // Jamba (Mamba + attention hybrid)
    "falcon-h1",     // Falcon-H1 hybrid
    "granitehybrid", // Granite 4 hybrid
    "nemotron_h",    // Nemotron-H hybrid
    "plamo2",        // PLaMo2 hybrid
    "qwen3next",     // Qwen3-Next (gated DeltaNet hybrid)
];

/// Is `arch` a recurrent/hybrid state-space family (unsafe to shard over RPC)?
pub fn is_recurrent_arch(arch: &str) -> bool {
    let a = arch.to_lowercase();
    RECURRENT_ARCH_PREFIXES.iter().any(|p| a.starts_with(p))
}

/// Read a single model file into a [`Model`]. A header that fails to parse is
/// tolerated - `arch` falls back to "?" so a broken file still lists.
pub fn load(path: &Path) -> Result<Model> {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let size_bytes = fs::metadata(path)?.len();
    let meta = read_gguf_meta(path).unwrap_or_default();
    let arch = match meta.get("general.architecture") {
        Some(GVal::Str(s)) => s.clone(),
        _ => "?".to_string(),
    };
    let ctx_max = meta
        .get(&format!("{arch}.context_length"))
        .and_then(|v| match v {
            GVal::U(n) => Some(*n),
            GVal::I(n) => Some(*n as u64),
            _ => None,
        });
    // MTP presence, detected arch-agnostically so it holds for future model
    // families: any `nextn`/`mtp` header key (the count layers advertise
    // themselves with), else the actual MTP tensors in the file.
    let has_mtp = meta_indicates_mtp(&meta) || has_mtp_tensors(path);
    Ok(Model {
        path: path.to_path_buf(),
        name: name.clone(),
        arch,
        params: param_from_name(&name),
        quant: quant_from_name(&name),
        size_bytes,
        ctx_max,
        has_mtp,
    })
}

/// Read an unsigned scalar from the metadata by key (coercing signed -> unsigned).
fn meta_u64(meta: &BTreeMap<String, GVal>, key: &str) -> Option<u64> {
    match meta.get(key)? {
        GVal::U(n) => Some(*n),
        GVal::I(n) if *n >= 0 => Some(*n as u64),
        _ => None,
    }
}

/// Read the KV-cache-sizing dimensions from a model's GGUF header. Returns None
/// if the header is unreadable or lacks the core fields (layer/head/embedding),
/// in which case callers fall back to a weights-only memory estimate.
pub fn read_dims(path: &Path) -> Option<crate::mem::Dims> {
    let meta = read_gguf_meta(path).ok()?;
    let arch = match meta.get("general.architecture") {
        Some(GVal::Str(s)) => s.clone(),
        _ => return None,
    };
    let n_layers = meta_u64(&meta, &format!("{arch}.block_count"))?;
    let n_head = meta_u64(&meta, &format!("{arch}.attention.head_count"))?;
    let n_embd = meta_u64(&meta, &format!("{arch}.embedding_length"))?;
    if n_head == 0 || n_layers == 0 {
        return None;
    }
    // KV heads default to query heads (MHA) when the GQA key is absent.
    let n_head_kv = meta_u64(&meta, &format!("{arch}.attention.head_count_kv")).unwrap_or(n_head);
    // head_dim: explicit key_length if present, else embedding/heads.
    let head_dim =
        meta_u64(&meta, &format!("{arch}.attention.key_length")).unwrap_or(n_embd / n_head.max(1));
    let ctx_train = meta_u64(&meta, &format!("{arch}.context_length"));
    Some(crate::mem::Dims {
        n_layers,
        n_head,
        n_head_kv,
        head_dim,
        ctx_train,
    })
}

/// Friendly, actionable guidance on where models live + how to add one.
/// Shown wherever a user might be missing models (empty `node list`, `doctor`,
/// `setup`) so the drop location is never a mystery.
pub fn where_to_put_models(models_dir: &str) -> String {
    format!(
        "Models directory: {models_dir}\n\
         First-time setup creates it (owned by you): run `llmtune setup`.\n\
         Then drop GGUF model files here - llmtune auto-detects `*.gguf` by architecture:\n\
         \x20 huggingface-cli download unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir {models_dir}\n\
         \x20 # or just copy/move any .gguf into that folder\n\
         Put models on a bigger disk: set `models_dir` in ~/.config/llmtune/fleet.toml, \
         export $LLMTUNE_MODELS_DIR, or symlink {models_dir} at an existing folder."
    )
}

/// Create the models directory if it doesn't exist, so it's a concrete place to
/// drop files into. Best-effort - returns whether it now exists.
pub fn ensure_dir(models_dir: &str) -> bool {
    let p = Path::new(models_dir);
    if !p.exists() {
        let _ = fs::create_dir_all(p);
    }
    p.is_dir()
}

/// Discover chat models in a directory, sorted by name. Missing dir -> empty.
pub fn discover(models_dir: &Path) -> Result<Vec<Model>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(models_dir) {
        Ok(rd) => rd,
        Err(_) => return Ok(out),
    };
    for ent in rd.flatten() {
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        let name = p
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !is_chat_model(&name) {
            continue;
        }
        if let Ok(m) = load(&p) {
            out.push(m);
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn gguf_array_unknown_element_type_errors_not_hangs() {
        // type 9 array, element type = 999 (unknown), count = u64::MAX.
        // Pre-fix this spun forever (unknown type consumed 0 bytes); now it errors.
        let mut b = Vec::new();
        b.extend_from_slice(&999u32.to_le_bytes()); // element type
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // count
        let mut r = Cursor::new(b);
        assert!(rd_val(&mut r, 9, 0).is_err());
    }

    #[test]
    fn gguf_nested_arrays_depth_capped() {
        // Each array declares one element that is itself an array. Pre-fix this
        // recursed to a stack overflow; the depth cap now errors first.
        let mut b = Vec::new();
        for _ in 0..(MAX_GGUF_DEPTH + 4) {
            b.extend_from_slice(&9u32.to_le_bytes()); // element type = array
            b.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        }
        let mut r = Cursor::new(b);
        assert!(rd_val(&mut r, 9, 0).is_err());
    }

    #[test]
    fn gguf_scalar_unknown_type_errors() {
        let mut r = Cursor::new(Vec::new());
        assert!(rd_val(&mut r, 42, 0).is_err());
    }

    #[test]
    fn mtp_detected_from_any_nextn_or_mtp_key() {
        // Today's key.
        let mut m = BTreeMap::new();
        m.insert("qwen35.nextn_predict_layers".to_string(), GVal::U(1));
        assert!(meta_indicates_mtp(&m));
        // A future arch keeps the convention -> still detected (arch-agnostic).
        let mut m = BTreeMap::new();
        m.insert("qwen4moe.nextn_predict_layers".to_string(), GVal::U(2));
        assert!(meta_indicates_mtp(&m));
        // An `mtp`-named bool key.
        let mut m = BTreeMap::new();
        m.insert("some_arch.has_mtp".to_string(), GVal::Bool(true));
        assert!(meta_indicates_mtp(&m));
    }

    /// Write a minimal valid GGUF (one `general.architecture` string KV, then the
    /// given tensor names) so the tensor-scan path can be exercised without a real
    /// multi-GB model. Also verifies the metadata-skip stays aligned.
    fn write_min_gguf(path: &Path, arch: &str, tensor_names: &[&str]) {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&(tensor_names.len() as u64).to_le_bytes()); // tensor_count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv_count
                                                  // one string KV: general.architecture = arch
        let k = b"general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&8u32.to_le_bytes()); // GGUF string type
        b.extend_from_slice(&(arch.len() as u64).to_le_bytes());
        b.extend_from_slice(arch.as_bytes());
        // tensor infos
        for n in tensor_names {
            b.extend_from_slice(&(n.len() as u64).to_le_bytes());
            b.extend_from_slice(n.as_bytes());
            b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            b.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
            b.extend_from_slice(&0u32.to_le_bytes()); // ggml type
            b.extend_from_slice(&0u64.to_le_bytes()); // offset
        }
        fs::write(path, b).unwrap();
    }

    #[test]
    fn tensor_scan_detects_mtp_independently_of_metadata() {
        // The future-proof fallback: MTP recognized from an actual nextn/mtp layer
        // tensor even when NO metadata key advertises it.
        let dir = std::env::temp_dir().join(format!("llt-mtp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mtp = dir.join("mtp.gguf");
        write_min_gguf(
            &mtp,
            "future_arch",
            &[
                "token_embd.weight",
                "blk.0.attn_q.weight",
                "blk.48.nextn.eh_proj.weight",
            ],
        );
        assert!(has_mtp_tensors(&mtp), "nextn tensor must be detected");
        // and the whole model reads has_mtp=true via the tensor path (no key)
        assert!(load(&mtp).unwrap().has_mtp);

        let plain = dir.join("plain.gguf");
        write_min_gguf(
            &plain,
            "future_arch",
            &["token_embd.weight", "blk.0.attn_q.weight", "output.weight"],
        );
        assert!(!has_mtp_tensors(&plain));
        assert!(!load(&plain).unwrap().has_mtp);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mtp_not_detected_when_absent_or_zero() {
        let mut m = BTreeMap::new();
        m.insert("qwen35.block_count".to_string(), GVal::U(48));
        assert!(!meta_indicates_mtp(&m));
        // A present-but-zero count is not MTP.
        let mut m = BTreeMap::new();
        m.insert("qwen35.nextn_predict_layers".to_string(), GVal::U(0));
        assert!(!meta_indicates_mtp(&m));
    }

    #[test]
    fn quant_parse() {
        assert_eq!(
            quant_from_name("Qwen3.5-9B-IQ2_M.gguf").as_deref(),
            Some("IQ2_M")
        );
        assert_eq!(
            quant_from_name("foo-UD-Q4_K_XL.gguf").as_deref(),
            Some("Q4_K_XL")
        );
        assert_eq!(quant_from_name("model-BF16.gguf").as_deref(), Some("BF16"));
        assert_eq!(quant_from_name("no-quant-here.gguf"), None);
    }

    #[test]
    fn param_parse() {
        assert_eq!(
            param_from_name("Qwen3.6-35B-A3B-IQ2.gguf").as_deref(),
            Some("35B-A3B")
        );
        assert_eq!(param_from_name("gemma-9B-q4.gguf").as_deref(), Some("9B"));
    }

    #[test]
    fn chat_filter() {
        assert!(is_chat_model("Qwen3.5-9B.gguf"));
        assert!(!is_chat_model("nomic-embedding-text.gguf"));
        assert!(!is_chat_model("bge-rerank.gguf"));
    }

    #[test]
    fn recurrent_arch_prefixes_match_families() {
        // The exact arch that crashed the RPC verification, plus prefix variants.
        for a in [
            "lfm2",
            "lfm2moe",
            "mamba",
            "mamba2",
            "rwkv6",
            "rwkv7",
            "rwkv6qwen2",
            "arwkv7",
            "jamba",
            "qwen3next",
            "granitehybrid",
            "nemotron_h",
            "plamo2",
            "falcon-h1",
        ] {
            assert!(is_recurrent_arch(a), "{a} must be flagged recurrent/hybrid");
        }
        // Case-insensitive (GGUF headers are usually lowercase, but don't rely on it).
        assert!(is_recurrent_arch("LFM2"));
    }

    #[test]
    fn dense_transformers_are_not_recurrent() {
        for a in [
            "qwen35",
            "qwen35moe",
            "gemma3",
            "llama",
            "phi4",
            "falcon",
            "?",
        ] {
            assert!(!is_recurrent_arch(a), "{a} must pool over RPC");
        }
    }

    #[test]
    fn gguf_roundtrip() {
        // Build a minimal GGUF with one string KV (general.architecture=llama)
        // and one uint32 KV (llama.context_length=4096), then parse it back.
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&2u64.to_le_bytes()); // kv count
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes()); // type string
        put_str(&mut b, "llama");
        put_str(&mut b, "llama.context_length");
        b.extend_from_slice(&4u32.to_le_bytes()); // type uint32
        b.extend_from_slice(&4096u32.to_le_bytes());

        let dir = std::env::temp_dir().join(format!("llmtune-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let f = dir.join("tiny-7B-Q4_K_M.gguf");
        fs::write(&f, &b).unwrap();
        let m = load(&f).unwrap();
        assert_eq!(m.arch, "llama");
        assert_eq!(m.ctx_max, Some(4096));
        assert_eq!(m.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(m.params.as_deref(), Some("7B"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_dims_from_header() {
        // GGUF with the KV-sizing dims: block_count, head counts (GQA), embedding,
        // context. head_dim is derived as embedding/heads (no key_length key).
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&5u64.to_le_bytes()); // kv count
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        let put_u32 = |b: &mut Vec<u8>, k: &str, v: u32| {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&4u32.to_le_bytes()); // type uint32
            b.extend_from_slice(&v.to_le_bytes());
        };
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut b, "qwen3");
        put_u32(&mut b, "qwen3.block_count", 28);
        put_u32(&mut b, "qwen3.attention.head_count", 16);
        put_u32(&mut b, "qwen3.attention.head_count_kv", 4); // GQA
        put_u32(&mut b, "qwen3.embedding_length", 2048);

        let dir = std::env::temp_dir().join(format!("llmtune-dims-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let f = dir.join("q.gguf");
        fs::write(&f, &b).unwrap();
        let d = read_dims(&f).expect("dims must parse");
        assert_eq!(d.n_layers, 28);
        assert_eq!(d.n_head, 16);
        assert_eq!(d.n_head_kv, 4);
        assert_eq!(d.head_dim, 2048 / 16); // embedding/heads
        let _ = fs::remove_dir_all(&dir);

        // A header missing the dims (only arch) -> None (weights-only fallback).
        let mut bare: Vec<u8> = Vec::new();
        bare.extend_from_slice(b"GGUF");
        bare.extend_from_slice(&3u32.to_le_bytes());
        bare.extend_from_slice(&0u64.to_le_bytes());
        bare.extend_from_slice(&1u64.to_le_bytes());
        put_str(&mut bare, "general.architecture");
        bare.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut bare, "llama");
        let dir2 = std::env::temp_dir().join(format!("llmtune-dims2-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir2);
        let f2 = dir2.join("bare.gguf");
        fs::write(&f2, &bare).unwrap();
        assert!(read_dims(&f2).is_none());
        let _ = fs::remove_dir_all(&dir2);
    }
}
