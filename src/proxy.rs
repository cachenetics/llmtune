// SPDX-License-Identifier: GPL-2.0-only
//! Swap-on-demand proxy - a stable OpenAI-compatible port that makes a
//! one-model-at-a-time BC-250 look like it hosts your whole library.
//!
//! A single BC-250 can only serve one model at a time (16 GiB UMA). This proxy
//! listens on a fixed port and speaks the OpenAI API; when a request names a model
//! that isn't the one currently loaded, it swaps llama-server to that model (the
//! same guarded, auto-reverting swap as `node load`), waits for health, then
//! transparently proxies the request to the real llama-server. To a client it
//! looks like every model in `models_dir` is hosted at one endpoint.
//!
//! Binds 127.0.0.1 by default. Liveness hardening (the proxy is a supported
//! LAN exposure via `endpoint expose`):
//!
//! - A small worker pool serves requests (bounded concurrency) - one slow or
//!   hung client can no longer wedge the endpoint for everyone else.
//! - Every accepted socket carries a read/write deadline (SO_RCVTIMEO /
//!   SO_SNDTIMEO inherited from the listener), so a client that stalls
//!   mid-body or never reads a streaming response is dropped, not waited on
//!   forever.
//! - GPU swaps stay strictly serialized behind [`SwapGate`], which also
//!   enforces a minimum hold between swaps: two clients alternating model
//!   names get 503 + Retry-After instead of melting the GPU in a stop/load
//!   loop (the swap-thrash guard).
//!
//! The pure helpers (path classification, model extraction, the model list,
//! the gate decision) are unit-tested; `serve` is the IO loop.

use crate::config::{Config, Node};
use crate::{llama, model, nodeops, settings};
use anyhow::{anyhow, Result};
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Max inbound request body the proxy will buffer (chat requests are small; this
/// stops a large/slow POST from exhausting memory once bound off-loopback).
const MAX_BODY_BYTES: u64 = 64 * 1024 * 1024;

/// Concurrent request workers. Small: enough that one stalled client doesn't
/// starve the rest, few enough that llama-server isn't flooded.
const PROXY_WORKERS: usize = 4;

/// Per-socket read/write deadline (any single blocking read/write on a client
/// socket, not the whole request - a streaming response that keeps moving is
/// never cut off).
const SOCKET_TIMEOUT: Duration = Duration::from_secs(30);

/// Constant-time byte equality - avoids leaking the api key through comparison
/// timing.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// True if the request carries `Authorization: Bearer <key>` matching `key`.
fn authorized(req: &tiny_http::Request, key: &str) -> bool {
    let want = format!("Bearer {key}");
    req.headers().iter().any(|h| {
        h.field
            .as_str()
            .as_str()
            .eq_ignore_ascii_case("Authorization")
            && ct_eq(h.value.as_str().as_bytes(), want.as_bytes())
    })
}

/// Where the proxy listens, and the swap-thrash hold.
pub struct ProxyOpts {
    pub host: String,
    pub port: u16,
    /// Minimum interval between proxy-triggered model swaps (0 disables the
    /// thrash guard).
    pub swap_hold: Duration,
}

// ---------------------------------------------------------------------------
// Swap-thrash guard: serialize GPU swaps and enforce a minimum hold between
// them. Without this, two clients alternating model names put the node in a
// permanent stop/swap/health-wait loop - anonymous once exposed with auth off.
// ---------------------------------------------------------------------------

/// What the gate says about a request that needs a model swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateDecision {
    /// Proceed with the swap (the gate is now held; call `finish_swap` after).
    Proceed,
    /// A swap is already in flight - retry shortly.
    Busy,
    /// Within the minimum hold after the last swap - retry in `secs`.
    HoldFor(u64),
}

/// Pure decision core (clock injected as "elapsed since the last swap"), so
/// the hold/busy logic is unit-testable without sleeping.
fn gate_decision(
    swapping: bool,
    since_last_swap: Option<Duration>,
    hold: Duration,
) -> GateDecision {
    if swapping {
        return GateDecision::Busy;
    }
    if let Some(elapsed) = since_last_swap {
        if elapsed < hold {
            // Ceil so "Retry-After: 0" can't happen while still held.
            let remaining = hold - elapsed;
            return GateDecision::HoldFor(remaining.as_secs().max(1));
        }
    }
    GateDecision::Proceed
}

struct GateState {
    swapping: bool,
    last_swap_done: Option<Instant>,
}

struct SwapGate {
    hold: Duration,
    state: Mutex<GateState>,
}

impl SwapGate {
    fn new(hold: Duration) -> Self {
        SwapGate {
            hold,
            state: Mutex::new(GateState {
                swapping: false,
                last_swap_done: None,
            }),
        }
    }

    /// Try to acquire the right to swap. On `Proceed` the caller MUST call
    /// [`finish_swap`](Self::finish_swap) when the swap attempt ends.
    fn begin_swap(&self) -> GateDecision {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let d = gate_decision(
            st.swapping,
            st.last_swap_done.map(|t| t.elapsed()),
            self.hold,
        );
        if d == GateDecision::Proceed {
            st.swapping = true;
        }
        d
    }

    /// Release the gate and arm the hold. Armed on failure too: a failing
    /// load attempt still stopped/restarted the server (GPU churn), so it
    /// must count against the thrash budget.
    fn finish_swap(&self) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.swapping = false;
        st.last_swap_done = Some(Instant::now());
    }
}

/// An OpenAI inference path whose request body's `model` field selects the model.
fn is_inference_path(path: &str) -> bool {
    let p = path.split('?').next().unwrap_or(path);
    p.ends_with("/chat/completions") || p.ends_with("/completions") || p.ends_with("/embeddings")
}

/// The bare path (without query string).
fn bare(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

/// Extract the requested model id from a JSON request body.
fn model_from_body(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(|s| s.to_string())
}

/// OpenAI `/v1/models` listing of every discoverable model (so a client sees the
/// whole library, not just the one loaded).
fn models_list(models_dir: &str) -> serde_json::Value {
    let models = model::discover(Path::new(models_dir)).unwrap_or_default();
    let data: Vec<_> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "owned_by": "llmtune",
            })
        })
        .collect();
    serde_json::json!({ "object": "list", "data": data })
}

fn json_header() -> tiny_http::Header {
    tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("static header is valid")
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "error": { "message": msg, "type": "llmtune_proxy" } }).to_string()
}

fn respond_json(req: tiny_http::Request, code: u16, body: &str) -> Result<()> {
    respond_json_retry(req, code, None, body)
}

/// Like [`respond_json`], with an optional `Retry-After` header (the gate's
/// hold/busy refusals tell well-behaved clients when to come back).
fn respond_json_retry(
    req: tiny_http::Request,
    code: u16,
    retry_after: Option<u64>,
    body: &str,
) -> Result<()> {
    let mut resp = tiny_http::Response::from_string(body)
        .with_status_code(code)
        .with_header(json_header());
    if let Some(secs) = retry_after {
        if let Ok(h) =
            tiny_http::Header::from_bytes(&b"Retry-After"[..], secs.to_string().as_bytes())
        {
            resp = resp.with_header(h);
        }
    }
    req.respond(resp)?;
    Ok(())
}

/// A proxy refusal: the status/Retry-After plus TWO messages - the `client`
/// body is generic (a proxy client must not learn internal paths, systemd
/// output, or backend addresses), while `log` carries the full detail for the
/// operator's proxy log.
struct Refuse {
    code: u16,
    retry_after: Option<u64>,
    client: String,
    log: String,
}

impl Refuse {
    fn new(code: u16, retry_after: Option<u64>, client: String, log: String) -> Self {
        Refuse {
            code,
            retry_after,
            client,
            log,
        }
    }
}

/// Ensure `requested` (an exact name or unique substring) is the served model,
/// swapping to it if not - through the swap gate, so concurrent swaps stay
/// serialized and rapid alternating requests can't thrash the GPU. On refusal
/// returns the [`Refuse`] to answer with.
#[allow(clippy::result_large_err)]
fn ensure_served(
    node: &Node,
    requested: &str,
    gate: &SwapGate,
) -> std::result::Result<String, Refuse> {
    let m = nodeops::resolve_model(node, requested).map_err(|e| {
        Refuse::new(
            503,
            None,
            // The requested name is the client's own input; the resolver's
            // detail (library dir, candidate list) stays server-side.
            format!("could not resolve model `{requested}`"),
            format!("could not resolve `{requested}`: {e}"),
        )
    })?;
    // The served model always passes - only a CHANGE goes through the gate.
    if llama::served_name(&node.llama_url).as_deref() == Some(m.name.as_str()) {
        return Ok(m.name);
    }
    match gate.begin_swap() {
        GateDecision::Busy => {
            let msg = "a model swap is already in progress - retry shortly".to_string();
            Err(Refuse::new(503, Some(2), msg.clone(), msg))
        }
        GateDecision::HoldFor(secs) => {
            let msg = format!(
                "swap-thrash guard: a model swap ran recently; retry in {secs}s \
                 (or request the currently served model)"
            );
            Err(Refuse::new(503, Some(secs), msg.clone(), msg))
        }
        GateDecision::Proceed => {
            let res = nodeops::load(node, &m.name);
            gate.finish_swap();
            let generic = format!("could not load `{}` - see the proxy log", m.name);
            match res {
                Ok((outcome, _used_default)) if outcome.ok => Ok(m.name),
                Ok((outcome, _)) => Err(Refuse::new(
                    503,
                    None,
                    generic,
                    format!("could not load `{}`: {}", m.name, outcome.detail),
                )),
                Err(e) => Err(Refuse::new(
                    503,
                    None,
                    generic,
                    format!("could not load `{}`: {e:#}", m.name),
                )),
            }
        }
    }
}

/// Forward `req`'s method/path/body to the backend and stream the response back
/// (status + content-type preserved). A non-2xx backend response is forwarded
/// transparently; only a transport failure becomes a 502.
fn proxy_to_backend(
    req: tiny_http::Request,
    method: &str,
    backend: &str,
    path: &str,
    body: &[u8],
    api_key: Option<&str>,
) -> Result<()> {
    let url = format!("{backend}{path}");
    // Generous read timeout: a long generation streams tokens over time.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(600))
        .build();
    let mut breq = agent
        .request(method, &url)
        .set("Content-Type", "application/json");
    // Forward the key to the backend (llama-server is launched with --api-key
    // whenever auth is on, so a keyless forward would 401).
    if let Some(key) = api_key {
        breq = breq.set("Authorization", &format!("Bearer {key}"));
    }
    let result = if body.is_empty() {
        breq.call()
    } else {
        breq.send_bytes(body)
    };
    let resp = match result {
        Ok(r) => r,
        // A non-2xx is still a real response - forward it as-is.
        Err(ureq::Error::Status(_code, r)) => r,
        Err(ureq::Error::Transport(t)) => {
            // Full transport detail (backend host:port, resolver errors) goes to
            // the operator's log only; the client gets a generic body.
            log_line(&format!("backend unreachable: {t}"));
            return respond_json(req, 502, &error_json("backend unreachable"));
        }
    };
    let status = resp.status();
    let ct = resp
        .header("Content-Type")
        .unwrap_or("application/json")
        .to_string();
    let ct_header = tiny_http::Header::from_bytes(&b"Content-Type"[..], ct.as_bytes())
        .unwrap_or_else(|_| json_header());
    let reader = resp.into_reader();
    // data_length None -> chunked, which streams SSE token-by-token.
    let response = tiny_http::Response::new(
        tiny_http::StatusCode(status),
        vec![ct_header],
        reader,
        None,
        None,
    );
    req.respond(response)?;
    Ok(())
}

/// Handle one request: serve `/v1/models` locally, ensure the named model is
/// loaded for an inference call, then proxy. Always responds.
fn handle(
    mut req: tiny_http::Request,
    node: &Node,
    backend: &str,
    api_key: Option<&str>,
    gate: &SwapGate,
) -> Result<()> {
    let method = req.method().as_str().to_string();
    let path = req.url().to_string();

    // Inbound auth: when a key is configured, every request must carry it. The
    // proxy is the front door to the same backend the key protects, so an
    // unauthenticated proxy would be a bypass of that key.
    if let Some(key) = api_key {
        if !authorized(&req, key) {
            return respond_json(
                req,
                401,
                &error_json("missing or invalid Authorization: Bearer <key>"),
            );
        }
    }

    // GET /v1/models - list the whole library from the local models dir.
    if method == "GET" && bare(&path).ends_with("/models") {
        return respond_json(req, 200, &models_list(&node.models_dir).to_string());
    }

    // Read the body once (needed both to pick the model and to forward), capped
    // so a large/slow POST can't exhaust memory.
    let mut body = Vec::new();
    if req
        .as_reader()
        .take(MAX_BODY_BYTES + 1)
        .read_to_end(&mut body)
        .is_err()
    {
        return respond_json(req, 400, &error_json("could not read request body"));
    }
    if body.len() as u64 > MAX_BODY_BYTES {
        return respond_json(req, 413, &error_json("request body too large"));
    }

    // For an inference call, swap to the requested model first (gated).
    if is_inference_path(&path) {
        if let Some(model) = model_from_body(&body) {
            match ensure_served(node, &model, gate) {
                Ok(name) => log_line(&format!("serving `{name}` for {method} {}", bare(&path))),
                Err(r) => {
                    log_line(&format!("refusing {method} {}: {}", bare(&path), r.log));
                    return respond_json_retry(req, r.code, r.retry_after, &error_json(&r.client));
                }
            }
        }
    }

    proxy_to_backend(req, &method, backend, &path, &body, api_key)
}

fn log_line(msg: &str) {
    println!("[proxy] {msg}");
}

/// Set SO_RCVTIMEO/SO_SNDTIMEO on the LISTENING socket. On Linux, accepted
/// sockets inherit both (the kernel copies `sk_rcvtimeo`/`sk_sndtimeo` when
/// cloning the socket on accept), which gives every client connection a
/// read/write deadline tiny_http has no API to set - a client that stalls
/// mid-upload or never reads a streaming response gets dropped instead of
/// pinning a worker forever. The inheritance is pinned by a unit test.
fn set_socket_timeouts(l: &std::net::TcpListener, t: Duration) -> Result<()> {
    use std::os::fd::AsRawFd;
    let tv = libc::timeval {
        tv_sec: t.as_secs() as libc::time_t,
        tv_usec: t.subsec_micros() as libc::suseconds_t,
    };
    for opt in [libc::SO_RCVTIMEO, libc::SO_SNDTIMEO] {
        // SAFETY: the fd is a valid open socket for the lifetime of `l`;
        // timeval is passed by pointer with its exact size.
        let rc = unsafe {
            libc::setsockopt(
                l.as_raw_fd(),
                libc::SOL_SOCKET,
                opt,
                &tv as *const libc::timeval as *const libc::c_void,
                std::mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        };
        if rc != 0 {
            return Err(anyhow!(
                "setting socket timeout: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(())
}

/// Run the proxy until interrupted.
pub fn serve(cfg: &Config, opts: &ProxyOpts) -> Result<()> {
    let node = cfg.local_node().clone();
    let backend = node.llama_url.trim_end_matches('/').to_string();
    let addr = format!("{}:{}", opts.host, opts.port);
    // The proxy enforces (and forwards) the same key the backend uses.
    let api_key = settings::api_key();
    let listener =
        std::net::TcpListener::bind(&addr).map_err(|e| anyhow!("could not bind {addr}: {e}"))?;
    set_socket_timeouts(&listener, SOCKET_TIMEOUT)?;
    let server = tiny_http::Server::from_listener(listener, None)
        .map_err(|e| anyhow!("could not serve on {addr}: {e}"))?;

    println!("llmtune swap-on-demand proxy");
    println!("  listening  http://{addr}");
    println!("  OpenAI     http://{addr}/v1   (point your harness here)");
    println!("  backend    {backend}");
    println!("  models dir {}", node.models_dir);
    println!(
        "  auth       {}",
        if api_key.is_some() {
            "Bearer key required"
        } else {
            "none (open)"
        }
    );
    if opts.host != "127.0.0.1" && opts.host != "localhost" && api_key.is_none() {
        println!(
            "  WARNING: bound to {} with NO auth - anyone on the network can reach \
             (and swap) your models. Set `llmtune endpoint auth on` or bind loopback.",
            opts.host
        );
    }
    println!(
        "  (swaps the served model on demand; {PROXY_WORKERS} workers, \
         {SOCKET_TIMEOUT:?} socket deadline, {:?} min hold between swaps)",
        opts.swap_hold
    );

    let server = Arc::new(server);
    let gate = Arc::new(SwapGate::new(opts.swap_hold));
    let node = Arc::new(node);
    let backend = Arc::new(backend);
    let api_key = Arc::new(api_key);
    let mut workers = Vec::new();
    for _ in 0..PROXY_WORKERS {
        let (server, gate, node, backend, api_key) = (
            server.clone(),
            gate.clone(),
            node.clone(),
            backend.clone(),
            api_key.clone(),
        );
        workers.push(std::thread::spawn(move || loop {
            let req = match server.recv() {
                Ok(r) => r,
                Err(_) => break, // listener closed
            };
            if let Err(e) = handle(req, &node, &backend, api_key.as_deref(), &gate) {
                // handle() consumed the request on every path it controls; this
                // only fires if responding itself failed (client hung up /
                // socket deadline hit) - just log it.
                log_line(&format!("request error: {e}"));
            }
        }));
    }
    for w in workers {
        let _ = w.join();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_matches_only_identical_bytes() {
        assert!(ct_eq(b"Bearer abc123", b"Bearer abc123"));
        assert!(!ct_eq(b"Bearer abc123", b"Bearer abc124"));
        assert!(!ct_eq(b"Bearer abc", b"Bearer abc123")); // length mismatch
        assert!(!ct_eq(b"", b"x"));
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn inference_paths_classified() {
        assert!(is_inference_path("/v1/chat/completions"));
        assert!(is_inference_path("/v1/completions?foo=1"));
        assert!(is_inference_path("/v1/embeddings"));
        assert!(!is_inference_path("/v1/models"));
        assert!(!is_inference_path("/health"));
    }

    #[test]
    fn model_extracted_from_body() {
        let b = br#"{"model":"Qwen3-30B","messages":[]}"#;
        assert_eq!(model_from_body(b).as_deref(), Some("Qwen3-30B"));
        // missing / non-string / invalid -> None
        assert_eq!(model_from_body(br#"{"messages":[]}"#), None);
        assert_eq!(model_from_body(b"not json"), None);
    }

    #[test]
    fn models_list_is_openai_shaped() {
        // empty/missing dir -> a valid empty list, never an error.
        let v = models_list("/nonexistent/llmtune/models");
        assert_eq!(v["object"], "list");
        assert!(v["data"].is_array());
        assert_eq!(v["data"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn bare_strips_query() {
        assert_eq!(bare("/v1/models?limit=5"), "/v1/models");
        assert_eq!(bare("/v1/chat/completions"), "/v1/chat/completions");
    }

    #[test]
    fn gate_decision_holds_then_releases() {
        let hold = Duration::from_secs(60);
        // no prior swap, none in flight -> proceed
        assert_eq!(gate_decision(false, None, hold), GateDecision::Proceed);
        // a swap in flight always wins (strict serialization)
        assert_eq!(
            gate_decision(true, Some(Duration::from_secs(999)), hold),
            GateDecision::Busy
        );
        // within the hold -> refused with the remaining time (ceiled, never 0)
        assert_eq!(
            gate_decision(false, Some(Duration::from_secs(10)), hold),
            GateDecision::HoldFor(50)
        );
        assert_eq!(
            gate_decision(false, Some(Duration::from_millis(59_800)), hold),
            GateDecision::HoldFor(1),
            "sub-second remainder must not become Retry-After: 0"
        );
        // past the hold -> proceed again
        assert_eq!(
            gate_decision(false, Some(Duration::from_secs(60)), hold),
            GateDecision::Proceed
        );
        // hold 0 = guard disabled (any elapsed passes)
        assert_eq!(
            gate_decision(false, Some(Duration::ZERO), Duration::ZERO),
            GateDecision::Proceed
        );
    }

    #[test]
    fn swap_gate_serializes_and_arms_hold() {
        // The thrash scenario: swap, then an immediate different-model request
        // must be held, and a concurrent swap must be Busy.
        let gate = SwapGate::new(Duration::from_secs(60));
        assert_eq!(gate.begin_swap(), GateDecision::Proceed);
        // while swapping, everyone else is Busy
        assert_eq!(gate.begin_swap(), GateDecision::Busy);
        gate.finish_swap();
        // hold armed: the next different-model request is refused with a delay
        match gate.begin_swap() {
            GateDecision::HoldFor(secs) => assert!((1..=60).contains(&secs), "{secs}"),
            other => panic!("expected HoldFor, got {other:?}"),
        }
        // a zero hold disables the guard entirely
        let free = SwapGate::new(Duration::ZERO);
        assert_eq!(free.begin_swap(), GateDecision::Proceed);
        free.finish_swap();
        assert_eq!(free.begin_swap(), GateDecision::Proceed);
    }

    #[test]
    fn socket_timeouts_inherited_by_accepted_connections() {
        // The #36 liveness mechanism: SO_RCVTIMEO/SO_SNDTIMEO set on the
        // listener must be inherited by accepted sockets (Linux copies them on
        // accept) - this is what bounds a stalled client read/write.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        set_socket_timeouts(&listener, Duration::from_secs(30)).unwrap();
        let addr = listener.local_addr().unwrap();
        let _client = std::net::TcpStream::connect(addr).unwrap();
        let (accepted, _) = listener.accept().unwrap();
        assert_eq!(
            accepted.read_timeout().unwrap(),
            Some(Duration::from_secs(30)),
            "accepted socket must inherit the read deadline"
        );
        assert_eq!(
            accepted.write_timeout().unwrap(),
            Some(Duration::from_secs(30)),
            "accepted socket must inherit the write deadline"
        );
    }
}
