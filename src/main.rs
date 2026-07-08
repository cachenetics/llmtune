// SPDX-License-Identifier: GPL-2.0-only
//! llmtune - public inference engine for the AMD BC-250.
//! CLI-first engine; the TUI (M3) is a view over the same operations.

// Nested `if let ... { if cond }` reads clearer than a collapsed let-chain here
// and let-chains aren't stable on this edition.
#![allow(clippy::collapsible_if, clippy::collapsible_match)]

mod agentic;
mod bench;
mod build;
mod cli;
mod cluster;
mod cmds;
mod compare;
mod config;
mod doctor;
mod endpoint;
mod fmt;
mod history;
mod identity;
mod library;
mod llama;
mod lock;
mod mem;
mod model;
mod netboot;
mod netboot_image;
mod netboot_node;
mod netboot_server;
mod nodeops;
mod paths;
mod profile;
mod proxy;
mod settings;
mod setup;
mod swap;
mod telemetry;
mod transport;
mod ui;

fn main() {
    // Die quietly on a closed pipe (`llmtune --json ... | head`) instead of
    // Rust's default "failed printing to stdout" panic - agents pipe us.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    // cli::run renders failures itself (JSON object on stderr under --json,
    // the classic `error:` line otherwise) and returns the exit code under
    // the agentic convention: 0 ok, 1 error, 2 guard refusal (src/agentic.rs).
    std::process::exit(cli::run());
}
