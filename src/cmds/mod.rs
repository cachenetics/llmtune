// SPDX-License-Identifier: GPL-2.0-only
//! `cmd_*` handlers grouped by command family - the render layer over the ops
//! modules (nodeops/transport/cluster/build/profile). Split out of main.rs.

pub mod build;
pub mod cluster;
pub mod fleet;
pub mod misc;
pub mod models;
pub mod node;
pub mod profile;
