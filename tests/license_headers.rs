// SPDX-License-Identifier: GPL-2.0-only
//! Release-hygiene gate: every Rust source file carries the project SPDX
//! license header, and the manifest license matches, so a stray file cannot
//! ship mislicensed.

use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn every_source_file_has_the_gpl2_spdx_header() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&src, &mut files);
    assert!(
        files.len() >= 45,
        "expected the full source tree, found {} files",
        files.len()
    );
    let bad: Vec<String> = files
        .iter()
        .filter(|f| {
            !std::fs::read_to_string(f)
                .unwrap()
                .starts_with("// SPDX-License-Identifier: GPL-2.0-only")
        })
        .map(|f| f.display().to_string())
        .collect();
    assert!(
        bad.is_empty(),
        "files missing the GPL-2.0-only SPDX header: {bad:?}"
    );
}

#[test]
fn manifest_and_docs_license_is_gpl2_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("license = \"GPL-2.0-only\""),
        "Cargo.toml license field must be GPL-2.0-only"
    );
    let license = std::fs::read_to_string(root.join("LICENSE")).unwrap();
    assert!(
        license.contains("GNU GENERAL PUBLIC LICENSE") && license.contains("Version 2, June 1991"),
        "LICENSE must be the GPLv2 text"
    );
    for f in ["README.md", "SPEC.md", "NOTICE"] {
        let body = std::fs::read_to_string(root.join(f)).unwrap();
        assert!(
            !body.contains("Apache-2.0") && !body.contains("Apache License"),
            "{f} still references the Apache license"
        );
    }
}

/// The TLS linking exception (issue #43) must travel with the code: the grant
/// file exists, uses the recognized special-exception form, names the TLS
/// stack, and LICENSE/README/NOTICE all point at it.
#[test]
fn tls_linking_exception_is_granted_and_referenced() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let grant = std::fs::read_to_string(root.join("COPYING.LINKING-EXCEPTION")).unwrap();
    assert!(
        grant.contains("In addition, as a special exception"),
        "grant must use the recognized GPL special-exception form"
    );
    for lib in ["ring", "rustls", "ureq"] {
        assert!(grant.contains(lib), "grant must name the {lib} library");
    }
    assert!(
        grant.contains("obey the GNU General Public License in all respects"),
        "grant must keep the GPL binding for the rest of the code"
    );
    for f in ["LICENSE", "README.md", "NOTICE"] {
        let body = std::fs::read_to_string(root.join(f)).unwrap();
        assert!(
            body.contains("COPYING.LINKING-EXCEPTION"),
            "{f} must reference COPYING.LINKING-EXCEPTION"
        );
    }
}
