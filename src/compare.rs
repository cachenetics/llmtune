// SPDX-License-Identifier: GPL-2.0-only
//! Quant/throughput comparison from bench history - answers "is IQ2 worth it
//! over Q4" by grouping a node's benchmark records into model FAMILIES (the same
//! model at different quants/contexts) and showing the best throughput of each
//! variant side by side. The grouping + reduction are pure functions (tested);
//! the CLI just renders them.

use crate::history::Record;
use serde::Serialize;
use std::collections::BTreeMap;

/// One measured configuration of a family (a quant at a context), reduced to its
/// best run.
#[derive(Debug, Clone, Serialize)]
pub struct Variant {
    pub quant: String,
    pub ctx: u32,
    pub best_gen_tok_s: f64,
    pub best_prompt_tok_s: f64,
    /// The llama.cpp build the best run used, if recorded.
    pub build: Option<String>,
    /// How many runs are behind this best.
    pub runs: usize,
}

/// A model family (e.g. `Qwen3-30B-A3B`) and its measured variants, fastest first.
#[derive(Debug, Clone, Serialize)]
pub struct Family {
    pub key: String,
    pub variants: Vec<Variant>,
}

/// The family key for a model: its name without the extension and quant token, so
/// `Qwen3-30B-A3B-IQ2_XXS.gguf` and `Qwen3-30B-A3B-Q4_K_M.gguf` share a family.
pub fn family_key(model: &str, quant: &Option<String>) -> String {
    let mut s = model.strip_suffix(".gguf").unwrap_or(model).to_string();
    if let Some(q) = quant {
        // Drop the quant token wherever it sits (with or without a separator).
        for pat in [format!("-{q}"), format!(".{q}"), format!("_{q}"), q.clone()] {
            s = s.replace(&pat, "");
        }
    }
    s.trim_matches(|c| c == '-' || c == '_' || c == '.' || c == ' ')
        .to_string()
}

/// Group records into families, each variant reduced to its best run. Families
/// are sorted by key; variants within a family by gen tok/s (fastest first).
pub fn group(records: &[Record]) -> Vec<Family> {
    // family -> (quant, ctx) -> accumulated best
    let mut fam: BTreeMap<String, BTreeMap<(String, u32), Variant>> = BTreeMap::new();
    for r in records {
        // Ignore fantasy records so a bogus tok/s can't become a family's best -
        // see PerfBench::is_plausible.
        if !r.perf.is_plausible() {
            continue;
        }
        let key = family_key(&r.model, &r.quant);
        let quant = r.quant.clone().unwrap_or_else(|| "?".to_string());
        let vk = (quant.clone(), r.ctx);
        let entry = fam.entry(key).or_default().entry(vk).or_insert(Variant {
            quant,
            ctx: r.ctx,
            best_gen_tok_s: f64::MIN,
            best_prompt_tok_s: 0.0,
            build: None,
            runs: 0,
        });
        entry.runs += 1;
        if r.perf.gen_tok_s > entry.best_gen_tok_s {
            entry.best_gen_tok_s = r.perf.gen_tok_s;
            entry.best_prompt_tok_s = r.perf.prompt_tok_s;
            entry.build = r.build.clone();
        }
    }
    let mut families: Vec<Family> = fam
        .into_iter()
        .map(|(key, variants)| {
            let mut variants: Vec<Variant> = variants.into_values().collect();
            variants.sort_by(|a, b| {
                b.best_gen_tok_s
                    .partial_cmp(&a.best_gen_tok_s)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Family { key, variants }
        })
        .collect();
    families.sort_by(|a, b| a.key.cmp(&b.key));
    families
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::PerfBench;

    fn rec(model: &str, quant: &str, ctx: u32, gen: f64) -> Record {
        Record {
            ts: 0,
            node: "n".into(),
            model: model.into(),
            arch: "qwen3moe".into(),
            quant: Some(quant.into()),
            ctx,
            profile: "qwen35moe".into(),
            perf: PerfBench {
                gen_tok_s: gen,
                prompt_tok_s: gen * 10.0,
                n_gen: 128,
                n_prompt: 512,
                ..Default::default()
            },
            notes: String::new(),
            build: None,
            cluster: None,
            cluster_members: vec![],
        }
    }

    #[test]
    fn family_key_strips_quant_and_ext() {
        assert_eq!(
            family_key("Qwen3-30B-A3B-IQ2_XXS.gguf", &Some("IQ2_XXS".into())),
            "Qwen3-30B-A3B"
        );
        assert_eq!(
            family_key("gemma-9B-Q4_K_M.gguf", &Some("Q4_K_M".into())),
            "gemma-9B"
        );
        // no quant known -> just drop the extension
        assert_eq!(family_key("mystery.gguf", &None), "mystery");
    }

    #[test]
    fn group_reduces_to_best_and_sorts_fastest_first() {
        let recs = vec![
            rec("Qwen3-30B-A3B-IQ2_XXS.gguf", "IQ2_XXS", 32768, 18.0),
            rec("Qwen3-30B-A3B-IQ2_XXS.gguf", "IQ2_XXS", 32768, 20.0), // best for IQ2
            rec("Qwen3-30B-A3B-Q4_K_M.gguf", "Q4_K_M", 32768, 12.0),
        ];
        let fams = group(&recs);
        assert_eq!(fams.len(), 1);
        let f = &fams[0];
        assert_eq!(f.key, "Qwen3-30B-A3B");
        assert_eq!(f.variants.len(), 2);
        // fastest variant first
        assert_eq!(f.variants[0].quant, "IQ2_XXS");
        assert!((f.variants[0].best_gen_tok_s - 20.0).abs() < 1e-9);
        assert_eq!(f.variants[0].runs, 2);
        assert_eq!(f.variants[1].quant, "Q4_K_M");
    }

    #[test]
    fn distinct_ctx_are_separate_variants() {
        let recs = vec![
            rec("m-Q4_K_M.gguf", "Q4_K_M", 32768, 12.0),
            rec("m-Q4_K_M.gguf", "Q4_K_M", 98304, 9.0),
        ];
        let fams = group(&recs);
        assert_eq!(
            fams[0].variants.len(),
            2,
            "same quant, different ctx -> two rows"
        );
    }
}
