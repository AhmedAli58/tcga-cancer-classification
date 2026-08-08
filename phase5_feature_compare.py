"""
Phase 5: Cross-MODEL feature-importance comparison + known-gene annotation.

Where Phase 4 compares driver genes ACROSS CANCERS (BRCA vs LUAD), Phase 5
compares them ACROSS MODELS (LR, SVM, RF, XGB) within each cancer, then:

  1. Ranks each model's importance table and takes its top-N genes.
  2. Finds CONSENSUS driver genes = genes in the top-N of MULTIPLE models
     (a gene flagged important by several independent model families is far
     more trustworthy than one flagged by a single model).
  3. Maps Ensembl gene IDs -> HGNC symbols (via mygene, if available).
  4. Flags KNOWN cancer-relevant genes using a curated marker list, so the
     report can say "these consensus drivers are established BRCA/LUAD genes,
     which validates the pipeline."

Run AFTER phase3_train.py (needs the *_{lr,svm,rf,xgb}_importances.csv files).
Gracefully uses whichever model importance files exist.
"""

import os
import glob
import pandas as pd

TOP_N = 30          # top genes per model to treat as "important"
MODEL_TAGS = ["lr", "svm", "rf", "xgb"]
MODEL_LABEL = {"lr": "LogReg", "svm": "SVM", "rf": "RandomForest", "xgb": "XGBoost"}

# --- Known-gene annotation, WITH an explicit provenance/citation per gene.
# Two tiers, so every annotation states where the claim comes from:
#
#   Tier 1 (NOTEBOOK): gene symbols that are actually named as biomarkers in
#   the user's NotebookLM source papers. Queried directly from the notebook
#   (2026-07-13). The only source in that collection that enumerates specific
#   biomarker gene symbols is Dalmolin et al., which used XGBoost + SHAP to
#   isolate genes per TCGA cancer type. That query surfaced exactly:
#       HKDC1, LMX1B  -> BRCA   NAPSA -> LUAD
#   These are the sourced-from-the-papers annotations.
#
#   Tier 2 (EXTERNAL): canonical clinical markers that are NOT named in the
#   notebook sources. Kept only so the annotation still has recall, but each
#   is explicitly cited as "external reference" so it is never mistaken for a
#   notebook-sourced claim.
#
# Structure: symbol -> {"cancer": <BRCA|LUAD|pan-cancer>, "source": <citation>}.
# Used only to ANNOTATE, never to filter results.
NOTEBOOK_SRC = ("Dalmolin et al., 'Feature Selection in Cancer Classification' "
                "(NotebookLM source; XGBoost+SHAP biomarker)")
EXTERNAL_SRC = "Canonical clinical marker (external reference; not named in NotebookLM sources)"

KNOWN_GENES = {
    # Tier 1 - sourced from the NotebookLM papers
    "HKDC1": {"cancer": "BRCA", "source": NOTEBOOK_SRC},
    "LMX1B": {"cancer": "BRCA", "source": NOTEBOOK_SRC},
    "NAPSA": {"cancer": "LUAD", "source": NOTEBOOK_SRC},
    # Tier 2 - canonical BRCA markers (external reference)
    "ESR1": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "PGR": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "ERBB2": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "GATA3": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "FOXA1": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "TFF1": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "TFF3": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "XBP1": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "CDH1": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "BRCA1": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "BRCA2": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "MLPH": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "AGR2": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "GREB1": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    "AR": {"cancer": "BRCA", "source": EXTERNAL_SRC},
    # Tier 2 - canonical LUAD markers (external reference)
    "NKX2-1": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "TTF1": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "SFTPC": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "SFTPB": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "SFTPA1": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "SFTPA2": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "SFTA3": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "SCGB1A1": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "EGFR": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "KRAS": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "ALK": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "ROS1": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "MET": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    "MUC1": {"cancer": "LUAD", "source": EXTERNAL_SRC},
    # Tier 2 - pan-cancer proliferation markers (external reference)
    "MKI67": {"cancer": "pan-cancer", "source": EXTERNAL_SRC},
    "TOP2A": {"cancer": "pan-cancer", "source": EXTERNAL_SRC},
    "CCNB1": {"cancer": "pan-cancer", "source": EXTERNAL_SRC},
    "CCNB2": {"cancer": "pan-cancer", "source": EXTERNAL_SRC},
    "AURKA": {"cancer": "pan-cancer", "source": EXTERNAL_SRC},
    "BIRC5": {"cancer": "pan-cancer", "source": EXTERNAL_SRC},
    "PCNA": {"cancer": "pan-cancer", "source": EXTERNAL_SRC},
}

try:
    import mygene
    _MG = mygene.MyGeneInfo()
    HAS_MYGENE = True
except Exception:
    HAS_MYGENE = False
    print("NOTE: mygene not installed -> gene symbols will be blank. "
          "Run: pip3 install mygene")


def strip_version(ensembl_id: str) -> str:
    return ensembl_id.split(".")[0]


def map_symbols(ensembl_ids):
    """Return {ensembl_id_no_version: symbol}. Empty if mygene unavailable."""
    if not HAS_MYGENE or not ensembl_ids:
        return {}
    clean = list({strip_version(g) for g in ensembl_ids})
    out = {}
    try:
        res = _MG.querymany(clean, scopes="ensembl.gene", fields="symbol",
                            species="human", verbose=False)
        for r in res:
            if "symbol" in r:
                out[r["query"]] = r["symbol"]
    except Exception as e:
        print(f"  mygene lookup failed ({e}); continuing without symbols.")
    return out


def load_model_topN(cancer_type, base_dir):
    """Return {model_tag: [top-N ensembl ids]} for models whose files exist."""
    topn = {}
    for tag in MODEL_TAGS:
        path = f"{base_dir}/{cancer_type}_{tag}_importances.csv"
        if os.path.exists(path):
            df = pd.read_csv(path).sort_values("importance", ascending=False)
            topn[tag] = df.head(TOP_N)["gene"].tolist()
    return topn


def is_known(symbol, cancer_short):
    """Return (tag, source_citation) for a gene symbol.

    tag is '' if unknown, else e.g. 'BRCA', 'pan-cancer', or 'LUAD+pan-cancer'.
    source_citation states WHERE the annotation comes from (a NotebookLM source
    paper vs. an external canonical marker), so no annotation is uncited.
    """
    if not symbol or symbol not in KNOWN_GENES:
        return "", ""
    entry = KNOWN_GENES[symbol]
    cancer = entry["cancer"]
    # only flag a cancer-specific marker for the matching cohort; pan-cancer always applies
    if cancer == "pan-cancer":
        tag = "pan-cancer"
    elif cancer == cancer_short:
        tag = cancer_short
    else:
        return "", ""  # e.g. a LUAD marker showing up in the BRCA table - don't claim it
    return tag, entry["source"]


def consensus_for_cancer(cancer_type, base_dir, cancer_short):
    print("\n" + "=" * 60)
    print(f"CROSS-MODEL CONSENSUS: {cancer_type}")
    print("=" * 60)

    topn = load_model_topN(cancer_type, base_dir)
    if not topn:
        print(f"  No importance files found in {base_dir}. Run phase3_train.py first.")
        return None
    print(f"  Models available: {[MODEL_LABEL[t] for t in topn]}")

    # Count in how many models' top-N each gene appears
    gene_hits = {}
    gene_models = {}
    for tag, genes in topn.items():
        for g in genes:
            gene_hits[g] = gene_hits.get(g, 0) + 1
            gene_models.setdefault(g, []).append(MODEL_LABEL[tag])

    n_models = len(topn)
    symbols = map_symbols(list(gene_hits.keys()))

    rows = []
    for g, hits in gene_hits.items():
        sym = symbols.get(strip_version(g), "")
        tag, source = is_known(sym, cancer_short)
        rows.append({
            "gene_ensembl": g,
            "symbol": sym,
            "n_models_topN": hits,
            "models": ", ".join(sorted(gene_models[g])),
            "known_gene": tag,
            "known_gene_source": source,
        })
    df = pd.DataFrame(rows).sort_values(
        ["n_models_topN", "symbol"], ascending=[False, True]
    ).reset_index(drop=True)

    out_path = f"{base_dir}/{cancer_type}_cross_model_consensus.csv"
    df.to_csv(out_path, index=False)

    full_consensus = df[df["n_models_topN"] == n_models]
    strong = df[df["n_models_topN"] >= max(2, n_models - 1)]
    known_hits = df[df["known_gene"] != ""]

    print(f"  Genes in ALL {n_models} models' top-{TOP_N} (full consensus): {len(full_consensus)}")
    for _, r in full_consensus.iterrows():
        print(f"    {r['symbol'] or r['gene_ensembl']}  "
              f"[{r['known_gene'] or 'novel/other'}]  ({r['models']})")
    print(f"  Genes in >= {max(2, n_models - 1)} models: {len(strong)}")
    print(f"  Consensus genes that are KNOWN markers: {len(known_hits)}")
    for _, r in known_hits.sort_values("n_models_topN", ascending=False).iterrows():
        print(f"    {r['symbol']} [{r['known_gene']}] in {r['n_models_topN']} models "
              f"({r['models']})  <- {r['known_gene_source']}")
    print(f"  Saved: {out_path}")

    return df


def main():
    brca = consensus_for_cancer("TCGA_BRCA", "Data/TCGA_BRCA", "BRCA")
    luad = consensus_for_cancer("TCGA_LUAD", "Data/TCGA_LUAD", "LUAD")

    # Cross-cancer view of the consensus drivers (ties into Phase 4)
    if brca is not None and luad is not None:
        b = set(brca[brca["n_models_topN"] >= 2]["symbol"]) - {""}
        l = set(luad[luad["n_models_topN"] >= 2]["symbol"]) - {""}
        print("\n" + "=" * 60)
        print("CONSENSUS DRIVERS: SHARED vs CANCER-SPECIFIC (symbols)")
        print("=" * 60)
        print(f"  Shared (consensus in BOTH cancers): {sorted(b & l)}")
        print(f"  BRCA-only consensus: {sorted(b - l)}")
        print(f"  LUAD-only consensus: {sorted(l - b)}")


if __name__ == "__main__":
    main()
