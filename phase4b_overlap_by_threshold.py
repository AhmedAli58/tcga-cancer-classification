"""
Phase 4b: Cross-cancer gene-overlap analysis across ALL models and thresholds.

Additional, reproducible analysis. Does NOT modify or overwrite any existing
result. Extends the exact method of phase4_compare.py:
    1. For a given model, load BRCA and LUAD importance tables.
    2. Restrict BOTH to the shared gene universe (genes present in both
       cohorts' filtered gene sets) BEFORE ranking - identical to
       phase4_compare.py (shared_gene_universe = brca_genes & luad_genes;
       then .isin(shared_gene_universe) on each cohort).
    3. Take the top-N genes per cohort (by importance, descending) and count
       the intersection.

The only change vs phase4_compare.py is: this runs for all four models
(lr, svm, rf, xgb) and five thresholds (10, 30, 50, 100, 200), and writes a
tidy table. The per-model, per-threshold procedure is otherwise the same
shared-universe-then-top-N intersection.

Output: Data/phase4b_overlap_by_model_threshold.csv
"""

import pandas as pd

MODELS = {
    "Logistic Regression": "lr",
    "Linear SVM": "svm",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}
THRESHOLDS = [10, 30, 50, 100, 200]

BRCA_DIR = "Data/TCGA_BRCA"
LUAD_DIR = "Data/TCGA_LUAD"


def load_importances(cancer_type, base_dir, model_tag):
    """Load an importance table. Same path pattern as phase4_compare.py."""
    path = f"{base_dir}/{cancer_type}_{model_tag}_importances.csv"
    df = pd.read_csv(path)
    # Explicit sort (descending) so ranking does not depend on file order.
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def overlap_for_model(model_tag):
    brca = load_importances("TCGA_BRCA", BRCA_DIR, model_tag)
    luad = load_importances("TCGA_LUAD", LUAD_DIR, model_tag)

    brca_genes = set(brca["gene"])
    luad_genes = set(luad["gene"])
    shared_gene_universe = brca_genes & luad_genes  # identical to phase4_compare.py

    # Restrict to shared universe BEFORE taking top-N (identical to phase4).
    brca_shared = brca[brca["gene"].isin(shared_gene_universe)].reset_index(drop=True)
    luad_shared = luad[luad["gene"].isin(shared_gene_universe)].reset_index(drop=True)

    row = {"shared_universe_size": len(shared_gene_universe)}
    for n in THRESHOLDS:
        brca_top = set(brca_shared.head(n)["gene"])
        luad_top = set(luad_shared.head(n)["gene"])
        row[f"top_{n}"] = len(brca_top & luad_top)
    return row


def main():
    print("=" * 60)
    print("CROSS-CANCER GENE OVERLAP - ALL MODELS x ALL THRESHOLDS")
    print("(shared-gene-universe method, identical to phase4_compare.py)")
    print("=" * 60)

    rows = []
    for model_name, tag in MODELS.items():
        r = overlap_for_model(tag)
        rows.append({"model": model_name,
                     **{f"top_{n}": r[f"top_{n}"] for n in THRESHOLDS},
                     "shared_universe_size": r["shared_universe_size"]})
        print(f"{model_name:<22} " +
              " ".join(f"top{n}={r[f'top_{n}']}" for n in THRESHOLDS) +
              f"  (universe={r['shared_universe_size']})")

    out = pd.DataFrame(rows)[["model"] + [f"top_{n}" for n in THRESHOLDS] + ["shared_universe_size"]]
    out.to_csv("Data/phase4b_overlap_by_model_threshold.csv", index=False)
    print("\nSaved: Data/phase4b_overlap_by_model_threshold.csv")


if __name__ == "__main__":
    main()
