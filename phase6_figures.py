"""
Phase 6: Generate report/presentation figures from saved outputs.

Reads the CSVs produced by the earlier phases and writes PNGs into figures/.
Every figure is optional-safe: if an input CSV is missing, that figure is
skipped with a note (so you can run it after a partial pipeline).

Inputs used (produce them by running phase2 -> phase3 -> phase4 -> phase5):
  Data/phase3_model_metrics.csv                         (phase3)
  Data/<COHORT>/<COHORT>_rf_importances.csv             (phase3)
  Data/<COHORT>/<COHORT>_labels_aligned.csv             (phase2)
  Data/<COHORT>/<COHORT>_cross_model_consensus.csv      (phase5)
  Data/phase4_cross_cancer_gene_comparison.csv          (phase4)

Outputs: figures/*.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "figures"
COHORTS = [("TCGA_BRCA", "BRCA"), ("TCGA_LUAD", "LUAD")]
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_performance():
    path = "Data/phase3_model_metrics.csv"
    if not os.path.exists(path):
        print("  [skip] performance figure - run phase3_train.py first"); return
    df = pd.read_csv(path)
    metrics = ["F1", "MCC", "AUC"]
    cohorts = df["cancer"].unique()
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 4.2), squeeze=False)
    for ax, cohort in zip(axes[0], cohorts):
        sub = df[df["cancer"] == cohort].set_index("model")[metrics]
        sub.plot(kind="bar", ax=ax, rot=20, width=0.75)
        ax.set_title(f"{cohort}: held-out test performance")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Model performance across classifiers", fontweight="bold")
    _save(fig, "fig_performance.png")


def fig_class_balance():
    frames = []
    for cohort, short in COHORTS:
        p = f"Data/{cohort}/{cohort}_labels_aligned.csv"
        if os.path.exists(p):
            vc = pd.read_csv(p)["Tissue Type"].value_counts()
            frames.append((short, vc.get("Tumor", 0), vc.get("Normal", 0)))
    if not frames:
        print("  [skip] class balance - no labels_aligned files"); return
    labels = [f[0] for f in frames]
    tumor = [f[1] for f in frames]
    normal = [f[2] for f in frames]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, tumor, label="Tumor")
    ax.bar(labels, normal, bottom=tumor, label="Normal")
    for i, (t, n) in enumerate(zip(tumor, normal)):
        ax.text(i, t + n + 5, f"{t}:{n}\n({t/max(n,1):.1f}:1)", ha="center", fontsize=9)
    ax.set_ylabel("samples")
    ax.set_title("Class imbalance per cohort\n(context for the over-classification problem)")
    ax.legend()
    _save(fig, "fig_class_balance.png")


def fig_top_genes():
    for cohort, short in COHORTS:
        p = f"Data/{cohort}/{cohort}_rf_importances.csv"
        if not os.path.exists(p):
            print(f"  [skip] top genes {short} - run phase3_train.py"); continue
        df = pd.read_csv(p).head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(df["gene"].astype(str), df["importance"])
        ax.set_title(f"{short}: top 15 genes (Random Forest importance)")
        ax.set_xlabel("importance")
        ax.tick_params(axis="y", labelsize=7)
        _save(fig, f"fig_top_genes_{short}.png")


def fig_consensus():
    for cohort, short in COHORTS:
        p = f"Data/{cohort}/{cohort}_cross_model_consensus.csv"
        if not os.path.exists(p):
            print(f"  [skip] consensus {short} - run phase5"); continue
        df = pd.read_csv(p)
        top = df.sort_values("n_models_topN", ascending=False).head(15).iloc[::-1]
        labels = top["symbol"].fillna("").replace("", pd.NA).fillna(top["gene_ensembl"])
        colors = ["#c0392b" if str(k) not in ("", "nan") else "#2c3e50"
                  for k in top["known_gene"]]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(labels.astype(str), top["n_models_topN"], color=colors)
        ax.set_xlabel("number of models with gene in top-30")
        ax.set_title(f"{short}: cross-model consensus drivers\n(red = known cancer gene)")
        ax.tick_params(axis="y", labelsize=8)
        _save(fig, f"fig_consensus_{short}.png")


def fig_cross_cancer():
    p = "Data/phase4_cross_cancer_gene_comparison.csv"
    if not os.path.exists(p):
        print("  [skip] cross-cancer - run phase4_compare.py"); return
    df = pd.read_csv(p)
    counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["#27ae60", "#2980b9", "#8e44ad"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha="center")
    ax.set_ylabel("number of driver genes")
    ax.set_title("BRCA vs LUAD driver genes\n(shared vs cancer-specific)")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "fig_cross_cancer.png")


def main():
    print("Generating figures ->", FIG_DIR)
    fig_performance()
    fig_class_balance()
    fig_top_genes()
    fig_consensus()
    fig_cross_cancer()
    print("Done.")


if __name__ == "__main__":
    main()
