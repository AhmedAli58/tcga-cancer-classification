"""
Phase 6: Generate the report figures from saved outputs.

Reads the CSVs produced by the earlier phases and writes PNGs into figures/.
Each figure is optional-safe: if an input is missing, that figure is skipped
with a note.

Inputs used:
  Data/phase3_model_metrics.csv                (phase3)  -> fig_performance
  Data/<COHORT>/<COHORT>_rf_importances.csv    (phase3)  -> fig_cross_cancer

Outputs: figures/fig_performance.png, figures/fig_cross_cancer.png
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


def fig_cross_cancer():
    # Compute counts directly from the RF importance files using the SAME method
    # as phase4b_overlap_by_threshold.py: explicit sort by importance (descending),
    # restrict to the shared gene universe, then take top-30 per cohort.
    TOP_N = 30
    brca_p = "Data/TCGA_BRCA/TCGA_BRCA_rf_importances.csv"
    luad_p = "Data/TCGA_LUAD/TCGA_LUAD_rf_importances.csv"
    if not (os.path.exists(brca_p) and os.path.exists(luad_p)):
        print("  [skip] cross-cancer - run phase3_train.py"); return

    brca = pd.read_csv(brca_p).sort_values("importance", ascending=False).reset_index(drop=True)
    luad = pd.read_csv(luad_p).sort_values("importance", ascending=False).reset_index(drop=True)

    shared_universe = set(brca["gene"]) & set(luad["gene"])
    brca_shared = brca[brca["gene"].isin(shared_universe)].reset_index(drop=True)
    luad_shared = luad[luad["gene"].isin(shared_universe)].reset_index(drop=True)

    brca_top = set(brca_shared.head(TOP_N)["gene"])
    luad_top = set(luad_shared.head(TOP_N)["gene"])

    n_shared = len(brca_top & luad_top)
    n_brca_only = len(brca_top - luad_top)
    n_luad_only = len(luad_top - brca_top)

    labels = ["Shared", "BRCA-specific", "LUAD-specific"]
    values = [n_shared, n_brca_only, n_luad_only]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["#27ae60", "#2980b9", "#8e44ad"])
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center")
    ax.set_ylabel("number of predictive features")
    ax.set_title("Top-30 Random Forest predictive features: BRCA vs LUAD")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "fig_cross_cancer.png")


def main():
    print("Generating figures ->", FIG_DIR)
    # Only the two figures used in the final report are generated here.
    fig_performance()
    fig_cross_cancer()
    print("Done.")


if __name__ == "__main__":
    main()
