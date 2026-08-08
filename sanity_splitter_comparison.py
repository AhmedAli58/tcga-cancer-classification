"""
Sanity check (methodology): StratifiedGroupKFold vs plain GroupKFold.

The probabl evaluate-ml-pipeline methodology says: for grouped data use
GroupKFold, and do NOT reach for Stratified* variants just because the
classes are imbalanced - stratification compresses across-fold variance and
gives over-confident error bars. nested_cv.py uses StratifiedGroupKFold
precisely because of imbalance, so this probe checks the concrete trade-off
on the real cohorts:

  - Does either splitter ever let a patient straddle train/test? (leakage)
  - What does each splitter do to per-fold test-set normal counts? (the
    reason stratification was chosen - and the risk of dropping it)

No models are fit; this only inspects fold geometry.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

RANDOM_STATE = 42
OUTER_FOLDS = 5
LABEL_COL = "Tissue Type"
ID_COL = "Sample ID"
MIN_COUNT = 10


def load(cancer_type, base_dir):
    expr = pd.read_csv(f"{base_dir}/{cancer_type}_geneexpression.csv", index_col=0)
    labels = pd.read_csv(f"{base_dir}/{cancer_type}_labels.csv")
    labels = labels[labels[ID_COL].isin(expr.index)].reset_index(drop=True)
    expr = expr.loc[labels[ID_COL].values]
    min_class = labels[LABEL_COL].value_counts().min()
    mask = (expr >= MIN_COUNT).sum(axis=0) >= min_class
    return expr.loc[:, mask], labels


def fold_geometry(name, splitter, X, y, groups, needs_y):
    print(f"\n  --- {name} ---")
    normal_counts, overlaps = [], []
    for i, (tr, te) in enumerate(
        splitter.split(X, y, groups=groups) if needs_y else splitter.split(X, y, groups=groups)
    ):
        n_norm = int((y.iloc[te] == 0).sum())
        n_tum = int((y.iloc[te] == 1).sum())
        ov = len(set(groups.iloc[tr]) & set(groups.iloc[te]))
        normal_counts.append(n_norm)
        overlaps.append(ov)
        print(f"    Fold {i+1}: test Tumor:Normal = {n_tum}:{n_norm}"
              f"  (normal fraction {n_norm/(n_tum+n_norm):.3f})  patient_overlap={ov}")
    print(f"    normals-per-fold: min={min(normal_counts)}, max={max(normal_counts)}, "
          f"std={np.std(normal_counts):.1f}   |  any zero-normal fold? "
          f"{'YES (breaks AUC/MCC)' if min(normal_counts)==0 else 'no'}   |  "
          f"max patient overlap across folds: {max(overlaps)}")
    return normal_counts


def run(cancer_type, base_dir):
    print("=" * 72)
    print(f"SPLITTER COMPARISON: {cancer_type}")
    print("=" * 72)
    X, labels = load(cancer_type, base_dir)
    groups = labels[ID_COL].str[:12]
    y = (labels[LABEL_COL] == "Tumor").astype(int)
    print(f"  overall Tumor:Normal = {int((y==1).sum())}:{int((y==0).sum())}  "
          f"({(y==0).mean()*100:.1f}% normal)")

    sgkf = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gkf = GroupKFold(n_splits=OUTER_FOLDS)
    fold_geometry("StratifiedGroupKFold (current code)", sgkf, X, y, groups, needs_y=True)
    fold_geometry("GroupKFold (methodology's recommendation)", gkf, X, y, groups, needs_y=False)


def main():
    run("TCGA_BRCA", "Data/TCGA_BRCA")
    run("TCGA_LUAD", "Data/TCGA_LUAD")


if __name__ == "__main__":
    main()
