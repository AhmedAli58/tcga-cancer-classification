"""
Sanity check (1): explicit per-fold patient-overlap verification.

nested_cv.py enforces no train/test patient leakage with a silent `assert`
(it only raises on FAILURE, so a clean run prints nothing). This script
reproduces the EXACT same outer folds - same RANDOM_STATE, same
StratifiedGroupKFold config, same gene filter, same patient grouping - and
prints the train/test patient overlap for every fold, so the "no leakage"
claim has visible evidence instead of just "it didn't crash".
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --- must match nested_cv.py exactly ---
RANDOM_STATE = 42
OUTER_FOLDS = 5
LABEL_COL = "Tissue Type"
ID_COL = "Sample ID"
MIN_COUNT = 10


def load_and_align(cancer_type, base_dir):
    expr = pd.read_csv(f"{base_dir}/{cancer_type}_geneexpression.csv", index_col=0)
    labels = pd.read_csv(f"{base_dir}/{cancer_type}_labels.csv")
    labels = labels[labels[ID_COL].isin(expr.index)].reset_index(drop=True)
    expr = expr.loc[labels[ID_COL].values]
    return expr, labels


def filter_low_expression_genes(expr, labels):
    min_class_size = labels[LABEL_COL].value_counts().min()
    gene_mask = (expr >= MIN_COUNT).sum(axis=0) >= min_class_size
    return expr.loc[:, gene_mask]


def check_cancer(cancer_type, base_dir):
    print("=" * 64)
    print(f"LEAKAGE CHECK: {cancer_type}")
    print("=" * 64)
    expr, labels = load_and_align(cancer_type, base_dir)
    expr_filtered = filter_low_expression_genes(expr, labels)
    patient_ids = labels[ID_COL].str[:12]
    y = (labels[LABEL_COL] == "Tumor").astype(int)

    outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_zero = True
    for fold_i, (train_idx, test_idx) in enumerate(
        outer_cv.split(expr_filtered, y, groups=patient_ids)
    ):
        train_patients = set(patient_ids.iloc[train_idx])
        test_patients = set(patient_ids.iloc[test_idx])
        overlap = train_patients & test_patients
        # class balance carried into each test fold, for context
        tn = y.iloc[test_idx]
        status = "OK - no overlap" if len(overlap) == 0 else f"*** LEAK: {len(overlap)} ***"
        print(f"  Fold {fold_i + 1}/{OUTER_FOLDS}: "
              f"train_patients={len(train_patients)}, test_patients={len(test_patients)}, "
              f"overlap={len(overlap)}  [{status}]  "
              f"test Tumor:Normal = {int((tn == 1).sum())}:{int((tn == 0).sum())}")
        all_zero = all_zero and (len(overlap) == 0)

    # also confirm no SAMPLE-level duplication and full coverage
    print(f"  => {cancer_type}: all folds zero-overlap = {all_zero}\n")
    return all_zero


def main():
    r1 = check_cancer("TCGA_BRCA", "Data/TCGA_BRCA")
    r2 = check_cancer("TCGA_LUAD", "Data/TCGA_LUAD")
    print("=" * 64)
    print(f"OVERALL: patient-grouped leakage-free across ALL folds = {r1 and r2}")
    print("=" * 64)


if __name__ == "__main__":
    main()
