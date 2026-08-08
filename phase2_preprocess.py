"""
Phase 2: Preprocessing pipeline for TCGA BRCA and LUAD RNA-seq data.

Pipeline order (per frozen literature review + supervisor guidance):
    1. Load raw counts, align samples to labels
    2. Filter low-expression genes (global step, low leakage risk per literature)
    3. Check skewness of raw data to justify log2 transform
    4. Patient-grouped, stratified train/test split
       - stratified: preserves Tumor:Normal class ratio in both partitions
       - patient-grouped: prevents the same patient's tumor + matched normal
         sample from appearing in both train and test (TCGA-specific leakage
         risk - patient barcode is the first 12 characters of Sample ID)
    5. Normalize: library-size normalize (train stats only) -> log2(x+1)
    6. Save train/test splits separately per cancer type

Leakage rules enforced:
    - No patient appears in both train and test (verified with an explicit
      overlap check printed at runtime).
    - Any statistic used for normalization (library size, median) is computed
      on the TRAINING split only, then applied to both train and test.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42
TEST_SIZE = 0.20
LABEL_COL = "Tissue Type"
ID_COL = "Sample ID"


def load_and_align(cancer_type: str, base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load expression + labels for a cancer type and align samples between them."""
    expr = pd.read_csv(f"{base_dir}/{cancer_type}_geneexpression.csv", index_col=0)
    labels = pd.read_csv(f"{base_dir}/{cancer_type}_labels.csv")

    labels = labels[labels[ID_COL].isin(expr.index)].reset_index(drop=True)
    expr = expr.loc[labels[ID_COL].values]

    print(f"{cancer_type}: expression {expr.shape}, labels {labels.shape}")
    return expr, labels


def check_skewness(expr: pd.DataFrame, cancer_type: str) -> float:
    """
    Report skewness of raw expression values to verify (not assume) that a
    log2 transform is justified. Positive skew = long right tail, which log2
    is meant to correct for.
    """
    skew_val = skew(expr.values.flatten())
    print(f"{cancer_type}: raw expression skewness = {skew_val:.3f} "
          f"({'right-skewed, log2 justified' if skew_val > 1 else 'check manually'})")
    return skew_val


def filter_low_expression_genes(expr: pd.DataFrame, labels: pd.DataFrame,
                                 cancer_type: str, min_count: int = 10) -> pd.DataFrame:
    """
    Keep genes with >= min_count reads in at least as many samples as the
    smaller class. Global step applied before the split - treated as data
    cleaning rather than a learned/label-informed parameter, per the frozen
    literature review.
    """
    min_class_size = labels[LABEL_COL].value_counts().min()
    gene_mask = (expr >= min_count).sum(axis=0) >= min_class_size
    expr_filtered = expr.loc[:, gene_mask]

    print(f"{cancer_type}: genes {expr.shape[1]} -> {expr_filtered.shape[1]} "
          f"({expr.shape[1] - expr_filtered.shape[1]} removed)")
    return expr_filtered


def split_data(expr: pd.DataFrame, labels: pd.DataFrame, cancer_type: str):
    """
    Patient-grouped, stratified train/test split.

    - Stratified: preserves the Tumor:Normal class ratio in both train and test.
    - Patient-grouped: the same patient's samples (e.g. a matched tumor +
      normal pair) never appear split across both train and test. TCGA
      patient ID is the first 12 characters of the Sample ID barcode.

    Both constraints are enforced together via StratifiedGroupKFold, and the
    result is verified with an explicit zero-overlap check.
    """
    patient_ids = labels[ID_COL].str[:12]
    y = labels[LABEL_COL]

    n_splits = round(1 / TEST_SIZE)  # e.g. TEST_SIZE=0.20 -> 5 folds -> ~80/20
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(expr, y, groups=patient_ids))

    X_train, X_test = expr.iloc[train_idx], expr.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    print(f"{cancer_type}: split into train {X_train.shape[0]} / test {X_test.shape[0]} "
          f"(patient-grouped, stratified)")
    print(f"  Train class distribution:\n{y_train[LABEL_COL].value_counts().to_string()}")
    print(f"  Test class distribution:\n{y_test[LABEL_COL].value_counts().to_string()}")

    # Explicit leakage verification - must be 0
    train_patients = set(patient_ids.iloc[train_idx])
    test_patients = set(patient_ids.iloc[test_idx])
    overlap = train_patients & test_patients
    print(f"  Patient overlap check: {len(overlap)} (should be 0)")
    if len(overlap) > 0:
        raise ValueError(
            f"{cancer_type}: {len(overlap)} patients appear in BOTH train and test. "
            f"Split is invalid - do not proceed to normalization or model training."
        )

    return X_train, X_test, y_train, y_test


def normalize(X_train: pd.DataFrame, X_test: pd.DataFrame, cancer_type: str):
    """
    Library-size normalize, then log2(x+1).

    Library sizes are computed from RAW counts for each sample individually.
    The MEDIAN library size used to rescale both train and test is computed
    on the TRAINING set only, to avoid leaking test-set information into the
    normalization applied to training data. This same rule applies whether
    using a single split (as here) or cross-validation (fit within each fold).
    """
    train_lib_sizes = X_train.sum(axis=1)
    test_lib_sizes = X_test.sum(axis=1)
    median_lib_size = train_lib_sizes.median()  # train-only statistic

    X_train_norm = X_train.div(train_lib_sizes, axis=0) * median_lib_size
    X_test_norm = X_test.div(test_lib_sizes, axis=0) * median_lib_size  # uses TRAIN median

    X_train_norm = np.log2(X_train_norm + 1)
    X_test_norm = np.log2(X_test_norm + 1)

    print(f"{cancer_type}: normalized using train median library size = {median_lib_size:.0f}")

    missing_train = X_train_norm.isnull().sum().sum()
    missing_test = X_test_norm.isnull().sum().sum()
    print(f"{cancer_type}: missing values after normalization - train: {missing_train}, test: {missing_test}")

    return X_train_norm, X_test_norm


def save_outputs(X_train, X_test, y_train, y_test, cancer_type: str, base_dir: str):
    X_train.to_csv(f"{base_dir}/{cancer_type}_X_train.csv")
    X_test.to_csv(f"{base_dir}/{cancer_type}_X_test.csv")
    y_train.to_csv(f"{base_dir}/{cancer_type}_y_train.csv", index=False)
    y_test.to_csv(f"{base_dir}/{cancer_type}_y_test.csv", index=False)
    print(f"{cancer_type}: saved train/test splits to {base_dir}/")


def preprocess_cancer_type(cancer_type: str, base_dir: str):
    print("\n" + "=" * 60)
    print(f"PROCESSING {cancer_type}")
    print("=" * 60)

    expr, labels = load_and_align(cancer_type, base_dir)
    check_skewness(expr, cancer_type)
    expr_filtered = filter_low_expression_genes(expr, labels, cancer_type)
    X_train, X_test, y_train, y_test = split_data(expr_filtered, labels, cancer_type)
    X_train_norm, X_test_norm = normalize(X_train, X_test, cancer_type)
    save_outputs(X_train_norm, X_test_norm, y_train, y_test, cancer_type, base_dir)

    print(f"{cancer_type} PREPROCESSING COMPLETE")
    print(f"  Final train shape: {X_train_norm.shape}, test shape: {X_test_norm.shape}")


def main():
    preprocess_cancer_type("TCGA_BRCA", "Data/TCGA_BRCA")
    preprocess_cancer_type("TCGA_LUAD", "Data/TCGA_LUAD")
    print("\nDone. All preprocessed train/test splits saved to Data/ subfolders.")


if __name__ == "__main__":
    main()