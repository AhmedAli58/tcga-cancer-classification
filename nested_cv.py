"""
Nested cross-validation with hyperparameter tuning for BRCA and LUAD.

This replaces the single train/test split with a proper nested CV design:

  OUTER LOOP (5 folds, patient-grouped, stratified):
      Rotates which patients are held out as the test set. Gives a genuine
      mean +/- std performance estimate instead of one point estimate from
      a single lucky/unlucky split.

  INNER LOOP (3 folds, patient-grouped, stratified, inside each outer
  training set only):
      Tunes hyperparameters via RandomizedSearchCV. The inner loop NEVER
      sees the outer test fold, so tuning cannot leak into the reported
      performance.

  NORMALIZATION:
      Gene filtering is done once globally (per the frozen literature
      review - treated as data cleaning, low leakage risk). Library-size
      normalization + log2 is refit inside EACH outer fold's training
      portion only, then applied to that fold's test portion - exactly
      the same leakage rule used in Phase 2, just repeated per fold.

  FEATURE IMPORTANCE STABILITY:
      For Random Forest, records the top-30 genes from EACH outer fold, so
      we can check afterward how consistent the "important genes" list is
      across different patient subsets - this is what makes the Phase 4
      gene list defensible rather than a one-split artifact.

  RESULTS PERSISTENCE:
      Every outer fold's F1/Precision/Recall/MCC/AUC is written to disk per
      model per cancer (Data/nested_cv_fold_metrics.csv), alongside a summary
      table of mean +/- std across folds (Data/nested_cv_summary.csv). These
      saved tables are the report's final results - not just console output.

NOTE: This takes considerably longer to run than the baseline pass -
5 outer folds x tuning search x 4 models x 2 cancer types. Expect a long
runtime; let it run in the background.
"""

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Run: pip3 install xgboost")

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_ITER_SEARCH = 8  # RandomizedSearchCV iterations per model, kept low for runtime
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


def normalize_fold(X_train_raw, X_test_raw):
    train_lib = X_train_raw.sum(axis=1)
    test_lib = X_test_raw.sum(axis=1)
    median_lib = train_lib.median()

    X_train_norm = np.log2(X_train_raw.div(train_lib, axis=0) * median_lib + 1)
    X_test_norm = np.log2(X_test_raw.div(test_lib, axis=0) * median_lib + 1)
    return X_train_norm, X_test_norm


METRIC_KEYS = ["F1", "Precision", "Recall", "MCC", "AUC"]


def compute_metrics(y_true, y_pred, y_score):
    """All five headline metrics for one fold.
    y_score = P(positive class) for LR/RF/XGB, or decision_function score for
    the Linear SVM; roc_auc_score accepts either."""
    return {
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
    }


def get_search_space(model_type):
    if model_type == "lr":
        return {"C": uniform(0.001, 10)}
    elif model_type == "svm":
        return {"C": uniform(0.001, 10)}
    elif model_type == "rf":
        return {
            "n_estimators": randint(150, 400),
            "max_depth": randint(3, 20),
            "min_samples_leaf": randint(1, 5),
        }
    elif model_type == "xgb":
        return {
            "n_estimators": randint(150, 400),
            "max_depth": randint(2, 8),
            "learning_rate": uniform(0.01, 0.3),
        }


def tune_and_fit(model_type, X_train, y_train, groups_train):
    inner_cv = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if model_type == "lr":
        base = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                   max_iter=2000, random_state=RANDOM_STATE)
    elif model_type == "svm":
        base = LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    elif model_type == "rf":
        base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    elif model_type == "xgb":
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        base = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                              random_state=RANDOM_STATE, n_jobs=-1)

    search = RandomizedSearchCV(
        base, get_search_space(model_type), n_iter=N_ITER_SEARCH,
        scoring="f1", cv=inner_cv.split(X_train, y_train, groups=groups_train),
        random_state=RANDOM_STATE, n_jobs=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def nested_cv_for_cancer(cancer_type, base_dir):
    print("\n" + "=" * 60)
    print(f"NESTED CV: {cancer_type}")
    print("=" * 60)

    expr, labels = load_and_align(cancer_type, base_dir)
    expr_filtered = filter_low_expression_genes(expr, labels)
    patient_ids = labels[ID_COL].str[:12]
    y = (labels[LABEL_COL] == "Tumor").astype(int)

    outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {"Logistic Regression": [], "Linear SVM": [], "Random Forest": [], "XGBoost": []}
    rf_top_genes_per_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(expr_filtered, y, groups=patient_ids)):
        print(f"\n  --- Outer fold {fold_i + 1}/{OUTER_FOLDS} ---")

        X_train_raw, X_test_raw = expr_filtered.iloc[train_idx], expr_filtered.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = patient_ids.iloc[train_idx]

        # Leakage check
        overlap = set(patient_ids.iloc[train_idx]) & set(patient_ids.iloc[test_idx])
        assert len(overlap) == 0, f"Patient leakage in outer fold {fold_i}!"

        X_train_norm, X_test_norm = normalize_fold(X_train_raw, X_test_raw)
        feature_names = X_train_norm.columns.tolist()

        # Scaled matrices shared by the linear models (fit on this fold's train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_norm)
        X_test_scaled = scaler.transform(X_test_norm)

        # --- Logistic Regression (scaled, tuned) ---
        lr_best, lr_params = tune_and_fit("lr", X_train_scaled, y_train, groups_train)
        y_pred = lr_best.predict(X_test_scaled)
        y_proba = lr_best.predict_proba(X_test_scaled)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["params"] = lr_params
        results["Logistic Regression"].append(m)
        print(f"    LR:  F1={m['F1']:.3f}, P={m['Precision']:.3f}, R={m['Recall']:.3f}, "
              f"MCC={m['MCC']:.3f}, AUC={m['AUC']:.3f}, params={lr_params}")

        # --- Linear SVM (scaled, tuned) ---
        # AUC uses decision_function scores (LinearSVC has no predict_proba).
        svm_best, svm_params = tune_and_fit("svm", X_train_scaled, y_train, groups_train)
        y_pred = svm_best.predict(X_test_scaled)
        y_dscore = svm_best.decision_function(X_test_scaled)
        m = compute_metrics(y_test, y_pred, y_dscore)
        m["params"] = svm_params
        results["Linear SVM"].append(m)
        print(f"    SVM: F1={m['F1']:.3f}, P={m['Precision']:.3f}, R={m['Recall']:.3f}, "
              f"MCC={m['MCC']:.3f}, AUC={m['AUC']:.3f}, params={svm_params}")

        # --- Random Forest (tuned) ---
        rf_best, rf_params = tune_and_fit("rf", X_train_norm, y_train, groups_train)
        y_pred = rf_best.predict(X_test_norm)
        y_proba = rf_best.predict_proba(X_test_norm)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["params"] = rf_params
        results["Random Forest"].append(m)
        print(f"    RF:  F1={m['F1']:.3f}, P={m['Precision']:.3f}, R={m['Recall']:.3f}, "
              f"MCC={m['MCC']:.3f}, AUC={m['AUC']:.3f}, params={rf_params}")

        top_idx = np.argsort(rf_best.feature_importances_)[::-1][:30]
        rf_top_genes_per_fold.append(set(np.array(feature_names)[top_idx]))

        # --- XGBoost (tuned) ---
        if HAS_XGB:
            xgb_best, xgb_params = tune_and_fit("xgb", X_train_norm, y_train, groups_train)
            y_pred = xgb_best.predict(X_test_norm)
            y_proba = xgb_best.predict_proba(X_test_norm)[:, 1]
            m = compute_metrics(y_test, y_pred, y_proba)
            m["params"] = xgb_params
            results["XGBoost"].append(m)
            print(f"    XGB: F1={m['F1']:.3f}, P={m['Precision']:.3f}, R={m['Recall']:.3f}, "
                  f"MCC={m['MCC']:.3f}, AUC={m['AUC']:.3f}, params={xgb_params}")

    # --- Summary across outer folds ---
    print(f"\n  {cancer_type} NESTED CV SUMMARY (mean +/- std across {OUTER_FOLDS} outer folds)")
    for model_name, fold_results in results.items():
        if fold_results:
            parts = [f"{k}={np.mean([r[k] for r in fold_results]):.3f}"
                     f"+/-{np.std([r[k] for r in fold_results]):.3f}" for k in METRIC_KEYS]
            print(f"    {model_name}: " + ", ".join(parts))

    # --- Feature importance stability ---
    print(f"\n  {cancer_type} RF top-30 gene stability across outer folds:")
    gene_counts = {}
    for gene_set in rf_top_genes_per_fold:
        for g in gene_set:
            gene_counts[g] = gene_counts.get(g, 0) + 1
    stable_genes = {g: c for g, c in gene_counts.items() if c == OUTER_FOLDS}
    print(f"    Genes in top-30 in ALL {OUTER_FOLDS} folds (fully stable): {len(stable_genes)}")
    partial_genes = {g: c for g, c in gene_counts.items() if c >= OUTER_FOLDS - 1 and c < OUTER_FOLDS}
    print(f"    Genes in top-30 in {OUTER_FOLDS - 1}/{OUTER_FOLDS} folds: {len(partial_genes)}")

    # Save stable gene list for Phase 4
    stable_df = pd.DataFrame(
        [{"gene": g, "folds_present": c} for g, c in sorted(gene_counts.items(), key=lambda x: -x[1])]
    )
    stable_df.to_csv(f"{base_dir}/{cancer_type}_rf_gene_stability.csv", index=False)
    print(f"    Saved gene stability table: {base_dir}/{cancer_type}_rf_gene_stability.csv")

    return results, stable_genes


def build_fold_rows(cancer_short, results):
    """One row per (model, outer fold) with all five metrics."""
    rows = []
    for model_name, fold_results in results.items():
        for i, r in enumerate(fold_results):
            row = {"cancer": cancer_short, "model": model_name, "fold": i + 1}
            for k in METRIC_KEYS:
                row[k] = round(float(r[k]), 4)
            rows.append(row)
    return rows


def build_summary_rows(cancer_short, results):
    """One row per (model) with mean and std across outer folds for each metric."""
    rows = []
    for model_name, fold_results in results.items():
        if not fold_results:
            continue
        row = {"cancer": cancer_short, "model": model_name, "n_folds": len(fold_results)}
        for k in METRIC_KEYS:
            vals = [r[k] for r in fold_results]
            row[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{k}_std"] = round(float(np.std(vals)), 4)
        rows.append(row)
    return rows


def main():
    brca_results, brca_stable = nested_cv_for_cancer("TCGA_BRCA", "Data/TCGA_BRCA")
    luad_results, luad_stable = nested_cv_for_cancer("TCGA_LUAD", "Data/TCGA_LUAD")

    all_results = [("BRCA", brca_results), ("LUAD", luad_results)]

    # --- Persist fold-by-fold and summary tables (report-ready) ---
    fold_rows, summary_rows = [], []
    for cancer_short, results in all_results:
        fold_rows.extend(build_fold_rows(cancer_short, results))
        summary_rows.extend(build_summary_rows(cancer_short, results))

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    fold_path = "Data/nested_cv_fold_metrics.csv"
    summary_path = "Data/nested_cv_summary.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL NESTED CV RESULTS (report-ready)")
    print("=" * 60)
    for cancer, results in all_results:
        print(f"\n{cancer}:")
        for model_name, fold_results in results.items():
            if fold_results:
                parts = [f"{k}={np.mean([r[k] for r in fold_results]):.3f}"
                         f"+/-{np.std([r[k] for r in fold_results]):.3f}" for k in METRIC_KEYS]
                print(f"  {model_name}: " + ", ".join(parts))

    print(f"\nSaved fold-by-fold metrics: {fold_path}  ({len(fold_df)} rows)")
    print(f"Saved summary (mean+/-std): {summary_path}  ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()