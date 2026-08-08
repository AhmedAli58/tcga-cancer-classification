"""
Sanity check (2): permutation (label-shuffling) test.

If the near-perfect nested-CV scores reflect real tumor-vs-normal biology,
then destroying the label<->expression association (by shuffling labels)
should collapse performance toward chance:
    MCC -> ~0, balanced accuracy -> ~0.5, AUC -> ~0.5.
If scores stay high on shuffled labels, something leaks or the CV is broken.

To keep this fast (the full nested run took ~2h), we use the SAME outer CV
(5-fold, patient-grouped StratifiedGroupKFold, per-fold library-size+log2
normalization) but FIXED, sensible hyperparameters instead of the inner
RandomizedSearch. We run REAL and PERMUTED labels through the identical
harness so the only thing that changes is the shuffle - a clean A/B.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
PERM_SEED = 0
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


def normalize_fold(X_train_raw, X_test_raw):
    train_lib = X_train_raw.sum(axis=1)
    test_lib = X_test_raw.sum(axis=1)
    median_lib = train_lib.median()
    X_train_norm = np.log2(X_train_raw.div(train_lib, axis=0) * median_lib + 1)
    X_test_norm = np.log2(X_test_raw.div(test_lib, axis=0) * median_lib + 1)
    return X_train_norm, X_test_norm


def fixed_models(y_train):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = {
        "Logistic Regression": ("scaled", LogisticRegression(
            penalty="l1", solver="liblinear", class_weight="balanced",
            C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        "Linear SVM": ("scaled", LinearSVC(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)),
        "Random Forest": ("raw", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)),
    }
    if HAS_XGB:
        models["XGBoost"] = ("raw", XGBClassifier(
            n_estimators=200, scale_pos_weight=spw, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1))
    return models


def metrics(y_true, y_pred, y_score):
    return {
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "BalAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
    }


def run_cv(expr_filtered, y, patient_ids, label_tag):
    outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    agg = {}
    for train_idx, test_idx in outer_cv.split(expr_filtered, y, groups=patient_ids):
        assert not (set(patient_ids.iloc[train_idx]) & set(patient_ids.iloc[test_idx]))
        X_tr_raw, X_te_raw = expr_filtered.iloc[train_idx], expr_filtered.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        X_tr, X_te = normalize_fold(X_tr_raw, X_te_raw)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, (space, model) in fixed_models(y_tr).items():
            if space == "scaled":
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)
                y_score = (model.decision_function(X_te_s)
                           if not hasattr(model, "predict_proba")
                           else model.predict_proba(X_te_s)[:, 1])
            else:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_score = model.predict_proba(X_te)[:, 1]
            m = metrics(y_te, y_pred, y_score)
            agg.setdefault(name, []).append(m)

    print(f"\n  [{label_tag}] mean +/- std across {OUTER_FOLDS} folds:")
    summary = {}
    for name, folds in agg.items():
        summary[name] = {k: (np.mean([f[k] for f in folds]), np.std([f[k] for f in folds]))
                         for k in folds[0]}
        s = summary[name]
        print(f"    {name:20s} "
              f"F1={s['F1'][0]:.3f}+/-{s['F1'][1]:.3f}  "
              f"BalAcc={s['BalAcc'][0]:.3f}+/-{s['BalAcc'][1]:.3f}  "
              f"MCC={s['MCC'][0]:.3f}+/-{s['MCC'][1]:.3f}  "
              f"AUC={s['AUC'][0]:.3f}+/-{s['AUC'][1]:.3f}")
    return summary


def run_cancer(cancer_type, base_dir, out_rows):
    print("=" * 72)
    print(f"PERMUTATION TEST: {cancer_type}")
    print("=" * 72)
    expr, labels = load_and_align(cancer_type, base_dir)
    expr_filtered = filter_low_expression_genes(expr, labels)
    patient_ids = labels[ID_COL].str[:12]
    y_real = (labels[LABEL_COL] == "Tumor").astype(int)

    rng = np.random.RandomState(PERM_SEED)
    y_perm = pd.Series(rng.permutation(y_real.values), index=y_real.index)
    n_same = int((y_perm.values == y_real.values).mean() * 100)
    print(f"  Prevalence Tumor={y_real.mean():.3f}; shuffled labels match originals "
          f"{n_same}% by chance (seed={PERM_SEED}).")

    real = run_cv(expr_filtered, y_real, patient_ids, "REAL labels")
    perm = run_cv(expr_filtered, y_perm, patient_ids, "PERMUTED labels")

    for name in real:
        out_rows.append({
            "cancer": cancer_type, "model": name,
            "MCC_real": round(real[name]["MCC"][0], 3),
            "MCC_perm": round(perm[name]["MCC"][0], 3),
            "BalAcc_real": round(real[name]["BalAcc"][0], 3),
            "BalAcc_perm": round(perm[name]["BalAcc"][0], 3),
            "AUC_real": round(real[name]["AUC"][0], 3),
            "AUC_perm": round(perm[name]["AUC"][0], 3),
        })
    print()


def main():
    rows = []
    run_cancer("TCGA_BRCA", "Data/TCGA_BRCA", rows)
    run_cancer("TCGA_LUAD", "Data/TCGA_LUAD", rows)
    df = pd.DataFrame(rows)
    out = "Data/sanity_permutation_test.csv"
    df.to_csv(out, index=False)
    print("=" * 72)
    print("SUMMARY: real vs permuted (should see MCC/BalAcc/AUC collapse toward chance)")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
