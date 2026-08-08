"""
Phase 3 (final pass): Baseline models + CV stability check.

Models: Logistic Regression (L1, scaled), Linear SVM (scaled),
Random Forest (unscaled), XGBoost (unscaled).

Changes from previous version:
  - FIX: precision_score / recall_score are now imported (were used in
    evaluate() but missing from imports -> crash).
  - NEW: Linear SVM (LinearSVC) added as a 4th model. Scaled like LR.
    AUC is computed from decision_function scores (LinearSVC has no
    predict_proba). Feature importance = |coef_|.
  - NEW: the headline metrics table is now SAVED to
    Data/phase3_model_metrics.csv (previously only printed).
  - Logistic Regression and SVM use StandardScaler (fit on train only,
    applied to test) - RF/XGBoost remain unscaled (tree models are
    scale-invariant).
  - Adds a lightweight 3-fold cross-validation stability check, run ONLY on
    the training data (test set is never touched by CV), to show that
    single-split results aren't a fluke. Reports mean +/- std per metric.
  - Saves full model importances for Phase 4 / Phase 5.
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
    print("WARNING: xgboost not installed. Run: pip3 install xgboost")

RANDOM_STATE = 42
CV_FOLDS = 3  # kept low to finish quickly; increase later for the final report


def load_split(cancer_type: str, base_dir: str):
    X_train = pd.read_csv(f"{base_dir}/{cancer_type}_X_train.csv", index_col=0)
    X_test = pd.read_csv(f"{base_dir}/{cancer_type}_X_test.csv", index_col=0)
    y_train = pd.read_csv(f"{base_dir}/{cancer_type}_y_train.csv")
    y_test = pd.read_csv(f"{base_dir}/{cancer_type}_y_test.csv")

    y_train_bin = (y_train["Tissue Type"] == "Tumor").astype(int)
    y_test_bin = (y_test["Tissue Type"] == "Tumor").astype(int)

    return X_train, X_test, y_train, y_test, y_train_bin, y_test_bin


def evaluate(model_name, y_true, y_pred, y_score):
    """y_score = probability of the positive class OR a decision-function score.
    roc_auc_score accepts either."""
    metrics = {
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
    }
    print(f"\n  {model_name}:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
    return metrics


def _importances(model, model_type):
    if model_type in ("lr", "svm"):
        return np.abs(model.coef_[0])
    return model.feature_importances_


def top_features(model, feature_names, model_type, n=15):
    importances = _importances(model, model_type)
    idx = np.argsort(importances)[::-1][:n]
    return [(feature_names[i], importances[i]) for i in idx]


def save_full_importances(model, feature_names, model_type, cancer_type, base_dir, tag):
    importances = _importances(model, model_type)
    df = pd.DataFrame({"gene": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    out_path = f"{base_dir}/{cancer_type}_{tag}_importances.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved full importance table: {out_path}")


def cv_stability_check(X_train, y_train_bin, y_train_full, cancer_type):
    """3-fold patient-grouped stratified CV on the TRAINING data only.
    Shows whether results are stable across different splits, not just a
    property of the one split used for the headline numbers."""
    print(f"\n  --- CV stability check ({CV_FOLDS}-fold, training data only) ---")
    patient_ids = y_train_full["Sample ID"].str[:12]
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {"Logistic Regression": [], "Linear SVM": [], "Random Forest": [], "XGBoost": []}

    for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train_bin, groups=patient_ids)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train_bin.iloc[tr_idx], y_train_bin.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        lr = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                 max_iter=2000, random_state=RANDOM_STATE)
        lr.fit(X_tr_scaled, y_tr)
        fold_metrics["Logistic Regression"].append(matthews_corrcoef(y_val, lr.predict(X_val_scaled)))

        svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
        svm.fit(X_tr_scaled, y_tr)
        fold_metrics["Linear SVM"].append(matthews_corrcoef(y_val, svm.predict(X_val_scaled)))

        rf = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        fold_metrics["Random Forest"].append(matthews_corrcoef(y_val, rf.predict(X_val)))

        if HAS_XGB:
            spw = (y_tr == 0).sum() / (y_tr == 1).sum()
            xgb = XGBClassifier(n_estimators=150, scale_pos_weight=spw,
                                 eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
            xgb.fit(X_tr, y_tr)
            fold_metrics["XGBoost"].append(matthews_corrcoef(y_val, xgb.predict(X_val)))

        print(f"    Fold {fold_i + 1} done")

    print(f"\n  {cancer_type} CV stability (MCC across {CV_FOLDS} folds, training data):")
    for model_name, scores in fold_metrics.items():
        if scores:
            print(f"    {model_name}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}  "
                  f"(folds: {[round(s, 3) for s in scores]})")

    return fold_metrics


def train_and_evaluate(cancer_type: str, base_dir: str):
    print("\n" + "=" * 60)
    print(f"TRAINING MODELS: {cancer_type}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = load_split(cancer_type, base_dir)
    feature_names = X_train.columns.tolist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}

    # Shared scaler for the linear models (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Logistic Regression (L1, scaled) ---
    lr = LogisticRegression(
        penalty="l1", solver="liblinear", class_weight="balanced",
        max_iter=2000, random_state=RANDOM_STATE
    )
    lr.fit(X_train_scaled, y_train_bin)
    y_pred = lr.predict(X_test_scaled)
    y_score = lr.predict_proba(X_test_scaled)[:, 1]
    results["Logistic Regression"] = evaluate("Logistic Regression (L1, scaled)", y_test_bin, y_pred, y_score)
    results["Logistic Regression"]["top_features"] = top_features(lr, feature_names, "lr")
    save_full_importances(lr, feature_names, "lr", cancer_type, base_dir, "lr")

    # --- Linear SVM (scaled) ---
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train_bin)
    y_pred = svm.predict(X_test_scaled)
    y_score = svm.decision_function(X_test_scaled)   # no predict_proba on LinearSVC
    results["Linear SVM"] = evaluate("Linear SVM (scaled)", y_test_bin, y_pred, y_score)
    results["Linear SVM"]["top_features"] = top_features(svm, feature_names, "svm")
    save_full_importances(svm, feature_names, "svm", cancer_type, base_dir, "svm")

    # --- Random Forest (unscaled, scale-invariant) ---
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train_bin)
    y_pred = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = evaluate("Random Forest", y_test_bin, y_pred, y_score)
    results["Random Forest"]["top_features"] = top_features(rf, feature_names, "rf")
    save_full_importances(rf, feature_names, "rf", cancer_type, base_dir, "rf")

    # --- XGBoost (unscaled) ---
    if HAS_XGB:
        scale_pos_weight = (y_train_bin == 0).sum() / (y_train_bin == 1).sum()
        xgb = XGBClassifier(
            n_estimators=300, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        )
        xgb.fit(X_train, y_train_bin)
        y_pred = xgb.predict(X_test)
        y_score = xgb.predict_proba(X_test)[:, 1]
        results["XGBoost"] = evaluate("XGBoost", y_test_bin, y_pred, y_score)
        results["XGBoost"]["top_features"] = top_features(xgb, feature_names, "xgb")
        save_full_importances(xgb, feature_names, "xgb", cancer_type, base_dir, "xgb")

    print(f"\n  Top 10 genes (Random Forest) for {cancer_type}:")
    for gene, score in results["Random Forest"]["top_features"][:10]:
        print(f"    {gene}: {score:.4f}")

    cv_stability_check(X_train, y_train_bin, y_train, cancer_type)

    return results


def save_metrics_table(all_results, out_path="Data/phase3_model_metrics.csv"):
    """Flatten held-out test metrics into one tidy CSV (report-ready)."""
    rows = []
    metric_keys = ["F1", "Precision", "Recall", "Balanced Accuracy", "MCC", "AUC"]
    for cancer, results in all_results:
        for model_name, metrics in results.items():
            row = {"cancer": cancer, "model": model_name}
            for k in metric_keys:
                row[k] = round(float(metrics[k]), 4)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved headline metrics table: {out_path}")
    return df


def main():
    brca_results = train_and_evaluate("TCGA_BRCA", "Data/TCGA_BRCA")
    luad_results = train_and_evaluate("TCGA_LUAD", "Data/TCGA_LUAD")

    all_results = [("BRCA", brca_results), ("LUAD", luad_results)]

    print("\n" + "=" * 60)
    print("SUMMARY TABLE (held-out test set)")
    print("=" * 60)
    for cancer, results in all_results:
        print(f"\n{cancer}:")
        for model_name, metrics in results.items():
            print(f"  {model_name}: F1={metrics['F1']:.3f}, "
                  f"Prec={metrics['Precision']:.3f}, Rec={metrics['Recall']:.3f}, "
                  f"BalAcc={metrics['Balanced Accuracy']:.3f}, "
                  f"MCC={metrics['MCC']:.3f}, AUC={metrics['AUC']:.3f}")

    save_metrics_table(all_results)


if __name__ == "__main__":
    main()
