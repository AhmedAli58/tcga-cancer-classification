# TCGA Cancer Classification

Tumor versus normal tissue classification from RNA-seq gene expression, using two
TCGA cohorts: breast cancer (BRCA) and lung adenocarcinoma (LUAD). The project trains
and compares four classifiers per cohort, identifies the genes driving each model,
and compares those genes across the two cancers.

## Data
RNA-seq gene expression for BRCA (1,217 samples) and LUAD (590 samples), roughly
60,660 genes per sample (Ensembl IDs), each labelled tumor or normal. Both cohorts
are imbalanced towards tumor tissue (about nine to one). Data files are not included
in this repository.

## Pipeline
Scripts are run in order:

- `phase1_inspect.py` — inspect the raw data (dimensions, class balance).
- `phase2_preprocess.py` — filter low-expression genes, library-size normalize and
  log2 transform, and create a stratified, patient-grouped 80/20 train/test split
  with normalization fit on training data only.
- `phase3_train.py` — train and evaluate four models (logistic regression, linear SVM,
  random forest, XGBoost). Reports F1, precision, recall, balanced accuracy, MCC, AUC,
  and saves per-model feature importances.
- `nested_cv.py` — nested cross-validation (outer 5-fold for evaluation, inner 3-fold
  RandomizedSearchCV for hyperparameter tuning) to produce stable performance estimates.
- `phase4_compare.py` — cross-cancer comparison of the top genes on a shared gene set.
- `phase5_feature_compare.py` — cross-model consensus of important genes per cohort.
- `phase6_figures.py` — generate the figures in `figures/`.
- `map_gene_names.py` — map Ensembl IDs to gene symbols.

## Validation
- `sanity_leakage_check.py` — confirms no patient appears in both train and test.
- `sanity_splitter_comparison.py` — compares splitting strategies.
- `sanity_confounder_check.py` — checks the signal is biological, not collection-site batch.
- `sanity_permutation_test.py` — permutation test; performance collapses to chance on
  shuffled labels, confirming the models learn real signal.

## Requirements
See `requirements.txt`.
