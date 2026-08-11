# TCGA Cancer Classification

Tumor versus normal tissue classification from RNA-seq gene expression, using two
TCGA cohorts: breast cancer (BRCA) and lung adenocarcinoma (LUAD). Four classifiers
are trained and compared per cohort, the genes driving each model are identified,
and those genes are compared across the two cancers.

## Data
Raw RNA-seq read counts for BRCA (1,217 samples) and LUAD (590 samples), 60,660
genes per sample (Ensembl IDs), each sample labelled tumor or normal. Both cohorts
are imbalanced towards tumor tissue (about nine to one). Data files are not included
in this repository.

## Pipeline
Run in order:

- `phase1_inspect.py` — inspect the raw data (dimensions, class balance).
- `phase2_preprocess.py` — filter low-expression genes, library-size normalize and
  log2(x+1) transform, and create a stratified, patient-grouped 80/20 train/test
  split with normalization fit on training data only.
- `phase3_train.py` — train and evaluate four models (logistic regression, linear
  SVM, random forest, XGBoost) on the held-out split. Reports F1, precision, recall,
  balanced accuracy, MCC, and AUC, and saves per-model feature importances.
- `nested_cv.py` — nested cross-validation (outer 5-fold for evaluation, inner 3-fold
  RandomizedSearchCV for hyperparameter tuning) for stable performance estimates, and
  Random Forest gene-stability across folds.
- `phase4_compare.py` — cross-cancer comparison of the top-30 Random Forest genes on
  the shared gene set.
- `phase4b_overlap_by_threshold.py` — reproducible cross-cancer gene overlap for all
  four models at top 10/30/50/100/200, using the same shared-gene-universe method.
- `map_gene_names.py` — map Ensembl gene IDs to gene symbols (via mygene).

## Validation
- `sanity_leakage_check.py` — verifies no patient appears in both train and test.
- `sanity_splitter_comparison.py` — compares splitting strategies.
- `sanity_confounder_check.py` — checks whether the signal is biological rather than
  driven by collection site (batch), for the top genes tested.
- `sanity_permutation_test.py` — label-shuffling check. On shuffled labels, performance
  drops to chance, which provides supporting evidence (not proof) that the models learn
  real signal rather than noise. This was a single shuffle and covered logistic
  regression, linear SVM, and random forest.

## Requirements
See `requirements.txt`.
