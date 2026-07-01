# %% Setup
import pandas as pd

base = "Data/TCGA_BRCA"

# %% Peek at files
geneexp_peek = pd.read_csv(f"{base}/TCGA_BRCA_geneexpression.csv", nrows=5)
labels_peek = pd.read_csv(f"{base}/TCGA_BRCA_labels.csv", nrows=5)
metadata_peek = pd.read_csv(f"{base}/TCGA_BRCA_metadata.csv", nrows=5)

print("=== Gene expression (head) ===")
print(geneexp_peek.iloc[:, :6])
print("Shape (rows shown):", geneexp_peek.shape)

print("\n=== Labels (head) ===")
print(labels_peek)

print("\n=== Metadata (head) ===")
print(metadata_peek)

# %% Check sample ID overlap between expression and labels
geneexp_ids = pd.read_csv(f"{base}/TCGA_BRCA_geneexpression.csv", usecols=[0]).iloc[:, 0]
labels_full = pd.read_csv(f"{base}/TCGA_BRCA_labels.csv")

geneexp_ids = set(geneexp_ids)
label_ids = set(labels_full["Sample ID"])

print("Samples in geneexpression but not in labels:", len(geneexp_ids - label_ids))
print("Samples in labels but not in geneexpression:", len(label_ids - geneexp_ids))
print("Samples in both:", len(geneexp_ids & label_ids))

# %% Check class balance
matched_labels = labels_full[labels_full["Sample ID"].isin(geneexp_ids)]
print(matched_labels["Tissue Type"].value_counts())

# %% LUAD inspection
base_luad = "Data/TCGA_LUAD"

geneexp_ids_luad = set(pd.read_csv(f"{base_luad}/TCGA_LUAD_geneexpression.csv", usecols=[0]).iloc[:, 0])
labels_luad = pd.read_csv(f"{base_luad}/TCGA_LUAD_labels.csv")
label_ids_luad = set(labels_luad["Sample ID"])

print("=== LUAD sample overlap ===")
print("In geneexpression but not labels:", len(geneexp_ids_luad - label_ids_luad))
print("In labels but not geneexpression:", len(label_ids_luad - geneexp_ids_luad))
print("In both:", len(geneexp_ids_luad & label_ids_luad))

print("\n=== LUAD class balance ===")
matched_labels_luad = labels_luad[labels_luad["Sample ID"].isin(geneexp_ids_luad)]
print(matched_labels_luad["Tissue Type"].value_counts())