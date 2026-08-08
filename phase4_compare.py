"""
Phase 4: Cross-cancer gene importance comparison (BRCA vs LUAD).

Loads the full Random Forest importance tables saved by phase3_train.py,
intersects the gene sets (BRCA and LUAD have slightly different filtered
gene sets), and identifies:
    1. Genes that are top drivers in BOTH cancers (shared biology)
    2. Genes that are top drivers in ONLY one cancer (cancer-specific)
"""

import pandas as pd

TOP_N = 30  # how many top genes per cancer to consider as "important"


def load_importances(cancer_type: str, base_dir: str, model_tag: str = "rf"):
    path = f"{base_dir}/{cancer_type}_{model_tag}_importances.csv"
    return pd.read_csv(path)


def compare_cancers():
    brca = load_importances("TCGA_BRCA", "Data/TCGA_BRCA")
    luad = load_importances("TCGA_LUAD", "Data/TCGA_LUAD")

    brca_genes = set(brca["gene"])
    luad_genes = set(luad["gene"])
    shared_gene_universe = brca_genes & luad_genes

    print("=" * 60)
    print("GENE SET OVERLAP")
    print("=" * 60)
    print(f"BRCA filtered genes: {len(brca_genes)}")
    print(f"LUAD filtered genes: {len(luad_genes)}")
    print(f"Genes present in BOTH (comparable universe): {len(shared_gene_universe)}")

    brca_shared = brca[brca["gene"].isin(shared_gene_universe)].reset_index(drop=True)
    luad_shared = luad[luad["gene"].isin(shared_gene_universe)].reset_index(drop=True)

    brca_top = set(brca_shared.head(TOP_N)["gene"])
    luad_top = set(luad_shared.head(TOP_N)["gene"])

    shared_drivers = brca_top & luad_top
    brca_only = brca_top - luad_top
    luad_only = luad_top - brca_top

    print("\n" + "=" * 60)
    print(f"TOP {TOP_N} DRIVER GENES PER CANCER (within shared gene universe)")
    print("=" * 60)

    print(f"\nSHARED top drivers (important in BOTH BRCA and LUAD): {len(shared_drivers)}")
    for gene in shared_drivers:
        brca_rank = brca_shared[brca_shared["gene"] == gene].index[0] + 1
        luad_rank = luad_shared[luad_shared["gene"] == gene].index[0] + 1
        print(f"  {gene}  (BRCA rank {brca_rank}, LUAD rank {luad_rank})")

    print(f"\nBRCA-SPECIFIC top drivers (not in LUAD's top {TOP_N}): {len(brca_only)}")
    for gene in brca_only:
        print(f"  {gene}")

    print(f"\nLUAD-SPECIFIC top drivers (not in BRCA's top {TOP_N}): {len(luad_only)}")
    for gene in luad_only:
        print(f"  {gene}")

    summary_rows = []
    for gene in shared_drivers:
        summary_rows.append({"gene": gene, "category": "shared"})
    for gene in brca_only:
        summary_rows.append({"gene": gene, "category": "BRCA_specific"})
    for gene in luad_only:
        summary_rows.append({"gene": gene, "category": "LUAD_specific"})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("Data/phase4_cross_cancer_gene_comparison.csv", index=False)
    print(f"\nSaved summary to Data/phase4_cross_cancer_gene_comparison.csv")

    return shared_drivers, brca_only, luad_only


if __name__ == "__main__":
    compare_cancers()