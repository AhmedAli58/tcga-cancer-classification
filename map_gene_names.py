"""
Maps Ensembl gene IDs (from Phase 3 top-feature output) to human-readable
gene symbols and names, using the mygene.info API.

Run this AFTER phase3_train.py. Paste in the Ensembl IDs printed in your
terminal output for each cancer type's "Top 10 genes (Random Forest)" list.
"""

import mygene

mg = mygene.MyGeneInfo()

# Paste your top gene IDs here (strip the version suffix, e.g. ".5")
brca_top_genes = [
    "ENSG00000168497", "ENSG00000149090", "ENSG00000029559",
    "ENSG00000165197", "ENSG00000166803", "ENSG00000229246",
    "ENSG00000168079", "ENSG00000277954", "ENSG00000077157",
    "ENSG00000130032",
]

luad_top_genes = [
    "ENSG00000163815", "ENSG00000182010", "ENSG00000066405",
    "ENSG00000131477", "ENSG00000154721", "ENSG00000198873",
    "ENSG00000158764", "ENSG00000135604", "ENSG00000140600",
    "ENSG00000143590",
]


def map_genes(gene_ids, label):
    print(f"\n{'=' * 60}")
    print(f"{label}: Top genes mapped to names")
    print('=' * 60)
    results = mg.querymany(gene_ids, scopes="ensembl.gene", fields="symbol,name", species="human")
    for r in results:
        symbol = r.get("symbol", "NOT FOUND")
        name = r.get("name", "")
        print(f"  {r['query']}: {symbol} — {name}")


map_genes(brca_top_genes, "BRCA")
map_genes(luad_top_genes, "LUAD")