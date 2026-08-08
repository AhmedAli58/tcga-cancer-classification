"""
Sanity check (3): is the COL10A1 / HSD17B6 signal biology or batch?

The only non-biological covariate recoverable from these (truncated) TCGA
barcodes is TSS = Tissue Source Site (barcode field 2, e.g. 'BH', 'E2') -
the collection site, and the classic TCGA batch confounder. Plate/portion
are not present in the IDs, so TSS is our batch proxy.

For each consensus gene we ask: does its tumor-vs-normal signal survive once
we account for collection site, or could site alone explain it?

Metrics reported per gene (on library-size + log2 normalized expression):
  - eta2(label)            : variance in gene explained by Tumor/Normal
  - eta2(site, all)        : variance explained by TSS across all samples
  - eta2(site | tumors)    : variance explained by TSS WITHIN tumors only
  - eta2(site | normals)   : variance explained by TSS WITHIN normals only
  - AUC(gene -> tumor)     : how well the raw gene separates T vs N
  - within-site direction  : among sites holding BOTH T and N, does the gene
                             move the same way (T>N or T<N) every time?

Reading it: if eta2(label) dominates and the within-site direction is
consistent, the signal is biological. If eta2(site|tumors) is large (site
drives the gene even inside one tissue type) the marker is batch-tainted.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABEL_COL = "Tissue Type"
ID_COL = "Sample ID"

GENES = {  # symbol -> Ensembl prefix (version-agnostic)
    "COL10A1": "ENSG00000123500",
    "HSD17B6": "ENSG00000025423",
}


def eta_squared(values, groups):
    """Classic ANOVA eta^2 = SS_between / SS_total for a numeric var over groups."""
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return ss_between / ss_total


def load_norm_expr(cancer_type, base_dir):
    raw = pd.read_csv(f"{base_dir}/{cancer_type}_geneexpression.csv", index_col=0)
    labels = pd.read_csv(f"{base_dir}/{cancer_type}_labels.csv")
    labels = labels[labels[ID_COL].isin(raw.index)].reset_index(drop=True)
    raw = raw.loc[labels[ID_COL].values]
    # library-size + log2 normalization (global; leakage irrelevant for EDA)
    lib = raw.sum(axis=1)
    norm = np.log2(raw.div(lib, axis=0) * lib.median() + 1)
    labels["TSS"] = labels[ID_COL].str.split("-").str[1]
    labels["is_tumor"] = (labels[LABEL_COL] == "Tumor").astype(int)
    return norm, labels


def resolve_column(norm, prefix):
    hits = [c for c in norm.columns if c.split(".")[0] == prefix]
    return hits[0] if hits else None


def analyze(cancer_type, base_dir):
    print("=" * 74)
    print(f"CONFOUNDER CHECK: {cancer_type}  (batch proxy = TSS / collection site)")
    print("=" * 74)
    norm, labels = load_norm_expr(cancer_type, base_dir)
    tss = labels["TSS"].values
    is_tumor = labels["is_tumor"].values

    # ---- Site vs label confounding structure ----
    n_sites = labels["TSS"].nunique()
    site_label = labels.groupby("TSS")["is_tumor"].agg(["size", "sum"])
    site_label["normals"] = site_label["size"] - site_label["sum"]
    both = site_label[(site_label["sum"] > 0) & (site_label["normals"] > 0)]
    normals_total = int((is_tumor == 0).sum())
    normals_in_both = int(both["normals"].sum())
    print(f"  Sites (TSS): {n_sites} | samples: {len(labels)} "
          f"| Tumor:Normal = {int(is_tumor.sum())}:{normals_total}")
    print(f"  Sites containing BOTH tumor and normal: {len(both)} "
          f"(they hold {normals_in_both}/{normals_total} of all normals)")
    # how concentrated are normals? top-3 sites' share
    top_sites = site_label.sort_values("normals", ascending=False)
    top3_share = top_sites["normals"].head(3).sum() / max(normals_total, 1)
    print(f"  Normal-sample concentration: top-3 sites hold {top3_share*100:.0f}% of normals\n")

    rows = []
    for sym, prefix in GENES.items():
        col = resolve_column(norm, prefix)
        if col is None:
            print(f"  {sym} ({prefix}): NOT in filtered expression - skipped")
            continue
        g = norm[col].values

        eta_label = eta_squared(g, is_tumor)
        eta_site_all = eta_squared(g, tss)
        eta_site_tum = eta_squared(g[is_tumor == 1], tss[is_tumor == 1])
        eta_site_nrm = eta_squared(g[is_tumor == 0], tss[is_tumor == 0])
        auc = roc_auc_score(is_tumor, g)
        # direction consistency within sites that have both classes
        dirs = []
        for site in both.index:
            mask = labels["TSS"].values == site
            gt = g[mask & (is_tumor == 1)]
            gn = g[mask & (is_tumor == 0)]
            if len(gt) and len(gn):
                dirs.append(np.sign(np.median(gt) - np.median(gn)))
        dirs = np.array(dirs)
        consistent = int((dirs == dirs[0]).sum()) if len(dirs) else 0
        overall_dir = "T>N" if np.median(g[is_tumor == 1]) > np.median(g[is_tumor == 0]) else "T<N"

        print(f"  --- {sym} ({col}) ---")
        print(f"    overall separation: AUC(gene->tumor) = {auc:.3f}   direction {overall_dir}")
        print(f"    eta2(label / Tumor-vs-Normal)      = {eta_label:.3f}   <- biological axis")
        print(f"    eta2(site, all samples)            = {eta_site_all:.3f}")
        print(f"    eta2(site | tumors only)           = {eta_site_tum:.3f}   <- batch axis (within tumor)")
        print(f"    eta2(site | normals only)          = {eta_site_nrm:.3f}   <- batch axis (within normal)")
        print(f"    within-site T-vs-N direction: {consistent}/{len(dirs)} sites agree ({overall_dir})")
        verdict = ("BIOLOGICAL (label dominates, direction consistent)"
                   if eta_label > 3 * max(eta_site_tum, 1e-9) and (len(dirs) == 0 or consistent == len(dirs))
                   else "AMBIGUOUS - inspect (site variance non-trivial)")
        print(f"    => {verdict}\n")
        rows.append({
            "cancer": cancer_type, "gene": sym, "ensembl": col,
            "AUC_tumor": round(auc, 3), "direction": overall_dir,
            "eta2_label": round(eta_label, 3), "eta2_site_all": round(eta_site_all, 3),
            "eta2_site_tumors": round(eta_site_tum, 3), "eta2_site_normals": round(eta_site_nrm, 3),
            "sites_both": len(dirs), "sites_dir_agree": consistent,
        })
    return rows


def main():
    rows = analyze("TCGA_BRCA", "Data/TCGA_BRCA")
    df = pd.DataFrame(rows)
    out = "Data/sanity_confounder_check.csv"
    df.to_csv(out, index=False)
    print("=" * 74)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
