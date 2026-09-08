from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# ============================================================
# 1. Paths and configuration
# ============================================================
DATA_DIR = Path(
    "/home/zyf/NPC/summary/"
    "output-XETG00149__0104093__Region_1__20260612__101530"
)
OUT_DIR = DATA_DIR / "processed"
ADATA_PATH = OUT_DIR / "xenium_roi_celltype_annotated.h5ad"
NMF_PATH = OUT_DIR / "meta_program_top100_genes_with_weights.csv"
OUTPUT_H5AD = OUT_DIR / "xenium_roi_celltype_annotated_cNMF_scored.h5ad"
OVERLAP_PATH = OUT_DIR / "cNMF_top100_gene_overlap.csv"
SUMMARY_PATH = OUT_DIR / "cNMF_scoring_summary.csv"

CELLTYPE_COL = "celltype"
MALIGNANT_LABELS = ["Malignant"]
EXPRESSION_LAYER = None
GENE_SYMBOL_COL = None


# ============================================================
# 2. Load AnnData and identify malignant cells
# ============================================================
adata = sc.read_h5ad(ADATA_PATH)

print(adata)
print("Number of cells:", adata.n_obs)
print("Number of genes:", adata.n_vars)

if CELLTYPE_COL not in adata.obs.columns:
    raise KeyError(
        f"'{CELLTYPE_COL}' is not in adata.obs; update CELLTYPE_COL to a valid column"
    )

print(adata.obs[CELLTYPE_COL].astype(str).value_counts().to_string())

malignant_mask = (
    adata.obs[CELLTYPE_COL].astype(str).isin(MALIGNANT_LABELS).to_numpy()
)
print("Number of malignant cells:", malignant_mask.sum())


# ============================================================
# 3. Identify the sample column
# ============================================================
sample_candidates = [
    "patient_id",
    "patients",
    "patient",
    "sample_id",
    "sample",
    "core_id",
]
SAMPLE_COL = next(
    (column for column in sample_candidates if column in adata.obs.columns),
    None,
)
print("Detected sample column:", SAMPLE_COL)

if SAMPLE_COL is not None:
    malignant_counts = (
        adata.obs.loc[malignant_mask]
        .groupby(SAMPLE_COL, observed=True)
        .size()
        .sort_values(ascending=False)
    )
    print(malignant_counts)


# ============================================================
# 4. Select and inspect the expression matrix
# ============================================================
if EXPRESSION_LAYER is None:
    X = adata.X
    expression_source = "adata.X"
else:
    if EXPRESSION_LAYER not in adata.layers:
        raise KeyError(f"'{EXPRESSION_LAYER}' is not in adata.layers")
    X = adata.layers[EXPRESSION_LAYER]
    expression_source = f"adata.layers['{EXPRESSION_LAYER}']"

print("Expression matrix source:", expression_source)
print("Matrix shape:", X.shape)
print("Sparse matrix:", sparse.issparse(X))

if sparse.issparse(X):
    print("Minimum nonzero expression:", X.data.min())
    print("Maximum nonzero expression:", X.data.max())
else:
    print("Minimum expression:", np.nanmin(X))
    print("Maximum expression:", np.nanmax(X))


# ============================================================
# 5. Load and validate the NMF program table
# ============================================================
nmf = pd.read_csv(NMF_PATH)

print(nmf.shape)
print(nmf.columns.tolist())
print(nmf.head())

required_columns = ["meta_program", "rank", "gene"]
missing_columns = [column for column in required_columns if column not in nmf.columns]
if missing_columns:
    raise KeyError(f"The NMF file is missing required columns: {missing_columns}")

nmf["meta_program"] = nmf["meta_program"].astype(str)
nmf["gene"] = nmf["gene"].astype(str).str.strip()
nmf["rank"] = pd.to_numeric(nmf["rank"], errors="coerce")

nmf_top100 = (
    nmf.dropna(subset=required_columns)
    .sort_values(["meta_program", "rank"])
    .drop_duplicates(subset=["meta_program", "gene"])
    .groupby("meta_program", group_keys=False)
    .head(100)
    .copy()
)
print(nmf_top100.groupby("meta_program").size())


# ============================================================
# 6. Resolve feature names and build the gene index
# ============================================================
print(adata.var_names[:20].tolist())
print(adata.var.head())

if GENE_SYMBOL_COL is None:
    feature_names = pd.Index(adata.var_names.astype(str))
else:
    if GENE_SYMBOL_COL not in adata.var.columns:
        raise KeyError(f"'{GENE_SYMBOL_COL}' is not in adata.var")
    feature_names = pd.Index(adata.var[GENE_SYMBOL_COL].astype(str))

print(feature_names[:20].tolist())

gene_to_var_index = {}
duplicate_gene_symbols = []

for var_index, gene_name in enumerate(feature_names):
    gene_upper = str(gene_name).strip().upper()
    if gene_upper in {"", "NAN", "NONE"}:
        continue
    if gene_upper not in gene_to_var_index:
        gene_to_var_index[gene_upper] = var_index
    else:
        duplicate_gene_symbols.append(gene_upper)

print("Number of genes available for matching:", len(gene_to_var_index))
print("Number of duplicate gene symbols:", len(set(duplicate_gene_symbols)))


# ============================================================
# 7. Match program genes to the Xenium panel
# ============================================================
programs = sorted(nmf_top100["meta_program"].unique().tolist())
print("NMF programs:", programs)

program_var_indices = {}
program_matched_genes = {}
overlap_records = []

for program in programs:
    program_table = nmf_top100.loc[
        nmf_top100["meta_program"] == program
    ].sort_values("rank")
    matched_indices = []
    matched_genes = []
    used_var_indices = set()

    for _, row in program_table.iterrows():
        gene = str(row["gene"]).strip()
        var_index = gene_to_var_index.get(gene.upper())
        present = var_index is not None

        if present and var_index not in used_var_indices:
            matched_indices.append(var_index)
            matched_genes.append(gene)
            used_var_indices.add(var_index)

        overlap_records.append(
            {
                "meta_program": program,
                "rank": row["rank"],
                "gene": gene,
                "present_in_xenium": present,
                "adata_var_index": var_index if present else np.nan,
                "adata_gene_name": (
                    str(feature_names[var_index]) if present else np.nan
                ),
            }
        )

    program_var_indices[program] = matched_indices
    program_matched_genes[program] = matched_genes

overlap_report = pd.DataFrame(overlap_records)
overlap_summary = overlap_report.groupby("meta_program").agg(
    top100_genes=("gene", "size"),
    genes_present=("present_in_xenium", "sum"),
)
overlap_summary["genes_missing"] = (
    overlap_summary["top100_genes"] - overlap_summary["genes_present"]
)
overlap_summary["overlap_fraction"] = (
    overlap_summary["genes_present"] / overlap_summary["top100_genes"]
)
print(overlap_summary)

for program in programs:
    if not program_var_indices[program]:
        raise ValueError(f"No genes from {program} matched the Xenium panel")


# ============================================================
# 8. Extract malignant-cell expression for all matched genes
# ============================================================
union_var_indices = sorted(
    {
        var_index
        for program in programs
        for var_index in program_var_indices[program]
    }
)
print("Number of unique matched genes across all programs:", len(union_var_indices))

var_index_to_union_position = {
    var_index: position for position, var_index in enumerate(union_var_indices)
}
malignant_indices = np.flatnonzero(malignant_mask)
print("Number of malignant cells:", len(malignant_indices))

X_malignant_union = X[malignant_indices, :][:, union_var_indices]
print(X_malignant_union.shape)


# ============================================================
# 9. Calculate gene means and standard deviations
# ============================================================
if sparse.issparse(X_malignant_union):
    gene_mean = np.asarray(X_malignant_union.mean(axis=0)).ravel()
    gene_mean_square = np.asarray(
        X_malignant_union.power(2).mean(axis=0)
    ).ravel()
else:
    X_malignant_union = np.asarray(X_malignant_union)
    gene_mean = np.mean(X_malignant_union, axis=0)
    gene_mean_square = np.mean(np.square(X_malignant_union), axis=0)

gene_variance = np.maximum(gene_mean_square - np.square(gene_mean), 0)
gene_std = np.sqrt(gene_variance)

print("Number of genes with zero standard deviation:", np.sum(gene_std == 0))
print("Number of genes with positive standard deviation:", np.sum(gene_std > 0))


# ============================================================
# 10. Calculate cNMF program scores
# ============================================================
score_columns = []
scoring_summary_records = []
valid_program_genes = {}

for program in programs:
    score_col = f"{program}_cNMF_score"
    score_columns.append(score_col)

    program_union_positions = [
        var_index_to_union_position[var_index]
        for var_index in program_var_indices[program]
    ]
    valid_positions = [
        position
        for position in program_union_positions
        if np.isfinite(gene_std[position]) and gene_std[position] > 0
    ]
    if not valid_positions:
        raise ValueError(
            f"All genes matched for {program} have zero standard deviation; "
            "the score cannot be calculated"
        )

    valid_var_indices = [union_var_indices[position] for position in valid_positions]
    valid_genes = [str(feature_names[var_index]) for var_index in valid_var_indices]
    valid_program_genes[program] = valid_genes

    program_mean = gene_mean[valid_positions]
    program_std = gene_std[valid_positions]
    X_program = X_malignant_union[:, valid_positions]

    if sparse.issparse(X_program):
        inverse_std = 1.0 / program_std
        scaled_expression_mean = np.asarray(
            X_program.multiply(inverse_std.reshape(1, -1)).mean(axis=1)
        ).ravel()
        centering_term = np.mean(program_mean / program_std)
        program_score = scaled_expression_mean - centering_term
    else:
        program_score = np.mean(
            (X_program - program_mean) / program_std,
            axis=1,
        )

    full_score = np.full(adata.n_obs, np.nan, dtype=np.float32)
    full_score[malignant_indices] = program_score.astype(np.float32)
    adata.obs[score_col] = full_score

    scoring_summary_records.append(
        {
            "meta_program": program,
            "top100_genes": int(
                (nmf_top100["meta_program"] == program).sum()
            ),
            "genes_present_in_xenium": len(program_var_indices[program]),
            "genes_used_for_score": len(valid_positions),
            "zero_variance_genes": (
                len(program_var_indices[program]) - len(valid_positions)
            ),
            "mean_score_malignant": float(np.mean(program_score)),
            "std_score_malignant": float(np.std(program_score, ddof=0)),
        }
    )

    print(
        program,
        "completed; genes used:",
        len(valid_positions),
        "mean score:",
        np.mean(program_score),
    )

scoring_summary = pd.DataFrame(scoring_summary_records)
print(scoring_summary)

for program in programs:
    print("\n", program, "genes used:", len(valid_program_genes[program]))
    print(valid_program_genes[program])


# ============================================================
# 11. Assign the highest-scoring program to each malignant cell
# ============================================================
malignant_score_df = adata.obs.loc[malignant_mask, score_columns]
max_program_values = malignant_score_df.idxmax(axis=1).str.replace(
    "_cNMF_score",
    "",
    regex=False,
)
max_score_values = malignant_score_df.max(axis=1)

max_program_array = np.full(adata.n_obs, None, dtype=object)
max_program_array[malignant_indices] = max_program_values.to_numpy()
adata.obs["cNMF_max_program"] = pd.Categorical(
    max_program_array,
    categories=programs,
)

max_score_array = np.full(adata.n_obs, np.nan, dtype=np.float32)
max_score_array[malignant_indices] = max_score_values.to_numpy(dtype=np.float32)
adata.obs["cNMF_max_score"] = max_score_array

print(adata.obs.loc[malignant_mask, "cNMF_max_program"].value_counts())


# ============================================================
# 12. Store scoring metadata
# ============================================================
adata.uns["cNMF_scoring_method"] = (
    "Mean Z-scored expression of top 100 cNMF-contributing genes present "
    "in the Xenium panel"
)
adata.uns["cNMF_expression_source"] = expression_source
adata.uns["cNMF_celltype_column"] = CELLTYPE_COL
adata.uns["cNMF_malignant_labels"] = np.array(MALIGNANT_LABELS, dtype=str)
adata.uns["cNMF_score_columns"] = np.array(score_columns, dtype=str)

for program in programs:
    adata.uns[f"{program}_cNMF_genes_used"] = np.array(
        valid_program_genes[program],
        dtype=str,
    )


# ============================================================
# 13. Save the scored AnnData object
# ============================================================
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
