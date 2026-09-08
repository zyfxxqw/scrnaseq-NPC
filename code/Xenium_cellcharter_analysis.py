from pathlib import Path

import anndata as ad
import cellcharter as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squidpy as sq


# ============================================================
# 1. Paths
# ============================================================
DATA_DIR = Path(
    "/home/zyf/NPC/summary/"
    "output-XETG00149__0104093__Region_1__20260612__101530"
)
OUT_DIR = DATA_DIR / "processed"
INPUT_FILE = OUT_DIR / "xenium_roi_celltype_annotated.h5ad"
CELLCHARTER_DIR = OUT_DIR / "cellcharter"
FIG_DIR = CELLCHARTER_DIR / "spatial_by_core"

CELLCHARTER_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================
adata = ad.read_h5ad(INPUT_FILE)

print(adata)
print("CellCharter version:", cc.__version__)
print("Squidpy version:", sq.__version__)


# ============================================================
# 3. Validate required data
# ============================================================
required_obs = ["core_id"]
required_obsm = ["spatial", "X_scVI"]

for key in required_obs:
    if key not in adata.obs:
        raise ValueError(f"Missing required key in adata.obs: {key}")

for key in required_obsm:
    if key not in adata.obsm:
        raise ValueError(f"Missing required key in adata.obsm: {key}")

if np.isnan(adata.obsm["spatial"]).any():
    raise ValueError("The spatial coordinates contain missing values")

if np.isnan(adata.obsm["X_scVI"]).any():
    raise ValueError("X_scVI contains missing values")

adata.obs["core_id"] = adata.obs["core_id"].astype(str).astype("category")

# Reduce memory use in subsequent steps.
adata.obsm["X_scVI"] = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)

print("\nCell counts by tissue core:")
print(adata.obs["core_id"].value_counts().sort_index())


# ============================================================
# 4. Build a Delaunay spatial graph for each core
# ============================================================
if hasattr(sq.gr, "spatial_neighbors_delaunay"):
    sq.gr.spatial_neighbors_delaunay(
        adata,
        spatial_key="spatial",
        library_key="core_id",
        percentile=99,
        key_added="spatial",
        n_jobs=1,
    )
else:
    sq.gr.spatial_neighbors(
        adata,
        spatial_key="spatial",
        library_key="core_id",
        coord_type="generic",
        delaunay=True,
        percentile=99,
        key_added="spatial",
    )

print("\nSpatial graph:")
print(adata.obsp["spatial_connectivities"].shape)
print("Number of spatial edges:", adata.obsp["spatial_connectivities"].nnz)


# ============================================================
# 5. Check for cross-core edges
# ============================================================
rows, cols = adata.obsp["spatial_connectivities"].nonzero()
core_values = adata.obs["core_id"].astype(str).to_numpy()
cross_core_edges = np.sum(core_values[rows] != core_values[cols])

print("Number of cross-core edges:", cross_core_edges)
if cross_core_edges != 0:
    raise ValueError("Cross-core spatial edges detected; verify library_key")


# ============================================================
# 6. Aggregate features across three spatial-neighbor layers
# ============================================================
cc.gr.aggregate_neighbors(
    adata,
    n_layers=3,
    use_rep="X_scVI",
    out_key="X_cellcharter",
    sample_key="core_id",
)

adata.obsm["X_cellcharter"] = np.asarray(
    adata.obsm["X_cellcharter"],
    dtype=np.float32,
)

print("\nX_scVI shape:", adata.obsm["X_scVI"].shape)
print("X_cellcharter shape:", adata.obsm["X_cellcharter"].shape)


# ============================================================
# 7. Run CellCharter stability clustering
# ============================================================
autok = cc.tl.ClusterAutoK(
    n_clusters=(3, 10),
    max_runs=5,
    convergence_tol=0.001,
)
autok.fit(adata, use_rep="X_cellcharter")

print("\nK with the highest stability:")
print(autok.best_k)
print("\nStability peak candidates:")
print(autok.peaks)


# ============================================================
# 8. Plot and save K stability
# ============================================================
cc.pl.autok_stability(autok)

plt.savefig(
    CELLCHARTER_DIR / "cellcharter_autok_stability.pdf",
    bbox_inches="tight",
)
plt.savefig(
    CELLCHARTER_DIR / "cellcharter_autok_stability.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()


# ============================================================
# 9. Select the final K
# ============================================================
SELECTED_K = int(autok.best_k)

# Set SELECTED_K manually after reviewing the stability plot if needed.
# SELECTED_K = 6

print("Selected K =", SELECTED_K)


# ============================================================
# 10. Predict CellCharter spatial regions
# ============================================================
predicted = autok.predict(
    adata,
    use_rep="X_cellcharter",
    k=SELECTED_K,
)
adata.obs["cellcharter_cluster"] = pd.Categorical(
    [f"CC{x}" for x in np.asarray(predicted).astype(str)]
)

print("\nCell counts by spatial region:")
print(adata.obs["cellcharter_cluster"].value_counts().sort_index())


# ============================================================
# 11. Save the AutoK model
# ============================================================
autok.save(CELLCHARTER_DIR / "autok_model", best_k=False)


# ============================================================
# 12. Identify the cell-type column
# ============================================================
celltype_candidates = [
    "celltype",
    "cell_type",
    "celltype1",
    "final_celltype",
]
celltype_key = next(
    (key for key in celltype_candidates if key in adata.obs.columns),
    None,
)

if celltype_key is None:
    raise ValueError("No cell-type column found; update celltype_candidates")

print("Cell-type column:", celltype_key)


# ============================================================
# 13. Calculate cell-type counts and proportions
# ============================================================
niche_celltype_count = pd.crosstab(
    adata.obs["cellcharter_cluster"],
    adata.obs[celltype_key],
)
niche_celltype_prop = niche_celltype_count.div(
    niche_celltype_count.sum(axis=1),
    axis=0,
)

niche_celltype_count.to_csv(
    CELLCHARTER_DIR / "cellcharter_celltype_counts.csv"
)
niche_celltype_prop.to_csv(
    CELLCHARTER_DIR / "cellcharter_celltype_proportions.csv"
)

print("\nCell-type composition by spatial region:")
print(niche_celltype_prop.round(3).to_string())


# ============================================================
# 14. Plot the cell-type composition heatmap
# ============================================================
plt.figure(
    figsize=(
        max(8, niche_celltype_prop.shape[1] * 0.6),
        max(4, niche_celltype_prop.shape[0] * 0.5),
    )
)
plt.imshow(niche_celltype_prop.to_numpy(), aspect="auto")
plt.xticks(
    range(niche_celltype_prop.shape[1]),
    niche_celltype_prop.columns,
    rotation=90,
)
plt.yticks(
    range(niche_celltype_prop.shape[0]),
    niche_celltype_prop.index,
)
plt.colorbar(label="Cell type proportion")
plt.xlabel("Cell type")
plt.ylabel("CellCharter region")
plt.tight_layout()

plt.savefig(
    CELLCHARTER_DIR / "cellcharter_celltype_heatmap.pdf",
    bbox_inches="tight",
)
plt.show()
plt.close()


# ============================================================
# 15. Identify the patient column
# ============================================================
if "patient_id" in adata.obs.columns:
    patient_key = "patient_id"
elif "patient" in adata.obs.columns:
    patient_key = "patient"
else:
    patient_key = None


# ============================================================
# 16. Calculate spatial-region proportions for each core
# ============================================================
core_region_count = pd.crosstab(
    adata.obs["core_id"],
    adata.obs["cellcharter_cluster"],
)
core_region_prop = core_region_count.div(
    core_region_count.sum(axis=1),
    axis=0,
)

core_region_count.to_csv(
    CELLCHARTER_DIR / "cellcharter_region_counts_by_core.csv"
)
core_region_prop.to_csv(
    CELLCHARTER_DIR / "cellcharter_region_proportions_by_core.csv"
)


# ============================================================
# 17. Calculate spatial-region proportions for each patient
# ============================================================
if patient_key is not None:
    patient_region_count = pd.crosstab(
        adata.obs[patient_key],
        adata.obs["cellcharter_cluster"],
    )
    patient_region_prop = patient_region_count.div(
        patient_region_count.sum(axis=1),
        axis=0,
    )
    patient_region_count.to_csv(
        CELLCHARTER_DIR / "cellcharter_region_counts_by_patient.csv"
    )
    patient_region_prop.to_csv(
        CELLCHARTER_DIR / "cellcharter_region_proportions_by_patient.csv"
    )


# ============================================================
# 18. Plot spatial regions for each core
# ============================================================
region_order = list(adata.obs["cellcharter_cluster"].cat.categories)

for core_id in adata.obs["core_id"].cat.categories:
    core = adata[adata.obs["core_id"] == core_id]
    xy = core.obsm["spatial"]
    fig, ax = plt.subplots(figsize=(8, 8))

    for region in region_order:
        mask = (
            core.obs["cellcharter_cluster"].astype(str).to_numpy() == region
        )
        if mask.sum() == 0:
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=2,
            label=region,
            rasterized=True,
        )

    ax.set_title(f"{core_id} | CellCharter K={SELECTED_K}")
    ax.set_aspect("equal")
    ax.axis("off")

    # Match the orientation used by Xenium Explorer.
    ax.invert_yaxis()
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        markerscale=4,
        frameon=False,
    )
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"{core_id}_cellcharter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        FIG_DIR / f"{core_id}_cellcharter.pdf",
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 19. Save the final CellCharter object
# ============================================================
OUTPUT_FILE = OUT_DIR / "xenium_roi_cellcharter_final.h5ad"
adata.write_h5ad(OUTPUT_FILE, compression="gzip")

print("\nCellCharter analysis completed")
print("Final object:", OUTPUT_FILE)
print("Results directory:", CELLCHARTER_DIR)
