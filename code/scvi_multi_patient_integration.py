from pathlib import Path

import numpy as np
import scanpy as sc
import scvi
import torch
from scipy import sparse


# ============================================================
# 1. Paths
# ============================================================
DATA_DIR = Path(
    "/home/zyf/NPC/summary/"
    "output-XETG00149__0104093__Region_1__20260612__101530"
)
OUT_DIR = DATA_DIR / "processed"
MODEL_DIR = OUT_DIR / "scvi_model_hvg"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Reproducibility
# ============================================================
scvi.settings.seed = 0
np.random.seed(0)
torch.manual_seed(0)


# ============================================================
# 3. Load the full-gene AnnData object
# adata.X: normalize_total + log1p
# adata.layers["counts"]: raw counts
# adata.var["highly_variable"]: HVG mask
# ============================================================
adata = sc.read_h5ad(OUT_DIR / "xenium_roi_gene_filtered_hvg.h5ad")

print("Full AnnData object:")
print(adata)


# ============================================================
# 4. Identify the patient column
# ============================================================
if "patient_id" in adata.obs.columns:
    batch_key = "patient_id"
elif "patient" in adata.obs.columns:
    batch_key = "patient"
else:
    raise ValueError("adata.obs must contain a 'patient' or 'patient_id' column")

adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype("category")

print("\nCell counts by patient:")
print(adata.obs[batch_key].value_counts().sort_index())


# ============================================================
# 5. Validate HVGs and raw counts
# ============================================================
if "highly_variable" not in adata.var.columns:
    raise ValueError("adata.var must contain a 'highly_variable' column")

if "counts" not in adata.layers:
    raise ValueError("adata.layers must contain a raw-count 'counts' layer")

n_hvg = int(adata.var["highly_variable"].sum())
print(f"\nNumber of highly variable genes: {n_hvg:,}")


# ============================================================
# 6. Create the scVI training object
# Use HVGs for training without removing other genes from adata.
# ============================================================
adata_scvi = adata[:, adata.var["highly_variable"]].copy()
counts = adata_scvi.layers["counts"]

if sparse.issparse(counts):
    counts = counts.tocsr()
    adata_scvi.layers["counts"] = counts
    count_values = counts.data
else:
    count_values = np.asarray(counts).ravel()

if np.any(count_values < 0):
    raise ValueError("The 'counts' layer contains negative values and cannot be used by scVI")

if not np.allclose(count_values, np.round(count_values)):
    raise ValueError(
        "The 'counts' layer contains non-integer values; verify that it is not log1p-transformed"
    )

print("\nscVI training object:")
print(adata_scvi)


# ============================================================
# 7. Register AnnData
# ============================================================
scvi.model.SCVI.setup_anndata(
    adata_scvi,
    layer="counts",
    batch_key=batch_key,
)


# ============================================================
# 8. Create the scVI model
# ============================================================
model = scvi.model.SCVI(
    adata_scvi,
    n_hidden=128,
    n_latent=30,
    n_layers=2,
    dropout_rate=0.1,
    dispersion="gene",
    gene_likelihood="nb",
)

print("\nModel summary:")
print(model)


# ============================================================
# 9. Train the model
# ============================================================
model.train(
    max_epochs=400,
    batch_size=512,
    early_stopping=True,
    accelerator="auto",
    devices="auto",
)


# ============================================================
# 10. Extract the scVI latent representation
# ============================================================
adata.obsm["X_scVI"] = model.get_latent_representation()
print("\nX_scVI shape:", adata.obsm["X_scVI"].shape)


# ============================================================
# 11. Build the neighbor graph from the scVI representation
# ============================================================
sc.pp.neighbors(
    adata,
    use_rep="X_scVI",
    n_neighbors=20,
    key_added="scvi_neighbors",
)


# ============================================================
# 12. UMAP
# ============================================================
sc.tl.umap(
    adata,
    neighbors_key="scvi_neighbors",
    min_dist=0.3,
    random_state=0,
)


# ============================================================
# 13. Initial Leiden clustering
# ============================================================
sc.tl.leiden(
    adata,
    neighbors_key="scvi_neighbors",
    resolution=0.6,
    key_added="leiden_scvi",
    random_state=0,
)


# ============================================================
# 14. Save the model
# ============================================================
model.save(MODEL_DIR, overwrite=True, save_anndata=False)


# ============================================================
# 15. Save the integrated full-gene AnnData object
# ============================================================
integrated_path = OUT_DIR / "xenium_roi_scvi_integrated.h5ad"
adata.write_h5ad(integrated_path, compression="gzip")

print("\nIntegrated AnnData saved to:", integrated_path)
print("scVI model saved to:", MODEL_DIR)
