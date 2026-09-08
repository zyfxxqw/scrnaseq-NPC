import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from cnmf import cNMF

PATIENT_KEY = "patients"     
COUNT_LAYER = "counts_RNA"       
OUTDIR = Path("./cnmf_npc1") #

K_RANGE = list(range(2, 11))   # K 
N_ITER  = 50                 # >=50 
NUM_HVG = 2000
SEED    = 14
DENSITY_THRESHOLD = 0.1

MIN_COUNTS_PER_CELL = 50
MIN_CELLS_PER_GENE  = 10
MIN_CELLS_PER_PATIENT = 100

PDIR = OUTDIR / "per_patient"
SDIR = OUTDIR / "summary"
PDIR.mkdir(parents=True, exist_ok=True)
SDIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad('malignant_patients.h5ad')
adata



# ====== Filter out the unwanted genes ======
toremove = [f"AC{i}" for i in range(0,10)] + [f"AL{i}" for i in range(0,10)] + ["LINC", "MT-", "ENSG"]
adata = adata[:, ~adata.var_names.str.startswith(tuple(toremove))].copy()
print("Filtered genes:", adata.n_vars)

#3
def counts_to_X(adata, layer="counts_RNA"):
    ad = adata.copy()
    ad.X = ad.layers[layer]
    if sp.issparse(ad.X):
        ad.X = ad.X.astype(np.float32)
    else:
        ad.X = np.asarray(ad.X, dtype=np.float32)
    return ad

def prefilter(ad):
    ad = ad.copy()
    sc.pp.filter_cells(ad, min_counts=MIN_COUNTS_PER_CELL)
    sc.pp.filter_genes(ad, min_cells=MIN_CELLS_PER_GENE)

    if sp.issparse(ad.X):
        cs = np.array(ad.X.sum(axis=1)).ravel()
        gs = np.array(ad.X.sum(axis=0)).ravel()
    else:
        cs = ad.X.sum(axis=1)
        gs = ad.X.sum(axis=0)

    return ad[cs > 0, gs > 0].copy()

def save_top_genes(top_genes, out_csv: Path):
    if isinstance(top_genes, pd.DataFrame):
        df = top_genes
    elif isinstance(top_genes, dict):
        df = pd.DataFrame({str(k): pd.Series(v) for k, v in top_genes.items()})
    elif isinstance(top_genes, list) and len(top_genes) > 0 and isinstance(top_genes[0], (list, tuple, pd.Index)):
        df = pd.DataFrame({f"program_{i}": pd.Series(g) for i, g in enumerate(top_genes)})
    else:
        df = pd.DataFrame({"program_0": pd.Series(top_genes if isinstance(top_genes, list) else [top_genes])})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


# In[4]:


adata_counts = counts_to_X(adata, layer=COUNT_LAYER)


# In[ ]:


# ====== Per-patient cNMF ======
patients = sorted(adata_counts.obs[PATIENT_KEY].astype(str).unique())
print("n_patients =", len(patients))
print(patients)

for pid in patients:
    ad_p = adata_counts[adata_counts.obs[PATIENT_KEY].astype(str) == pid].copy()
    if ad_p.n_obs < MIN_CELLS_PER_PATIENT:
        print(f"[SKIP] {pid}: {ad_p.n_obs} cells")
        continue

    ad_p = prefilter(ad_p)
    if ad_p.n_obs < MIN_CELLS_PER_PATIENT or ad_p.n_vars < 500:
        print(f"[SKIP] {pid}: after filter cells={ad_p.n_obs}, genes={ad_p.n_vars}")
        continue

    out_p = PDIR / pid
    out_p.mkdir(parents=True, exist_ok=True)

    counts_fn = out_p / f"{pid}.counts.h5ad"
    ad_p.write(counts_fn)

    cn = cNMF(output_dir=str(PDIR), name=pid)
    cn.prepare(counts_fn=str(counts_fn), components=K_RANGE, n_iter=N_ITER, seed=SEED, num_highvar_genes=NUM_HVG)
    cn.factorize(worker_i=0, total_workers=1)
    cn.combine()
    cn.k_selection_plot(close_fig=False)
    print(f"[DONE] {pid} -> {PDIR/pid}/{pid}.k_selection.png")


# In[7]:


from pathlib import Path
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

root = Path("./cnmf_npc1/per_patient")

pngs = sorted(root.rglob("*.k_selection.png"))
print("Found", len(pngs), "png files")
pngs[:5]  
ncols = 4  
n = len(pngs)
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
axes = axes.flatten() if n > 1 else [axes]

for ax, p in zip(axes, pngs):
    img = mpimg.imread(p)
    ax.imshow(img)
    ax.set_title(p.parent.name, fontsize=10) 
    ax.axis("off")


for ax in axes[len(pngs):]:
    ax.axis("off")

plt.tight_layout()
plt.show()


# In[36]:


kfile = SDIR / "K_CHOSEN.csv"
pd.DataFrame({"patient": patients, "chosen_k": ""}).to_csv(kfile, index=False)
print("Fill this file then save:", kfile)


# In[8]:


kfile = SDIR / "K_CHOSEN.csv"
kdf = pd.read_csv(kfile)
kdf["patient"] = kdf["patient"].astype(str)
kdf["chosen_k"] = pd.to_numeric(kdf["chosen_k"], errors="coerce").astype("Int64")

missing = kdf[kdf["chosen_k"].isna()]["patient"].tolist()
if len(missing) > 0:
    raise ValueError(f"These patients missing chosen_k in {kfile}: {missing}")

K_CHOSEN = dict(zip(kdf["patient"], kdf["chosen_k"].astype(int)))
print("K_CHOSEN head:", list(K_CHOSEN.items())[:5])

for pid, k in K_CHOSEN.items():
    out_p = PDIR / pid

    cn = cNMF(output_dir=str(PDIR), name=pid)  # PDIR + name=pid
    cn.consensus(k=k, density_threshold=DENSITY_THRESHOLD)

    usage, gep_scores, gep_tpm, top_genes = cn.load_results(K=k, density_threshold=DENSITY_THRESHOLD)
    usage.to_csv(out_p / f"{pid}.usage_norm.k{k}.dt{DENSITY_THRESHOLD}.csv")
    gep_scores.to_csv(out_p / f"{pid}.gene_spectra_score.k{k}.dt{DENSITY_THRESHOLD}.csv")
    gep_tpm.to_csv(out_p / f"{pid}.gene_spectra_tpm.k{k}.dt{DENSITY_THRESHOLD}.csv")
    save_top_genes(top_genes, out_p / f"{pid}.top_genes.k{k}.dt{DENSITY_THRESHOLD}.csv")

    print(f"[EXPORT] {pid}: K={k}")


# In[9]:


def plot_patient(pid: str):
    k = K_CHOSEN[pid]
    out_p = PDIR / pid

    usage = pd.read_csv(out_p / f"{pid}.usage_norm.k{k}.dt{DENSITY_THRESHOLD}.csv", index_col=0)
    usage.columns = [f"Usage_{i}" for i in range(usage.shape[1])]

    gep_scores = pd.read_csv(out_p / f"{pid}.gene_spectra_score.k{k}.dt{DENSITY_THRESHOLD}.csv", index_col=0)

    hvgs_path = out_p / f"{pid}.overdispersed_genes.txt"
    hvgs = pd.Index([x for x in hvgs_path.read_text().split("\n") if x])

    ad_p = adata_counts[adata_counts.obs[PATIENT_KEY].astype(str) == pid].copy()
    sc.pp.normalize_total(ad_p, target_sum=10000)
    sc.pp.log1p(ad_p)
    ad_p = ad_p[:, hvgs.intersection(ad_p.var_names)].copy()

    sc.tl.pca(ad_p)
    sc.pp.neighbors(ad_p)
    sc.tl.umap(ad_p)

    common = ad_p.obs_names.intersection(usage.index)
    ad_p = ad_p[common].copy()
    ad_p.obs = pd.concat([ad_p.obs, usage.loc[common]], axis=1)

    # UMAP usage
    sc.pl.umap(ad_p, color=list(usage.columns), ncols=3, vmin=0, vmax=1, title=[f"{pid} {c}" for c in usage.columns])

    # UMAP GEP
    ad_p.obs["GEP"] = usage.loc[common].idxmax(axis=1)
    sc.pl.umap(ad_p, color="GEP", title=f"{pid} GEP (argmax usage)")

    # Heatmap programs x top genes union
    prog_by_gene = gep_scores if gep_scores.shape[0] <= gep_scores.shape[1] else gep_scores.T  # programs x genes
    top_n = 30
    genes_union = []
    for prog in prog_by_gene.index:
        genes_union.extend(prog_by_gene.loc[prog].sort_values(ascending=False).head(top_n).index.tolist())
    genes_union = pd.Index(pd.unique(genes_union))
    mat = prog_by_gene.loc[:, genes_union]

    plt.figure(figsize=(min(18, 0.25*mat.shape[1] + 4), max(6, 0.4*mat.shape[0] + 2)))
    sns.heatmap(mat, cmap="viridis")
    plt.title(f"{pid} program heatmap (K={k}, top{top_n})")
    plt.xlabel("Top genes (union)")
    plt.ylabel("Programs")
    plt.tight_layout()
    plt.show()


for pid in list(K_CHOSEN.keys())[:2]:
    plot_patient(pid)


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch

blocks = []
for pid, k in K_CHOSEN.items():
    out_p = PDIR / pid
    df = pd.read_csv(out_p / f"{pid}.gene_spectra_score.k{k}.dt{DENSITY_THRESHOLD}.csv", index_col=0)
    prog_by_gene = df if df.shape[0] <= df.shape[1] else df.T  # programs x genes
    prog_by_gene.index = [f"{pid}_Prog{i}" for i in range(prog_by_gene.shape[0])]
    blocks.append(prog_by_gene)

all_prog_by_gene = pd.concat(blocks, axis=0).fillna(0.0)  # programs x genes
X = all_prog_by_gene.values

cos_sim = pd.DataFrame(cosine_similarity(X), index=all_prog_by_gene.index, columns=all_prog_by_gene.index)
linkage = sch.linkage(X, method="average", metric="cosine")

g = sns.clustermap(
    cos_sim,
    row_linkage=linkage,
    col_linkage=linkage,
    cmap="vlag",
    center=0.0,
    xticklabels=False,
    yticklabels=False
)
plt.suptitle("Program cosine similarity (average-linkage)", y=1.02)
plt.savefig(SDIR / "program_similarity_clustermap.png", dpi=300, bbox_inches="tight")
cos_sim.to_csv(SDIR / "program_cosine_similarity.csv")
plt.show()

print("Saved to:", SDIR)


# In[11]:


qc_template = SDIR / "program_qc_template.csv"

rows = []
for pid, k in K_CHOSEN.items():
    out_p = PDIR / pid
    topf = out_p / f"{pid}.top_genes.k{k}.dt{DENSITY_THRESHOLD}.csv"
    if not topf.exists():
        raise FileNotFoundError(f"Missing top_genes for {pid}: {topf}")

    tg = pd.read_csv(topf)  # columns: program_0, program_1, ...
    for col in tg.columns:
        genes = tg[col].dropna().astype(str).tolist()
        idx = col.replace("program_", "")  # "0"
        prog = f"{pid}_Prog{idx}"          # 统一全局 program ID

        rows.append({
            "program": prog,
            "patient": pid,
            "keep": "",          # 你填 1/0
            "note": "",          # 你写原因：mito/ribo/stress/...
            "top_genes_30": ",".join(genes[:30]),
            "top_genes_50": ",".join(genes[:50]),
        })

qc_df = pd.DataFrame(rows)
qc_df.to_csv(qc_template, index=False)
print("Wrote:", qc_template, "n_programs =", qc_df.shape[0])
qc_df.head()


# In[12]:


from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch

# 汇总 programs x genes（来自每患者 gene_spectra_score）
blocks = []
for pid, k in K_CHOSEN.items():
    out_p = PDIR / pid
    f = out_p / f"{pid}.gene_spectra_score.k{k}.dt{DENSITY_THRESHOLD}.csv"
    df = pd.read_csv(f, index_col=0)
    prog_by_gene = df if df.shape[0] <= df.shape[1] else df.T  # programs x genes
    prog_by_gene.index = [f"{pid}_Prog{i+1}" for i in range(prog_by_gene.shape[0])]
    blocks.append(prog_by_gene)

all_prog_by_gene = pd.concat(blocks, axis=0).fillna(0.0)  # programs x genes
print("all_prog_by_gene:", all_prog_by_gene.shape)
all_prog_by_gene.head()


# In[13]:


qc_file = SDIR / "program_qc_keep.csv"
qc = pd.read_csv(qc_file)
qc["program"] = qc["program"].astype(str).str.strip()
qc["keep"] = pd.to_numeric(qc["keep"], errors="coerce").fillna(0).astype(int)

keep_programs = qc.loc[qc["keep"] == 1, "program"].tolist()
keep_programs = [p for p in keep_programs if p in all_prog_by_gene.index]
print("kept programs:", len(keep_programs), "/", all_prog_by_gene.shape[0])

prog_keep = all_prog_by_gene.loc[keep_programs].copy()
X = prog_keep.values

cos_sim = pd.DataFrame(
    cosine_similarity(X),
    index=prog_keep.index,
    columns=prog_keep.index
)
linkage = sch.linkage(X, method="average", metric="cosine")


# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

def prog_to_patient(p):  # NPC10_Prog0 -> NPC10
    return p.split("_")[1]

# 患者颜色条
patients = sorted({prog_to_patient(p) for p in prog_keep.index})
pal_pat = sns.color_palette("tab20", n_colors=len(patients))
pat_color = dict(zip(patients, pal_pat))
row_patient = pd.Series([pat_color[prog_to_patient(p)] for p in prog_keep.index], index=prog_keep.index)

# 设定 program clusters 数（你可调）
N_CLUSTERS =4
cluster_id = sch.fcluster(linkage, t=N_CLUSTERS, criterion="maxclust")
cluster_id = pd.Series(cluster_id, index=prog_keep.index, name="cluster")

pal_clu = sns.color_palette("husl", n_colors=N_CLUSTERS)
clu_color = {i+1: pal_clu[i] for i in range(N_CLUSTERS)}
row_cluster = pd.Series([clu_color[c] for c in cluster_id.loc[prog_keep.index]], index=prog_keep.index)

row_colors = pd.DataFrame({"patient": row_patient, "cluster": row_cluster}, index=prog_keep.index)

g = sns.clustermap(
    cos_sim,
    row_linkage=linkage,
    col_linkage=linkage,
    row_colors=row_colors,
    col_colors=row_colors,
    cmap="vlag",
    center=0.0,
    xticklabels=False,
    yticklabels=False
)
plt.suptitle("QC-kept program similarity (avg-linkage)", y=1.02)
plt.savefig(SDIR / "program_similarity_clustermap_QCkept.pdf", bbox_inches="tight")
plt.show()
print("Saved:", SDIR / "program_similarity_clustermap_QCkept.png")





