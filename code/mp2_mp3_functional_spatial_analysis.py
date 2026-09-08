# MP2-MP3 functional spatial analysis
# Refined from the original notebook while preserving its analysis structure.

from pathlib import Path
import warnings
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# 0. Paths
# ============================================================
DATA_DIR = Path('/home/zyf/NPC/summary/output-XETG00149__0104093__Region_1__20260612__101530')
OUT_DIR = DATA_DIR / 'processed'
INPUT_FILE = OUT_DIR / 'xenium_roi_cellcharter_final.h5ad'
RESULT_DIR = OUT_DIR / 'MP2_Tcell_functional_spatial_analysis'
FIG_DIR = RESULT_DIR / 'figures'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Parameters
# ============================================================
PATIENT_KEY = 'patient'
CORE_KEY = 'core_id'
REGION_KEY = 'cellcharter_cluster'
CELLTYPE_KEY = 'celltype'
MP2_KEY = 'MP2_cNMF_score'
MP3_KEY = 'MP3_cNMF_score'
EXHAUSTED_KEY = 'Exhausted_score'
TREG_KEY = 'Treg_score'
CYTOTOXIC_KEY = 'Cytotoxic_score'
RADII = [20, 30, 50, 75, 100]
PRIMARY_RADIUS = 50
CHUNK_SIZE = 20000
MAX_MODEL_CELLS_PER_CORE = 8000
MAX_ANCHORS_PER_CORE = 5000
RANDOM_SEED = 1234
rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. Load data
# ============================================================
adata = ad.read_h5ad(INPUT_FILE)
print(adata)


# ============================================================
# 3. Validate required fields
# ============================================================
required_obs = [PATIENT_KEY, CORE_KEY, REGION_KEY, CELLTYPE_KEY, MP2_KEY, MP3_KEY, EXHAUSTED_KEY, TREG_KEY, CYTOTOXIC_KEY]
for key in required_obs:
    if key not in adata.obs.columns:
        raise ValueError(f'Missing required adata.obs field: {key}')
if 'spatial' not in adata.obsm:
    raise ValueError("adata.obsm does not contain 'spatial'")


# ============================================================
# 4. Define cell populations
# ============================================================
celltype = adata.obs[CELLTYPE_KEY].astype(str).to_numpy()
patient = adata.obs[PATIENT_KEY].astype(str).to_numpy()
core = adata.obs[CORE_KEY].astype(str).to_numpy()
region = adata.obs[REGION_KEY].astype(str).to_numpy()
is_malignant = celltype == 'Malignant'
is_cd8 = np.isin(celltype, ['Cytotoxic_T', 'Exhausted_T'])
is_treg = celltype == 'Treg'
is_tcell = np.isin(celltype, ['Other_T', 'Cytotoxic_T', 'Exhausted_T', 'Treg', 'Cycling_T'])
is_myeloid = np.isin(celltype, ['Myeloid', 'IFN_Myeloid', 'LAMP3_DC', 'pDC'])
is_exhausted_identity = celltype == 'Exhausted_T'
print('\nCell numbers')
print('Malignant:', is_malignant.sum())
print('CD8-like:', is_cd8.sum())
print('Treg:', is_treg.sum())
print('All T:', is_tcell.sum())
print('Myeloid:', is_myeloid.sum())


# ============================================================
# 5. Load spatial coordinates
# ============================================================
xy = np.asarray(adata.obsm['spatial'], dtype=np.float64)


# ============================================================
# 6. Standardize MP2 and MP3 scores within each core
# ============================================================
mp2_raw = adata.obs[MP2_KEY].astype(float).to_numpy()
mp3_raw = adata.obs[MP3_KEY].astype(float).to_numpy()
mp2_z = np.full(adata.n_obs, np.nan)
mp3_z = np.full(adata.n_obs, np.nan)

def safe_z(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros(len(x))
    return (x - np.nanmean(x)) / sd
for c in np.unique(core):
    mask = (core == c) & is_malignant
    if mask.sum() == 0:
        continue
    mp2_z[mask] = safe_z(mp2_raw[mask])
    mp3_z[mask] = safe_z(mp3_raw[mask])


# ============================================================
# 7. Define MP2-high malignant cells
# ============================================================
mp2_high = np.zeros(adata.n_obs, dtype=bool)
for c in np.unique(core):
    mask = (core == c) & is_malignant & np.isfinite(mp2_raw)
    if mask.sum() < 20:
        continue
    cutoff = np.quantile(mp2_raw[mask], 0.75)
    mp2_high[mask & (mp2_raw >= cutoff)] = True
adata.obs['MP2_high_malignant'] = mp2_high
print('\nMP2-high malignant:', mp2_high.sum())


# ============================================================
# 8. Calculate the IFN-response score
# ============================================================
IFN_GENES = ['IFIT1', 'IFIT3', 'ISG15', 'IFI6', 'GBP1', 'STAT1', 'IRF1', 'CXCL10']
if adata.raw is not None:
    gene_pool = set(adata.raw.var_names)
else:
    gene_pool = set(adata.var_names)
ifn_genes = [g for g in IFN_GENES if g in gene_pool]
print('\nIFN genes used:', ifn_genes)

def get_gene_vector(gene):
    if adata.raw is not None and gene in adata.raw.var_names:
        x = adata.raw[:, gene].X
    else:
        x = adata[:, gene].X
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()
if len(ifn_genes) >= 3:
    ifn_score = np.zeros(adata.n_obs, dtype=np.float32)
    for gene in ifn_genes:
        ifn_score += get_gene_vector(gene).astype(np.float32)
    ifn_score /= len(ifn_genes)
else:
    warnings.warn('Fewer than three IFN genes were found; using IFN_Myeloid identity as a proxy IFN signal.')
    ifn_score = (celltype == 'IFN_Myeloid').astype(float)
adata.obs['IFN_response_score'] = ifn_score


# ============================================================
# 9. Select T-cell populations for analysis
# ============================================================
target_mask = is_cd8 | is_treg
target_idx = np.where(target_mask)[0]
print('\nT-cell targets:', len(target_idx))


# ============================================================
# 10. Initialize spatial metrics
# ============================================================
n_target = len(target_idx)
result = {'global_index': target_idx.copy(), 'patient': patient[target_idx], 'core_id': core[target_idx], 'cellcharter_cluster': region[target_idx], 'celltype': celltype[target_idx], 'Exhausted_score': adata.obs[EXHAUSTED_KEY].astype(float).to_numpy()[target_idx], 'Treg_score': adata.obs[TREG_KEY].astype(float).to_numpy()[target_idx], 'Cytotoxic_score': adata.obs[CYTOTOXIC_KEY].astype(float).to_numpy()[target_idx], 'Tcell_IFN_score': ifn_score[target_idx], 'distance_to_MP2_high': np.full(n_target, np.nan)}
for r in RADII:
    result[f'local_MP2_burden_{r}'] = np.zeros(n_target)
    result[f'local_MP3_burden_{r}'] = np.zeros(n_target)
    result[f'local_MP2_mean_{r}'] = np.full(n_target, np.nan)
    result[f'local_MP3_mean_{r}'] = np.full(n_target, np.nan)
    result[f'malignant_density_{r}'] = np.zeros(n_target)
    result[f'Tcell_density_{r}'] = np.zeros(n_target)
    result[f'myeloid_density_{r}'] = np.zeros(n_target)
    result[f'myeloid_IFN_burden_{r}'] = np.zeros(n_target)


# ============================================================
# 11. Utility functions
# ============================================================
def density_from_count(count, radius):
    area_um2 = np.pi * radius ** 2
    return count * 1000000.0 / area_um2


# ============================================================
# 12. Calculate neighborhoods by core and chunk
# ============================================================
max_radius = max(RADII)
target_core = core[target_idx]
for c in np.unique(target_core):
    print('\nProcessing:', c)
    target_pos_core = np.where(target_core == c)[0]
    mal_idx = np.where((core == c) & is_malignant)[0]
    t_idx = np.where((core == c) & is_tcell)[0]
    my_idx = np.where((core == c) & is_myeloid)[0]
    mp2high_idx = np.where((core == c) & mp2_high)[0]
    if len(mal_idx) == 0:
        continue
    mal_tree = cKDTree(xy[mal_idx])
    t_tree = cKDTree(xy[t_idx]) if len(t_idx) > 0 else None
    my_tree = cKDTree(xy[my_idx]) if len(my_idx) > 0 else None
    mp2high_tree = cKDTree(xy[mp2high_idx]) if len(mp2high_idx) > 0 else None
    mal_mp2 = np.nan_to_num(mp2_z[mal_idx], nan=0)
    mal_mp3 = np.nan_to_num(mp3_z[mal_idx], nan=0)
    mal_mp2_positive = np.clip(mal_mp2, 0, None)
    mal_mp3_positive = np.clip(mal_mp3, 0, None)
    for start in range(0, len(target_pos_core), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(target_pos_core))
        local_positions = target_pos_core[start:stop]
        global_indices = target_idx[local_positions]
        target_xy = xy[global_indices]
        n_chunk = len(global_indices)
        if mp2high_tree is not None:
            d, _ = mp2high_tree.query(target_xy, k=1)
            result['distance_to_MP2_high'][local_positions] = d
        target_tree = cKDTree(target_xy)
        dm = target_tree.sparse_distance_matrix(mal_tree, max_distance=max_radius, output_type='coo_matrix')
        rows = dm.row
        cols = dm.col
        dist = dm.data
        for r in RADII:
            keep = dist <= r
            rr = rows[keep]
            cc = cols[keep]
            count = np.bincount(rr, minlength=n_chunk)
            mp2_sum = np.bincount(rr, weights=mal_mp2_positive[cc], minlength=n_chunk)
            mp3_sum = np.bincount(rr, weights=mal_mp3_positive[cc], minlength=n_chunk)
            mp2_total = np.bincount(rr, weights=mal_mp2[cc], minlength=n_chunk)
            mp3_total = np.bincount(rr, weights=mal_mp3[cc], minlength=n_chunk)
            result[f'local_MP2_burden_{r}'][local_positions] = density_from_count(mp2_sum, r)
            result[f'local_MP3_burden_{r}'][local_positions] = density_from_count(mp3_sum, r)
            result[f'malignant_density_{r}'][local_positions] = density_from_count(count, r)
            valid = count > 0
            mp2_mean = np.full(n_chunk, np.nan)
            mp3_mean = np.full(n_chunk, np.nan)
            mp2_mean[valid] = mp2_total[valid] / count[valid]
            mp3_mean[valid] = mp3_total[valid] / count[valid]
            result[f'local_MP2_mean_{r}'][local_positions] = mp2_mean
            result[f'local_MP3_mean_{r}'][local_positions] = mp3_mean
        if t_tree is not None:
            dt = target_tree.sparse_distance_matrix(t_tree, max_distance=max_radius, output_type='coo_matrix')
            trows = dt.row
            tdist = dt.data
            non_self = tdist > 1e-08
            trows = trows[non_self]
            tdist = tdist[non_self]
            for r in RADII:
                keep = tdist <= r
                count = np.bincount(trows[keep], minlength=n_chunk)
                result[f'Tcell_density_{r}'][local_positions] = density_from_count(count, r)
        if my_tree is not None:
            dy = target_tree.sparse_distance_matrix(my_tree, max_distance=max_radius, output_type='coo_matrix')
            yrows = dy.row
            ycols = dy.col
            ydist = dy.data
            my_ifn = ifn_score[my_idx]
            for r in RADII:
                keep = ydist <= r
                rr = yrows[keep]
                cc = ycols[keep]
                count = np.bincount(rr, minlength=n_chunk)
                ifn_sum = np.bincount(rr, weights=my_ifn[cc], minlength=n_chunk)
                result[f'myeloid_density_{r}'][local_positions] = density_from_count(count, r)
                result[f'myeloid_IFN_burden_{r}'][local_positions] = density_from_count(ifn_sum, r)


# ============================================================
# 13. Create the T-cell-level table
# ============================================================
tcell_state = pd.DataFrame(result)
tcell_state['is_CD8'] = tcell_state['celltype'].isin(['Cytotoxic_T', 'Exhausted_T'])
tcell_state['is_Treg'] = tcell_state['celltype'] == 'Treg'
tcell_state['is_Exhausted_identity'] = tcell_state['celltype'] == 'Exhausted_T'
tcell_state['patient_core'] = tcell_state['patient'].astype(str) + '_' + tcell_state['core_id'].astype(str)


# ============================================================
# 14. Add checkpoint-gene expression
# ============================================================
CHECKPOINT_GENES = ['PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'TOX']
for gene in CHECKPOINT_GENES:
    if gene not in gene_pool:
        print(gene, 'not in panel')
        continue
    vec = get_gene_vector(gene)
    tcell_state[gene] = vec[target_idx]


# ============================================================
# 15. Within-core standardization
# ============================================================
def group_zscore(df, column, group='core_id'):

    def _z(x):
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / sd
    return df.groupby(group, observed=True)[column].transform(_z)


# ============================================================
# 16. Create model variables for each radius
# ============================================================
for r in RADII:
    for variable in [f'local_MP2_burden_{r}', f'local_MP3_burden_{r}', f'malignant_density_{r}', f'Tcell_density_{r}', f'myeloid_density_{r}', f'myeloid_IFN_burden_{r}']:
        tcell_state[variable + '_z'] = group_zscore(tcell_state, variable)


# ============================================================
# 17. Standardize CD8 and Treg outcomes within each core
# ============================================================
cd8 = tcell_state[tcell_state['is_CD8']].copy()
treg = tcell_state[tcell_state['is_Treg']].copy()
cd8['Exhausted_score_z'] = group_zscore(cd8, 'Exhausted_score')
cd8['Cytotoxic_score_z'] = group_zscore(cd8, 'Cytotoxic_score')
cd8['Tcell_IFN_score_z'] = group_zscore(cd8, 'Tcell_IFN_score')
treg['Treg_score_z'] = group_zscore(treg, 'Treg_score')
treg['Tcell_IFN_score_z'] = group_zscore(treg, 'Tcell_IFN_score')


# ============================================================
# 18. Balance sampling across cores
# ============================================================
def balanced_sample(df, max_per_core, seed=1234):
    parts = []
    for core_id, temp in df.groupby('core_id', observed=True):
        if len(temp) > max_per_core:
            temp = temp.sample(max_per_core, random_state=seed)
        parts.append(temp)
    return pd.concat(parts, ignore_index=True)
cd8_model = balanced_sample(cd8, MAX_MODEL_CELLS_PER_CORE)
treg_model = balanced_sample(treg, MAX_MODEL_CELLS_PER_CORE)


# ============================================================
# 19. Fit mixed-effects models
# ============================================================
def fit_state_mixed_model(df, outcome, radius):
    mp2 = f'local_MP2_burden_{radius}_z'
    mp3 = f'local_MP3_burden_{radius}_z'
    malignant = f'malignant_density_{radius}_z'
    tdensity = f'Tcell_density_{radius}_z'
    myeloid = f'myeloid_density_{radius}_z'
    ifn = f'myeloid_IFN_burden_{radius}_z'
    formula = f'{outcome} ~ {mp2} + {mp3} + {mp2}:{mp3} + {malignant} + {tdensity} + {myeloid} + {ifn} + Tcell_IFN_score_z + C(cellcharter_cluster)'
    needed = [outcome, mp2, mp3, malignant, tdensity, myeloid, ifn, 'Tcell_IFN_score_z', 'patient', 'patient_core', 'cellcharter_cluster']
    dat = df[needed].dropna().copy()
    print('\nFitting:', outcome, 'radius:', radius, 'N:', len(dat))
    try:
        model = smf.mixedlm(formula, data=dat, groups='patient', re_formula='1', vc_formula={'core': '0 + C(patient_core)'})
        fit = model.fit(reml=False, method='lbfgs', maxiter=300, disp=False)
    except Exception as e:
        print('Nested model failed:', e)
        print('Fallback to core random intercept')
        model = smf.mixedlm(formula, data=dat, groups='patient_core')
        fit = model.fit(reml=False, method='lbfgs', maxiter=300, disp=False)
    return (fit, formula, len(dat))


# ============================================================
# 20. Extract model coefficients
# ============================================================
def extract_model_effects(fit, outcome, radius, n_cells):
    rows = []
    ci = fit.conf_int()
    for term in fit.fe_params.index:
        rows.append({'outcome': outcome, 'radius': radius, 'term': term, 'beta': fit.fe_params[term], 'SE': fit.bse[term], 'CI_low': ci.loc[term, 0], 'CI_high': ci.loc[term, 1], 'p_value': fit.pvalues[term], 'n_cells': n_cells})
    return pd.DataFrame(rows)


# ============================================================
# 21. Fit primary models at 50 micrometers
# ============================================================
model_effects = []
fit_exh, _, n_exh = fit_state_mixed_model(cd8_model, 'Exhausted_score_z', PRIMARY_RADIUS)
model_effects.append(extract_model_effects(fit_exh, 'Exhausted_score', PRIMARY_RADIUS, n_exh))
fit_cyt, _, n_cyt = fit_state_mixed_model(cd8_model, 'Cytotoxic_score_z', PRIMARY_RADIUS)
model_effects.append(extract_model_effects(fit_cyt, 'Cytotoxic_score', PRIMARY_RADIUS, n_cyt))
fit_treg, _, n_treg = fit_state_mixed_model(treg_model, 'Treg_score_z', PRIMARY_RADIUS)
model_effects.append(extract_model_effects(fit_treg, 'Treg_score', PRIMARY_RADIUS, n_treg))
state_effects = pd.concat(model_effects, ignore_index=True)
state_effects.to_csv(RESULT_DIR / 'state_mixed_model_primary_50um.csv', index=False)
print('\n===== PRIMARY MODEL =====')
primary_terms = state_effects[state_effects['term'].str.contains('MP2|MP3')]
print(primary_terms[['outcome', 'term', 'beta', 'CI_low', 'CI_high', 'p_value']].to_string(index=False))


# ============================================================
# 22. Run spatial-scale sensitivity models
# ============================================================
sensitivity = []
for r in RADII:
    fit, _, n = fit_state_mixed_model(cd8_model, 'Exhausted_score_z', r)
    sensitivity.append(extract_model_effects(fit, 'Exhausted_score', r, n))
    fit, _, n = fit_state_mixed_model(cd8_model, 'Cytotoxic_score_z', r)
    sensitivity.append(extract_model_effects(fit, 'Cytotoxic_score', r, n))
    fit, _, n = fit_state_mixed_model(treg_model, 'Treg_score_z', r)
    sensitivity.append(extract_model_effects(fit, 'Treg_score', r, n))
sensitivity = pd.concat(sensitivity, ignore_index=True)
sensitivity.to_csv(RESULT_DIR / 'state_model_radius_sensitivity.csv', index=False)


# ============================================================
# 23. Plot MP2 effects across spatial scales
# ============================================================
mp2_sensitivity = sensitivity[sensitivity['term'].str.startswith('local_MP2_burden_') & ~sensitivity['term'].str.contains(':')].copy()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
fig, ax = plt.subplots(figsize=(6, 4.5))
for outcome in ['Exhausted_score', 'Treg_score', 'Cytotoxic_score']:
    temp = mp2_sensitivity[mp2_sensitivity['outcome'] == outcome].sort_values('radius')
    ax.errorbar(temp['radius'], temp['beta'], yerr=[temp['beta'] - temp['CI_low'], temp['CI_high'] - temp['beta']], marker='o', capsize=3, label=outcome)
ax.axhline(0, linestyle='--', linewidth=0.8)
ax.set_xlabel('Neighborhood radius (µm)')
ax.set_ylabel('Adjusted MP2 effect (β)')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_DIR / 'MP2_effect_across_spatial_scales.pdf', bbox_inches='tight')
plt.close()


# ============================================================
# 24. Create distance bins from MP2-high malignant cells
# ============================================================
distance_bins = [0, 20, 30, 50, 75, 100, 150, np.inf]
distance_labels = ['0-20', '20-30', '30-50', '50-75', '75-100', '100-150', '>150']
cd8['distance_bin'] = pd.cut(cd8['distance_to_MP2_high'], bins=distance_bins, labels=distance_labels, right=False)
treg['distance_bin'] = pd.cut(treg['distance_to_MP2_high'], bins=distance_bins, labels=distance_labels, right=False)


# ============================================================
# 25. Summarize distance gradients by patient
# ============================================================
def make_patient_bin(df, outcome):
    return df.dropna(subset=['distance_bin', outcome]).groupby(['patient', 'distance_bin'], observed=True).agg(median_score=(outcome, 'median'), n_cells=(outcome, 'size')).reset_index()
distance_exhaustion = make_patient_bin(cd8, 'Exhausted_score')
distance_cytotoxic = make_patient_bin(cd8, 'Cytotoxic_score')
distance_treg = make_patient_bin(treg, 'Treg_score')
distance_exhaustion.to_csv(RESULT_DIR / 'distance_MP2_CD8_exhaustion.csv', index=False)
distance_cytotoxic.to_csv(RESULT_DIR / 'distance_MP2_CD8_cytotoxicity.csv', index=False)
distance_treg.to_csv(RESULT_DIR / 'distance_MP2_Treg_state.csv', index=False)


# ============================================================
# 26. Plot distance gradients
# ============================================================
def plot_distance_gradient(df, ylabel, filename):
    order = distance_labels
    summary = df.groupby('distance_bin', observed=True).agg(mean=('median_score', 'mean'), sem=('median_score', lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()))).reindex(order)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for p in df['patient'].unique():
        temp = df[df['patient'] == p].set_index('distance_bin').reindex(order)
        ax.plot(x, temp['median_score'], linewidth=0.7, alpha=0.25)
    ax.errorbar(x, summary['mean'], yerr=summary['sem'], marker='o', linewidth=2, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_xlabel('Distance to nearest MP2-high malignant cell (µm)')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches='tight')
    plt.close()
plot_distance_gradient(distance_exhaustion, 'CD8 T-cell exhaustion score', 'distance_MP2_CD8_exhaustion.pdf')
plot_distance_gradient(distance_cytotoxic, 'CD8 T-cell cytotoxicity score', 'distance_MP2_CD8_cytotoxicity.pdf')
plot_distance_gradient(distance_treg, 'Treg functional score', 'distance_MP2_Treg_score.pdf')


# ============================================================
# 27. Summarize checkpoint expression by distance
# ============================================================
checkpoint_summaries = []
for gene in CHECKPOINT_GENES:
    if gene not in cd8.columns:
        continue
    temp = cd8.dropna(subset=['distance_bin', gene]).groupby(['patient', 'distance_bin'], observed=True).agg(median_expression=(gene, 'median')).reset_index()
    temp['gene'] = gene
    checkpoint_summaries.append(temp)
if checkpoint_summaries:
    checkpoint_summaries = pd.concat(checkpoint_summaries, ignore_index=True)
    checkpoint_summaries.to_csv(RESULT_DIR / 'checkpoint_expression_by_MP2_distance.csv', index=False)


# ============================================================
# 28. Model binary exhausted-cell identity
# ============================================================
r = PRIMARY_RADIUS
mp2 = f'local_MP2_burden_{r}_z'
mp3 = f'local_MP3_burden_{r}_z'
malignant = f'malignant_density_{r}_z'
tdensity = f'Tcell_density_{r}_z'
myeloid = f'myeloid_density_{r}_z'
ifn = f'myeloid_IFN_burden_{r}_z'
binary_formula = f'is_Exhausted_identity ~ {mp2} + {mp3} + {mp2}:{mp3} + {malignant} + {tdensity} + {myeloid} + {ifn} + Tcell_IFN_score_z + C(cellcharter_cluster)'
binary_dat = cd8_model.dropna(subset=[mp2, mp3, malignant, tdensity, myeloid, ifn]).copy()
binary_dat['is_Exhausted_identity'] = (binary_dat['celltype'] == 'Exhausted_T').astype(int)
gee_binary = smf.gee(binary_formula, groups='patient_core', data=binary_dat, family=sm.families.Binomial()).fit()
binary_effects = pd.DataFrame({'term': gee_binary.params.index, 'beta': gee_binary.params.values, 'SE': gee_binary.bse.values, 'p_value': gee_binary.pvalues.values})
binary_effects['OR'] = np.exp(binary_effects['beta'])
binary_effects.to_csv(RESULT_DIR / 'Exhausted_identity_GEE_logistic.csv', index=False)


# ============================================================
# 29. Calculate abundance outcomes around malignant anchors
# ============================================================
anchor_rows = []
for c in np.unique(core):
    malignant_idx = np.where((core == c) & is_malignant)[0]
    if len(malignant_idx) == 0:
        continue
    if len(malignant_idx) > MAX_ANCHORS_PER_CORE:
        malignant_idx = rng.choice(malignant_idx, size=MAX_ANCHORS_PER_CORE, replace=False)
    anchor_xy = xy[malignant_idx]
    cd8_idx = np.where((core == c) & is_cd8)[0]
    exhausted_idx = np.where((core == c) & is_exhausted_identity)[0]
    treg_idx = np.where((core == c) & is_treg)[0]
    my_idx = np.where((core == c) & is_myeloid)[0]

    def query_count(source_idx):
        if len(source_idx) == 0:
            return np.zeros(len(anchor_xy), dtype=int)
        tree = cKDTree(xy[source_idx])
        neighbors = tree.query_ball_point(anchor_xy, r=PRIMARY_RADIUS)
        return np.fromiter((len(x) for x in neighbors), dtype=int)
    cd8_count = query_count(cd8_idx)
    exhausted_count = query_count(exhausted_idx)
    treg_count = query_count(treg_idx)
    my_count = query_count(my_idx)
    temp = pd.DataFrame({'patient': patient[malignant_idx], 'core_id': core[malignant_idx], 'patient_core': patient[malignant_idx].astype(str) + '_' + core[malignant_idx].astype(str), 'cellcharter_cluster': region[malignant_idx], 'MP2_z': mp2_z[malignant_idx], 'MP3_z': mp3_z[malignant_idx], 'CD8_count_50': cd8_count, 'Exhausted_count_50': exhausted_count, 'Treg_count_50': treg_count, 'myeloid_density_50': density_from_count(my_count, PRIMARY_RADIUS)})
    anchor_rows.append(temp)
anchor_df = pd.concat(anchor_rows, ignore_index=True)
anchor_df['myeloid_density_50_z'] = group_zscore(anchor_df, 'myeloid_density_50')
anchor_df.to_csv(RESULT_DIR / 'malignant_anchor_local_Tcell_counts.csv', index=False)


# ============================================================
# 30. Fit abundance models
# ============================================================
def fit_count_gee(outcome):
    formula = f'{outcome} ~ MP2_z + MP3_z + MP2_z:MP3_z + myeloid_density_50_z + C(cellcharter_cluster)'
    fit = smf.gee(formula, groups='patient_core', data=anchor_df, family=sm.families.Poisson()).fit()
    out = pd.DataFrame({'outcome': outcome, 'term': fit.params.index, 'beta': fit.params.values, 'SE': fit.bse.values, 'p_value': fit.pvalues.values})
    out['rate_ratio'] = np.exp(out['beta'])
    return out
count_effects = pd.concat([fit_count_gee('CD8_count_50'), fit_count_gee('Exhausted_count_50'), fit_count_gee('Treg_count_50')], ignore_index=True)
count_effects.to_csv(RESULT_DIR / 'Tcell_abundance_GEE_models.csv', index=False)


# ============================================================
# 31. Summarize quantity and state effects
# ============================================================
quantity_summary = count_effects[count_effects['term'].isin(['MP2_z', 'MP3_z'])].copy()
state_summary = state_effects[(state_effects['radius'] == PRIMARY_RADIUS) & state_effects['term'].isin(['local_MP2_burden_50_z', 'local_MP3_burden_50_z'])].copy()
quantity_summary.to_csv(RESULT_DIR / 'summary_MP2_MP3_quantity_effect.csv', index=False)
state_summary.to_csv(RESULT_DIR / 'summary_MP2_MP3_state_effect.csv', index=False)


# ============================================================
# 32. Save T-cell spatial metrics and AnnData
# ============================================================
tcell_state.to_csv(RESULT_DIR / 'Tcell_local_MP2_MP3_environment.csv', index=False)
distance_all = np.full(adata.n_obs, np.nan)
distance_all[target_idx] = tcell_state['distance_to_MP2_high'].to_numpy()
adata.obs['distance_to_MP2_high'] = distance_all
adata.write_h5ad(OUT_DIR / 'xenium_roi_MP2_Tcell_functional_spatial.h5ad', compression='gzip')
print('\nDONE')
print('Results:', RESULT_DIR)
