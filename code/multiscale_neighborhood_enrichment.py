# Continuous MP-score multiscale spatial-neighborhood enrichment
# Generated from the original notebook while preserving its analysis structure.

from pathlib import Path
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')


# ============================================================
# 1. Paths
# ============================================================
DATA_DIR = Path('/home/zyf/NPC/summary/output-XETG00149__0104093__Region_1__20260612__101530')
INFILE = DATA_DIR / 'processed' / 'xenium_roi_celltype_annotated_cNMF_scored.h5ad'
OUT_DIR = DATA_DIR / 'processed' / 'continuous_MP_multiscale_neighborhood'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Data columns
# ============================================================
GROUP_COL = 'core_id'
CELLTYPE_COL = 'celltype'
X_COL = 'x_centroid'
Y_COL = 'y_centroid'
SCORE_INFO = {'MP1': 'MP1_cNMF_score', 'MP2': 'MP2_cNMF_score', 'MP3': 'MP3_cNMF_score', 'MP4': 'MP4_cNMF_score'}
MP_ORDER = ['MP1', 'MP2', 'MP3', 'MP4']
TARGET_CELLTYPES = {'Treg': ['Treg'], 'Exhausted_T': ['Exhausted_T']}


# ============================================================
# 3. Multiscale neighborhood parameters
# ============================================================
RADIUS_LIST = [20, 30, 50, 75, 100]
NEIGHBOR_METRIC = 'target_fraction'
MIN_TOTAL_NEIGHBORS = 10
MIN_MALIGNANT_CELLS = 30
MIN_TARGET_CELLS = 3
N_PERMUTATIONS = 1000
RANDOM_SEED = 12345
SAVE_CELL_LEVEL = True


# ============================================================
# 4. PDF settings
# ============================================================
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.compression'] = 9
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


# ============================================================
# 5. Load data
# ============================================================
print('Reading:')
print(INFILE)
adata = sc.read_h5ad(INFILE)
print('\nAnnData:')
print(adata)


# ============================================================
# 6. Validate required columns
# ============================================================
required_columns = [GROUP_COL, CELLTYPE_COL, X_COL, Y_COL, *SCORE_INFO.values()]
missing_columns = [col for col in required_columns if col not in adata.obs.columns]
if missing_columns:
    raise KeyError(f'adata.obs is missing required columns: {missing_columns}\n\nAvailable columns:\n{adata.obs.columns.tolist()}')


# ============================================================
# 7. Prepare observation metadata
# ============================================================
obs = adata.obs.copy()
obs[GROUP_COL] = obs[GROUP_COL].astype(str)
obs[CELLTYPE_COL] = obs[CELLTYPE_COL].astype(str)
obs[X_COL] = pd.to_numeric(obs[X_COL], errors='coerce')
obs[Y_COL] = pd.to_numeric(obs[Y_COL], errors='coerce')
for score_col in SCORE_INFO.values():
    obs[score_col] = pd.to_numeric(obs[score_col], errors='coerce')
score_columns = list(SCORE_INFO.values())


# ============================================================
# 8. Validate target-cell labels
# ============================================================
print('\nTarget-cell label matches:')
for target_name, target_labels in TARGET_CELLTYPES.items():
    n_target = obs[CELLTYPE_COL].isin(target_labels).sum()
    print(f'{target_name}: {target_labels}; cell count={n_target}')
    if n_target == 0:
        print('\nAvailable cell-type labels:')
        print(obs[CELLTYPE_COL].value_counts().to_string())
        raise ValueError(f'No cells matched {target_name}, ; update the labels in TARGET_CELLTYPES')


# ============================================================
# 9. Identify malignant cells
# ============================================================
malignant_mask = obs[score_columns].notna().any(axis=1)
print('\nMalignant cells with cNMF scores:', malignant_mask.sum())
if malignant_mask.sum() == 0:
    raise ValueError('No malignant cells with cNMF scores were found')


# ============================================================
# 10. Count neighbors within a radius
# ============================================================
def count_neighbors(tree, query_coordinates, radius):
    """
    Count neighbors around each query coordinate within the specified radius.
    Use return_length when available and fall back for older SciPy versions.
    """
    try:
        counts = tree.query_ball_point(query_coordinates, r=radius, return_length=True)
        return np.asarray(counts, dtype=int)
    except TypeError:
        neighbor_lists = tree.query_ball_point(query_coordinates, r=radius)
        return np.fromiter((len(neighbors) for neighbors in neighbor_lists), dtype=int, count=len(neighbor_lists))


# ============================================================
# 11. Calculate Spearman rho from standardized ranks
# ============================================================
def spearman_with_rank_z(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < 3:
        return (np.nan, None, None, valid)
    x_rank = rankdata(x, method='average')
    y_rank = rankdata(y, method='average')
    x_sd = np.std(x_rank, ddof=1)
    y_sd = np.std(y_rank, ddof=1)
    if x_sd == 0 or y_sd == 0:
        return (np.nan, None, None, valid)
    x_rank_z = (x_rank - np.mean(x_rank)) / x_sd
    y_rank_z = (y_rank - np.mean(y_rank)) / y_sd
    rho = float(np.dot(x_rank_z, y_rank_z) / (n - 1))
    return (rho, x_rank_z, y_rank_z, valid)


# ============================================================
# 12. Safe Wilcoxon tests
# ============================================================
def safe_one_sample_wilcoxon(values, alternative='greater'):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return (np.nan, np.nan)
    if np.allclose(values, 0):
        return (0.0, 1.0)
    try:
        result = wilcoxon(values, alternative=alternative, zero_method='wilcox')
        return (float(result.statistic), float(result.pvalue))
    except ValueError:
        return (np.nan, np.nan)

def safe_paired_wilcoxon(x, y, alternative='two-sided'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return (np.nan, np.nan)
    if np.allclose(x - y, 0):
        return (0.0, 1.0)
    try:
        result = wilcoxon(x, y, alternative=alternative, zero_method='wilcox')
        return (float(result.statistic), float(result.pvalue))
    except ValueError:
        return (np.nan, np.nan)


# ============================================================
# 13. Run multiscale neighborhood analysis by core
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)
core_result_records = []
cell_level_records = []
core_ids = sorted(obs.loc[obs[GROUP_COL].notna(), GROUP_COL].unique().tolist())
print('\nNumber of cores to analyze:', len(core_ids))
print(core_ids)
for core_number, core_id in enumerate(core_ids, start=1):
    print(f'\n[{core_number}/{len(core_ids)}] Processing: {core_id}')
    core_obs = obs.loc[obs[GROUP_COL].eq(core_id)].copy()
    coordinate_valid = core_obs[X_COL].notna() & core_obs[Y_COL].notna()
    valid_core_obs = core_obs.loc[coordinate_valid].copy()
    all_coordinates = valid_core_obs[[X_COL, Y_COL]].to_numpy(dtype=float)
    all_cell_tree = cKDTree(all_coordinates)
    malignant_core = valid_core_obs.loc[valid_core_obs[score_columns].notna().any(axis=1)].copy()
    n_malignant = len(malignant_core)
    print('    Valid malignant cells:', n_malignant)
    if n_malignant < MIN_MALIGNANT_CELLS:
        for target_name in TARGET_CELLTYPES:
            for radius in RADIUS_LIST:
                for mp_name in MP_ORDER:
                    core_result_records.append({GROUP_COL: core_id, 'target': target_name, 'radius_um': radius, 'MP': mp_name, 'n_malignant_cells': n_malignant, 'n_target_cells': np.nan, 'n_cells_used': np.nan, 'spearman_rho': np.nan, 'permutation_mean_rho': np.nan, 'permutation_sd_rho': np.nan, 'permutation_Z': np.nan, 'p_enrichment': np.nan, 'p_depletion': np.nan, 'p_two_sided': np.nan, 'status': 'too_few_malignant_cells'})
        continue
    malignant_coordinates = malignant_core[[X_COL, Y_COL]].to_numpy(dtype=float)
    total_neighbor_counts = {}
    for radius in RADIUS_LIST:
        counts = count_neighbors(tree=all_cell_tree, query_coordinates=malignant_coordinates, radius=radius)
        counts = np.maximum(counts - 1, 0)
        total_neighbor_counts[radius] = counts
    for target_name, target_labels in TARGET_CELLTYPES.items():
        target_obs = valid_core_obs.loc[valid_core_obs[CELLTYPE_COL].isin(target_labels)].copy()
        n_target = len(target_obs)
        print(f'  {target_name} cells:{n_target}')
        if n_target < MIN_TARGET_CELLS:
            for radius in RADIUS_LIST:
                for mp_name in MP_ORDER:
                    core_result_records.append({GROUP_COL: core_id, 'target': target_name, 'radius_um': radius, 'MP': mp_name, 'n_malignant_cells': n_malignant, 'n_target_cells': n_target, 'n_cells_used': np.nan, 'spearman_rho': np.nan, 'permutation_mean_rho': np.nan, 'permutation_sd_rho': np.nan, 'permutation_Z': np.nan, 'p_enrichment': np.nan, 'p_depletion': np.nan, 'p_two_sided': np.nan, 'status': 'too_few_target_cells'})
            continue
        target_coordinates = target_obs[[X_COL, Y_COL]].to_numpy(dtype=float)
        target_tree = cKDTree(target_coordinates)
        for radius in RADIUS_LIST:
            target_counts = count_neighbors(tree=target_tree, query_coordinates=malignant_coordinates, radius=radius)
            total_counts = total_neighbor_counts[radius]
            circle_area = np.pi * radius * radius
            target_density = target_counts / circle_area
            target_fraction = np.divide(target_counts, total_counts, out=np.full(len(target_counts), np.nan, dtype=float), where=total_counts >= MIN_TOTAL_NEIGHBORS)
            if NEIGHBOR_METRIC == 'target_fraction':
                neighborhood_values = target_fraction
            elif NEIGHBOR_METRIC == 'target_count':
                neighborhood_values = target_counts.astype(float)
                neighborhood_values[total_counts < MIN_TOTAL_NEIGHBORS] = np.nan
            elif NEIGHBOR_METRIC == 'target_density':
                neighborhood_values = target_density.astype(float)
                neighborhood_values[total_counts < MIN_TOTAL_NEIGHBORS] = np.nan
            else:
                raise ValueError("NEIGHBOR_METRIC must be one of: 'target_fraction', 'target_count''target_density'")
            if SAVE_CELL_LEVEL:
                for cell_position, cell_id in enumerate(malignant_core.index):
                    cell_level_records.append({GROUP_COL: core_id, 'target': target_name, 'radius_um': radius, 'malignant_cell_id': cell_id, 'target_count': int(target_counts[cell_position]), 'total_neighbor_count': int(total_counts[cell_position]), 'target_fraction': target_fraction[cell_position], 'target_density_per_um2': target_density[cell_position], **{score_col: malignant_core.loc[cell_id, score_col] for score_col in score_columns}})
            for mp_name in MP_ORDER:
                score_col = SCORE_INFO[mp_name]
                scores = malignant_core[score_col].to_numpy(dtype=float)
                observed_rho, score_rank_z, neighborhood_rank_z, valid_mask = spearman_with_rank_z(scores, neighborhood_values)
                n_cells_used = int(valid_mask.sum())
                if n_cells_used < MIN_MALIGNANT_CELLS:
                    core_result_records.append({GROUP_COL: core_id, 'target': target_name, 'radius_um': radius, 'MP': mp_name, 'score_column': score_col, 'neighbor_metric': NEIGHBOR_METRIC, 'n_malignant_cells': n_malignant, 'n_target_cells': n_target, 'n_cells_used': n_cells_used, 'spearman_rho': np.nan, 'permutation_mean_rho': np.nan, 'permutation_sd_rho': np.nan, 'permutation_Z': np.nan, 'p_enrichment': np.nan, 'p_depletion': np.nan, 'p_two_sided': np.nan, 'median_target_count': np.nan, 'median_target_fraction': np.nan, 'status': 'too_few_valid_cells'})
                    continue
                if not np.isfinite(observed_rho):
                    core_result_records.append({GROUP_COL: core_id, 'target': target_name, 'radius_um': radius, 'MP': mp_name, 'score_column': score_col, 'neighbor_metric': NEIGHBOR_METRIC, 'n_malignant_cells': n_malignant, 'n_target_cells': n_target, 'n_cells_used': n_cells_used, 'spearman_rho': np.nan, 'permutation_mean_rho': np.nan, 'permutation_sd_rho': np.nan, 'permutation_Z': np.nan, 'p_enrichment': np.nan, 'p_depletion': np.nan, 'p_two_sided': np.nan, 'median_target_count': np.nan, 'median_target_fraction': np.nan, 'status': 'zero_variance'})
                    continue
                permutation_rhos = np.empty(N_PERMUTATIONS, dtype=float)
                n_valid = len(score_rank_z)
                for permutation_index in range(N_PERMUTATIONS):
                    permuted_score_rank_z = rng.permutation(score_rank_z)
                    permutation_rhos[permutation_index] = np.dot(permuted_score_rank_z, neighborhood_rank_z) / (n_valid - 1)
                permutation_mean = float(np.mean(permutation_rhos))
                permutation_sd = float(np.std(permutation_rhos, ddof=1))
                if np.isfinite(permutation_sd) and permutation_sd > 0:
                    permutation_z = (observed_rho - permutation_mean) / permutation_sd
                else:
                    permutation_z = np.nan
                p_enrichment = (1 + np.sum(permutation_rhos >= observed_rho)) / (N_PERMUTATIONS + 1)
                p_depletion = (1 + np.sum(permutation_rhos <= observed_rho)) / (N_PERMUTATIONS + 1)
                p_two_sided = min(1.0, 2.0 * min(p_enrichment, p_depletion))
                core_result_records.append({GROUP_COL: core_id, 'target': target_name, 'radius_um': radius, 'MP': mp_name, 'score_column': score_col, 'neighbor_metric': NEIGHBOR_METRIC, 'n_malignant_cells': n_malignant, 'n_target_cells': n_target, 'n_cells_used': n_cells_used, 'spearman_rho': observed_rho, 'permutation_mean_rho': permutation_mean, 'permutation_sd_rho': permutation_sd, 'permutation_Z': permutation_z, 'p_enrichment': p_enrichment, 'p_depletion': p_depletion, 'p_two_sided': p_two_sided, 'median_target_count': float(np.median(target_counts[valid_mask])), 'median_target_fraction': float(np.nanmedian(target_fraction[valid_mask])), 'status': 'ok'})
                print(f'    {radius:>3} μm {mp_name}: rho={observed_rho:.4f}, Z={permutation_z:.2f}, P={p_enrichment:.4g}')


# ============================================================
# 14. Prepare core-level results
# ============================================================
result_df = pd.DataFrame(core_result_records)
if len(result_df) == 0:
    raise ValueError('No results were generated; verify the data columns and cell-type labels')
result_df['p_enrichment_fdr'] = np.nan
result_df['p_two_sided_fdr'] = np.nan
for target_name in TARGET_CELLTYPES:
    for radius in RADIUS_LIST:
        valid_p_mask = result_df['target'].eq(target_name) & result_df['radius_um'].eq(radius) & result_df['status'].eq('ok') & result_df['p_enrichment'].notna()
        if valid_p_mask.sum() == 0:
            continue
        result_df.loc[valid_p_mask, 'p_enrichment_fdr'] = multipletests(result_df.loc[valid_p_mask, 'p_enrichment'], method='fdr_bh')[1]
        result_df.loc[valid_p_mask, 'p_two_sided_fdr'] = multipletests(result_df.loc[valid_p_mask, 'p_two_sided'], method='fdr_bh')[1]
CORE_RESULT_FILE = OUT_DIR / 'continuous_MP_multiscale_core_results.csv'
result_df.to_csv(CORE_RESULT_FILE, index=False)
print('\nCore-level results saved to:', CORE_RESULT_FILE)
valid_result = result_df.loc[result_df['status'].eq('ok')].copy()


# ============================================================
# 15. Save cell-level neighborhood results
# ============================================================
if SAVE_CELL_LEVEL:
    cell_level_df = pd.DataFrame(cell_level_records)
    CELL_LEVEL_FILE = OUT_DIR / 'continuous_MP_multiscale_cell_neighborhoods.csv.gz'
    cell_level_df.to_csv(CELL_LEVEL_FILE, index=False, compression='gzip')
    print('Cell-level neighborhood results saved to:', CELL_LEVEL_FILE)


# ============================================================
# 16. Summarize results across cores
# ============================================================
summary_df = valid_result.groupby(['target', 'radius_um', 'MP'], observed=True).agg(n_cores=(GROUP_COL, 'nunique'), mean_spearman_rho=('spearman_rho', 'mean'), median_spearman_rho=('spearman_rho', 'median'), q25_spearman_rho=('spearman_rho', lambda x: x.quantile(0.25)), q75_spearman_rho=('spearman_rho', lambda x: x.quantile(0.75)), mean_permutation_Z=('permutation_Z', 'mean'), median_permutation_Z=('permutation_Z', 'median')).reset_index()
SUMMARY_FILE = OUT_DIR / 'continuous_MP_multiscale_summary.csv'
summary_df.to_csv(SUMMARY_FILE, index=False)


# ============================================================
# 17. Test enrichment for each metaprogram
# ============================================================
one_sample_records = []
for target_name in TARGET_CELLTYPES:
    for radius in RADIUS_LIST:
        for mp_name in MP_ORDER:
            values = valid_result.loc[valid_result['target'].eq(target_name) & valid_result['radius_um'].eq(radius) & valid_result['MP'].eq(mp_name), 'spearman_rho'].dropna().to_numpy()
            statistic, p_value = safe_one_sample_wilcoxon(values, alternative='greater')
            one_sample_records.append({'target': target_name, 'radius_um': radius, 'MP': mp_name, 'n_cores': len(values), 'mean_rho': np.mean(values) if len(values) > 0 else np.nan, 'median_rho': np.median(values) if len(values) > 0 else np.nan, 'wilcoxon_statistic': statistic, 'p_rho_greater_than_0': p_value})
one_sample_df = pd.DataFrame(one_sample_records)
one_sample_df['p_fdr'] = np.nan
for target_name in TARGET_CELLTYPES:
    valid_mask = one_sample_df['target'].eq(target_name) & one_sample_df['p_rho_greater_than_0'].notna()
    if valid_mask.sum() > 0:
        one_sample_df.loc[valid_mask, 'p_fdr'] = multipletests(one_sample_df.loc[valid_mask, 'p_rho_greater_than_0'], method='fdr_bh')[1]
ONE_SAMPLE_FILE = OUT_DIR / 'continuous_MP_multiscale_one_sample_tests.csv'
one_sample_df.to_csv(ONE_SAMPLE_FILE, index=False)


# ============================================================
# 18. Compare MP1-MP4 with Friedman tests
# ============================================================
friedman_records = []
for target_name in TARGET_CELLTYPES:
    for radius in RADIUS_LIST:
        paired_table = valid_result.loc[valid_result['target'].eq(target_name) & valid_result['radius_um'].eq(radius)].pivot_table(index=GROUP_COL, columns='MP', values='spearman_rho').reindex(columns=MP_ORDER).dropna()
        if len(paired_table) >= 3:
            test_result = friedmanchisquare(*[paired_table[mp_name].to_numpy() for mp_name in MP_ORDER])
            statistic = float(test_result.statistic)
            p_value = float(test_result.pvalue)
        else:
            statistic = np.nan
            p_value = np.nan
        friedman_records.append({'target': target_name, 'radius_um': radius, 'n_complete_cores': len(paired_table), 'friedman_statistic': statistic, 'friedman_pvalue': p_value})
friedman_df = pd.DataFrame(friedman_records)
friedman_df['friedman_pvalue_fdr'] = np.nan
for target_name in TARGET_CELLTYPES:
    valid_mask = friedman_df['target'].eq(target_name) & friedman_df['friedman_pvalue'].notna()
    if valid_mask.sum() > 0:
        friedman_df.loc[valid_mask, 'friedman_pvalue_fdr'] = multipletests(friedman_df.loc[valid_mask, 'friedman_pvalue'], method='fdr_bh')[1]
FRIEDMAN_FILE = OUT_DIR / 'continuous_MP_multiscale_friedman_tests.csv'
friedman_df.to_csv(FRIEDMAN_FILE, index=False)


# ============================================================
# 19. Run pairwise comparisons among MP1-MP4
# ============================================================
pairwise_records = []
mp_pairs = list(combinations(MP_ORDER, 2))
for target_name in TARGET_CELLTYPES:
    for radius in RADIUS_LIST:
        paired_table = valid_result.loc[valid_result['target'].eq(target_name) & valid_result['radius_um'].eq(radius)].pivot_table(index=GROUP_COL, columns='MP', values='spearman_rho')
        for mp_a, mp_b in mp_pairs:
            if mp_a not in paired_table.columns or mp_b not in paired_table.columns:
                continue
            pair = paired_table[[mp_a, mp_b]].dropna()
            statistic, p_value = safe_paired_wilcoxon(pair[mp_a].to_numpy(), pair[mp_b].to_numpy(), alternative='two-sided')
            pairwise_records.append({'target': target_name, 'radius_um': radius, 'MP_A': mp_a, 'MP_B': mp_b, 'comparison': f'{mp_a}_vs_{mp_b}', 'n_paired_cores': len(pair), 'median_MP_A': pair[mp_a].median() if len(pair) > 0 else np.nan, 'median_MP_B': pair[mp_b].median() if len(pair) > 0 else np.nan, 'median_difference': (pair[mp_a] - pair[mp_b]).median() if len(pair) > 0 else np.nan, 'wilcoxon_statistic': statistic, 'p_value': p_value})
pairwise_df = pd.DataFrame(pairwise_records)
pairwise_df['p_fdr'] = np.nan
for target_name in TARGET_CELLTYPES:
    for radius in RADIUS_LIST:
        valid_mask = pairwise_df['target'].eq(target_name) & pairwise_df['radius_um'].eq(radius) & pairwise_df['p_value'].notna()
        if valid_mask.sum() > 0:
            pairwise_df.loc[valid_mask, 'p_fdr'] = multipletests(pairwise_df.loc[valid_mask, 'p_value'], method='fdr_bh')[1]
PAIRWISE_FILE = OUT_DIR / 'continuous_MP_multiscale_pairwise_tests.csv'
pairwise_df.to_csv(PAIRWISE_FILE, index=False)


# ============================================================
# 20. Compare MP2 with the other metaprograms
# ============================================================
mp2_records = []
for target_name in TARGET_CELLTYPES:
    for radius in RADIUS_LIST:
        paired_table = valid_result.loc[valid_result['target'].eq(target_name) & valid_result['radius_um'].eq(radius)].pivot_table(index=GROUP_COL, columns='MP', values='spearman_rho')
        for other_mp in ['MP1', 'MP3', 'MP4']:
            if 'MP2' not in paired_table.columns or other_mp not in paired_table.columns:
                continue
            pair = paired_table[['MP2', other_mp]].dropna()
            statistic, p_value = safe_paired_wilcoxon(pair['MP2'].to_numpy(), pair[other_mp].to_numpy(), alternative='greater')
            mp2_records.append({'target': target_name, 'radius_um': radius, 'comparison': f'MP2_vs_{other_mp}', 'n_paired_cores': len(pair), 'median_MP2': pair['MP2'].median() if len(pair) > 0 else np.nan, 'median_other': pair[other_mp].median() if len(pair) > 0 else np.nan, 'median_difference_MP2_minus_other': (pair['MP2'] - pair[other_mp]).median() if len(pair) > 0 else np.nan, 'wilcoxon_statistic': statistic, 'p_MP2_greater': p_value})
mp2_df = pd.DataFrame(mp2_records)
mp2_df['p_MP2_greater_fdr'] = np.nan
for target_name in TARGET_CELLTYPES:
    valid_mask = mp2_df['target'].eq(target_name) & mp2_df['p_MP2_greater'].notna()
    if valid_mask.sum() > 0:
        mp2_df.loc[valid_mask, 'p_MP2_greater_fdr'] = multipletests(mp2_df.loc[valid_mask, 'p_MP2_greater'], method='fdr_bh')[1]
MP2_FILE = OUT_DIR / 'continuous_MP2_multiscale_vs_other_MPs.csv'
mp2_df.to_csv(MP2_FILE, index=False)


# ============================================================
# 21. Plot multiscale curves
# ============================================================
def plot_multiscale_curve(summary_data, target_name, output_file):
    target_df = summary_data.loc[summary_data['target'].eq(target_name)].copy()
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    for mp_name in MP_ORDER:
        mp_df = target_df.loc[target_df['MP'].eq(mp_name)].sort_values('radius_um')
        ax.plot(mp_df['radius_um'], mp_df['median_spearman_rho'], marker='o', linewidth=1.5, label=mp_name)
        ax.fill_between(mp_df['radius_um'].to_numpy(), mp_df['q25_spearman_rho'].to_numpy(), mp_df['q75_spearman_rho'].to_numpy(), alpha=0.12)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('Neighborhood radius (μm)', fontsize=11)
    ax.set_ylabel('Spearman correlation with\nlocal target-cell enrichment', fontsize=11)
    ax.set_title(f'Multiscale neighborhood enrichment of {target_name}', fontsize=13)
    ax.legend(frameon=False, title=None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.show()
    plt.close(fig)
for target_name in TARGET_CELLTYPES:
    safe_target = target_name.replace(' ', '_').replace('/', '_')
    plot_multiscale_curve(summary_data=summary_df, target_name=target_name, output_file=OUT_DIR / f'continuous_MP_multiscale_{safe_target}_curve.pdf')


# ============================================================
# 22. Create multiscale boxplots
# ============================================================
def save_multiscale_boxplots(result_data, friedman_data, target_name, output_file):
    target_result = result_data.loc[result_data['target'].eq(target_name) & result_data['status'].eq('ok')].copy()
    with PdfPages(output_file) as pdf:
        for radius in RADIUS_LIST:
            radius_df = target_result.loc[target_result['radius_um'].eq(radius)].copy()
            paired_table = radius_df.pivot_table(index=GROUP_COL, columns='MP', values='spearman_rho').reindex(columns=MP_ORDER)
            plot_data = [paired_table[mp_name].dropna().to_numpy() if mp_name in paired_table.columns else np.array([]) for mp_name in MP_ORDER]
            fig, ax = plt.subplots(figsize=(5.0, 4.5))
            complete_table = paired_table.dropna()
            for _, row in complete_table.iterrows():
                ax.plot(np.arange(1, len(MP_ORDER) + 1), row[MP_ORDER].to_numpy(), linewidth=0.55, alpha=0.2, zorder=1)
            ax.boxplot(plot_data, labels=MP_ORDER, widths=0.58, showfliers=False, medianprops={'linewidth': 1.4}, boxprops={'linewidth': 1.1}, whiskerprops={'linewidth': 1.1}, capprops={'linewidth': 1.1})
            plot_rng = np.random.default_rng(RANDOM_SEED + radius)
            for position, values in enumerate(plot_data, start=1):
                if len(values) == 0:
                    continue
                jitter = plot_rng.normal(loc=0, scale=0.045, size=len(values))
                ax.scatter(np.full(len(values), position) + jitter, values, s=28, linewidths=0, alpha=0.85, zorder=3)
            ax.axhline(0, linestyle='--', linewidth=1)
            friedman_row = friedman_data.loc[friedman_data['target'].eq(target_name) & friedman_data['radius_um'].eq(radius)]
            if len(friedman_row) > 0 and pd.notna(friedman_row['friedman_pvalue'].iloc[0]):
                p_value = float(friedman_row['friedman_pvalue'].iloc[0])
                if p_value < 0.001:
                    p_text = f'Friedman P={p_value:.2e}'
                else:
                    p_text = f'Friedman P={p_value:.3f}'
            else:
                p_text = 'Friedman P=NA'
            ax.text(0.98, 0.98, p_text, transform=ax.transAxes, ha='right', va='top', fontsize=9)
            ax.set_xlabel('Malignant cNMF program', fontsize=11)
            ax.set_ylabel('Spearman correlation with\nlocal target-cell enrichment', fontsize=11)
            ax.set_title(f'{target_name} enrichment within {radius} μm', fontsize=13)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
for target_name in TARGET_CELLTYPES:
    safe_target = target_name.replace(' ', '_').replace('/', '_')
    save_multiscale_boxplots(result_data=result_df, friedman_data=friedman_df, target_name=target_name, output_file=OUT_DIR / f'continuous_MP_multiscale_{safe_target}_boxplots.pdf')


# ============================================================
# 23. Export core-by-radius-by-program matrices
# ============================================================
for target_name in TARGET_CELLTYPES:
    safe_target = target_name.replace(' ', '_').replace('/', '_')
    matrix = valid_result.loc[valid_result['target'].eq(target_name)].pivot_table(index=GROUP_COL, columns=['radius_um', 'MP'], values='spearman_rho')
    matrix.to_csv(OUT_DIR / f'continuous_MP_multiscale_{safe_target}_by_core_matrix.csv')


# ============================================================
# 24. Report outputs
# ============================================================
print('\n' + '=' * 75)
print('Continuous MP-score multiscale neighborhood enrichment analysis completed.')
print('\nOutput directory:')
print(OUT_DIR)
print('\nMain core-level results:')
print(CORE_RESULT_FILE)
print('\nCross-core summary:')
print(SUMMARY_FILE)
print('\nPer-program enrichment tests:')
print(ONE_SAMPLE_FILE)
print('\nOverall MP1-MP4 comparison:')
print(FRIEDMAN_FILE)
print('\nPairwise MP comparisons:')
print(PAIRWISE_FILE)
print('\nMP2 comparisons with other programs:')
print(MP2_FILE)
print('\nCross-core summary results:')
print(summary_df.to_string(index=False))
print('\nFocused MP2 comparison results:')
print(mp2_df.to_string(index=False))
print('=' * 75)
