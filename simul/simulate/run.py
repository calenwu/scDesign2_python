from typing import Dict, Union, List, Tuple, Any
import sys
sys.path.insert(0, "../../")
import anndata as ad
import pandas as pd
import numpy as np
import os
import re

import simul.base.splatter as splatter
import simul.simulate.utils as utils
import simul.patients.create_dataset as ds


from simul.base.config import SimCellConfig

from simul.patients.dataset import Dataset
from simul.cnv.sampling import ProgramDistribution
from scipy.stats import norm

from scDesign2.model_fitting import calc_u, fit_model_scDesign2
from scDesign2.data_simulation import simulate_count_scDesign2


def simulate_full_obs(
    dataset: Dataset, prob_dist: ProgramDistribution, p_drop: Union[List[float], np.ndarray, float] = 0.3,
    cell_types_avail: List[str] = []
) -> Dict[str, np.ndarray]:
    if type(p_drop) is float:
        p_drop = [p_drop, p_drop]
    all_malignant_obs = ds.simulate_malignant_comp_batches(dataset=dataset, prob_dist=prob_dist)
    all_malignant_obs, dataset = ds.drop_rarest_program(all_malignant_obs, dataset, p_1=p_drop[0], p_2=p_drop[1])
    all_healthy_obs = ds.simulate_healthy_comp_batches(dataset=dataset, cell_types_avail=cell_types_avail)
    full_obs = ds.get_full_obs(all_malignant_obs=all_malignant_obs, all_healthy_obs=all_healthy_obs)

    return full_obs


def get_common_mean_pc(
    config: SimCellConfig, full_obs: Dict[str, pd.DataFrame], rng: np.random.Generator
) -> np.ndarray:

    print("Sampling original mean...")
    gene_mean = splatter.sample_mean(rng=rng, shape=config.mean_shape, scale=config.mean_scale, size=(config.n_genes,))

    print("Sampling outlier factors...")
    outlier, outlier_factor = splatter.sample_outlier(
        rng=rng, p=config.p_outlier, location=config.outlier_loc, scale=config.outlier_scale, size=(config.n_genes,)
    )

    print("Changing mean to fit outliers...")
    modif_mean = splatter.transform_mean(mean=gene_mean, outlier=outlier, outlier_factor=outlier_factor)

    print("Getting per cell means...")
    mean_pp = splatter.get_mean_pp(mean=modif_mean, full_obs=full_obs)

    return mean_pp


def get_common_mean_pc_scdesign2(
    config: SimCellConfig,
    full_obs: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    programs: [str],
    subclones: [str],
    reference_dataset_path: str,
    pickle_path: str=None,
) -> np.ndarray:
    import pickle
    import copy
    print("Sampling original mean...")
    di = {}
    for k in full_obs.keys():
        # load the copula by storing it first as a pickle file from the preprocessing.ipynb and set the pickle_path, rather than simulating each batch from scratch
        if pickle_path:
            copula_result = pickle.load(open(pickle_path))
        else:
            data = pd.read_csv(reference_dataset_path, sep=',', index_col=0)
            data.columns = [re.sub(r"\.\d+$", "", string) for string in data.columns]
            copula_result = fit_model_scDesign2(data, list(set(data.columns)), sim_method = 'copula', jitter=True)

        # copula_result = pickle.load(open('/Users/georgye/Documents/repos/ethz/bsc_thesis/SplatterSim-mainScDesign2/notebook/data/copula_result_bcells_100_250_1400_1900_1350_09_05_02_01_ind.pkl', 'rb'))
        # copula_result = pickle.load(open('/Users/georgye/Documents/repos/ethz/bsc_thesis/SplatterSim-mainScDesign2/notebook/data/copula_result_bcells_100_250_1400_1900_1350_09_05_02_01_copy.pkl', 'rb'))
        # dont do this
        # data = pd.read_csv(os.path.dirname(os.path.abspath('')) + '/scDesign2/data/bcells_100_250_1400_1900_1350_09_05_02_01.csv', sep=',', index_col=0)
        for program in programs:
            for subclone in subclones:
                copula_result['{}_{}'.format(program, subclone)] = copy.deepcopy(copula_result[program])
        for program in programs:
            copula_result.pop(program)
        di[k] = copula_result
    return di

def get_group_cnv_transformed(
    mean_pp: np.ndarray,
    full_obs: Dict[str, np.ndarray],
    config: SimCellConfig,
    dataset: Dataset,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray]]:
    import copy
    print("Transforming mean linked to groups...")
    transformed_means, de_facs_groups = utils.transform_means_by_facs(
        rng=rng, config=config, full_obs=full_obs, mean_pc_pp=mean_pp, group_or_batch="group"
    )

    if config.batch_effect:
        print("Transforming mean linked to batch effect...")
        transformed_means, de_facs_be = utils.transform_means_by_facs(
            rng=rng, config=config, full_obs=full_obs, mean_pc_pp=transformed_means, group_or_batch="batch"
        )
    else:
        de_facs_be = {patient: pd.Series([]) for patient in transformed_means}
    print("Transforming mean linked to CNV profile...")
    transformed_means, gain_expr_full, loss_expr_full = utils.transform_malignant_means(
        full_obs=full_obs, transformed_means=transformed_means, dataset=dataset, shared_cnv=config.shared_cnv
    )

    return transformed_means, de_facs_groups, de_facs_be, gain_expr_full, loss_expr_full


def get_group_cnv_transformed_scDesign2(
    mean_pp: Dict[str, Dict[str, np.ndarray]],
    full_obs: Dict[str, np.ndarray],
    config: SimCellConfig,
    dataset: Dataset,
    rng: np.random.Generator
) -> Tuple[Dict[str, np.ndarray]]:
    # for k, v in mean_pp.items():
    #     for k2 in v.keys():
    #         temp = mean_pp[k][k2]
    #         mean_pp[k][k2] = mean_pp[k][k2] * 5
    print("Transforming mean linked to groups...")
    # transformed_means_copy = copy.deepcopy(mean_pp)
    transformed_means, de_facs_groups = utils.transform_means_by_facs_scdesign2(
        rng=rng, config=config, full_obs=full_obs, mean_pc_pp=mean_pp, group_or_batch="group"
    )

    # for k, v in transformed_means.items():
    #     print(k)
    #     for k2 in v.keys():
    #         temp = np.abs(mean_pp[k][k2] - transformed_means[k][k2])
    #         print('{}: {}'.format(k2, np.mean(temp[temp != 0])))

    if config.batch_effect:
        print("Transforming mean linked to batch effect...")
        transformed_means, de_facs_be = utils.transform_means_by_facs_scdesign2(
            rng=rng, config=config, full_obs=full_obs, mean_pc_pp=transformed_means, group_or_batch="batch"
        )
        # for k, v in transformed_means.items():
        #     for k2 in transformed_means.keys():
        #         for k3 in v:
        #             if k != k2:
        #                 if k3 in transformed_means[k2].keys():
        #                     temp = np.abs(transformed_means[k][k3] - transformed_means[k2][k3])
        #                     print('{} vs {} - {}: {}'.format(k, k2, k3, np.median(temp[temp != 0])))
                            # print('{} vs {} - {}: {}'.format(k, k2, k3, np.sum(np.abs(transformed_means[k][k3] - transformed_means[k2][k3]))))
    else:
        de_facs_be = {patient: pd.Series([]) for patient in mean_pp}    

    print("Transforming mean linked to CNV profile...")
    transformed_means, gain_expr_full, loss_expr_full = utils.transform_malignant_means_scdesign2(
        full_obs=full_obs, transformed_means=transformed_means, dataset=dataset, shared_cnv=config.shared_cnv)

    # for k, v in transformed_means.items():
    #     print(k)
    #     for k2 in v.keys():
    #         temp = np.abs(transformed_means_copy[k][k2] - transformed_means[k][k2])
    #         print('sum {}: {}'.format(k2, np.sum(temp)))
    #         print('old {}: {}'.format(k2, np.mean(transformed_means[k][k2][transformed_means[k][k2] != 0])))
    #         print('new {}: {}'.format(k2, np.mean(transformed_means_copy[k][k2][transformed_means_copy[k][k2] != 0])))

    return transformed_means, de_facs_groups, de_facs_be, gain_expr_full, loss_expr_full


def adjust_libsize(
    rng: np.random.Generator, transformed_means: Dict[str, np.ndarray], config: SimCellConfig
) -> Dict[str, np.ndarray]:
    print("Sampling cell-specific library size...")
    pat_libsize = splatter.sample_library_size(
        rng=rng, transformed_means=transformed_means, location=config.libsize_loc, scale=config.libsize_scale
    )
    print("Adjusting for library size...")
    libsize_means = splatter.libsize_adjusted_means(means=transformed_means, libsize=pat_libsize)
    return libsize_means


def sample_counts_patient(mean_pc: np.ndarray, config: SimCellConfig, rng: np.random.Generator) -> np.ndarray:

    print("Getting BCV...")
    bcv = splatter.sample_BCV(rng=rng, means=mean_pc, common_disp=config.common_disp, dof=config.dof)

    print("Calculating trended mean...")
    trended_mean = splatter.sample_trended_mean(rng=rng, means=mean_pc, bcv=bcv)

    print("Sampling true counts...")
    true_counts = splatter.sample_true_counts(rng=rng, means=trended_mean)

    print("Computing gene and cell-specific dropout probability...")
    dropout_prob = splatter.get_dropout_probability(
        means=trended_mean, midpoint=config.dropout_midpoint, shape=config.dropout_shape
    )

    print("Sampling dropout...")
    dropout = splatter.sample_dropout(rng=rng, dropout_prob=dropout_prob)

    print("Transforming counts with dropout...")
    counts = splatter.get_counts(true_counts=true_counts, dropout=dropout)

    return counts


def simulate_dataset(
    config: SimCellConfig, rng: np.random.Generator, full_obs: Dict[str, pd.DataFrame], dataset: Dataset
) -> Tuple[Dict[str, np.ndarray]]:
    mean_pp = get_common_mean_pc(config=config, full_obs=full_obs, rng=rng)
    transformed_means, de_facs_group, de_facs_be, gain_expr_full, loss_expr_full = get_group_cnv_transformed(
        mean_pp=mean_pp, full_obs=full_obs, config=config, dataset=dataset, rng=rng
    )
    # transformed_means = adjust_libsize(rng=rng, transformed_means=transformed_means, config=config)
    final_counts_pp = {}
    for pat in transformed_means:
        # final_counts_pp[pat] = sample_counts_patient(mean_pc=transformed_means[pat], config=config, rng=rng)
        final_counts_pp[pat] = sample_counts_patient(mean_pc=transformed_means[pat], config=config, rng=rng)

    return final_counts_pp, de_facs_group, de_facs_be, gain_expr_full, loss_expr_full


def recalculate_cov_matrix(v: Dict[str, Any], reference_dataset_path: str) -> np.ndarray:
    data = pd.read_csv(reference_dataset_path, sep=',', index_col=0)
    data.columns = [re.sub(r"\.\d+$", "", string) for string in data.columns]
    for k2 in v.keys():
        temp_data = data.filter(regex='.*{}.*'.format(k2.split('_')[0]))
        quantile_normal = norm.ppf(calc_u(temp_data.iloc[v[k2]['gene_sel1'], :], v[k2]['marginal_param1']))
        cov_mat = np.corrcoef(quantile_normal)
        if np.isnan(cov_mat).any():
            np.fill_diagonal(cov_mat, 1)
            # Compute the average of the finite (non-nan) values in the matrix
            avg = np.nanmean(cov_mat)
            # Replace any nan values with this average
            cov_mat = np.where(np.isnan(cov_mat), avg, cov_mat)
        if np.isinf(cov_mat).any():
            np.fill_diagonal(cov_mat, 1)
            # Compute the average of the finite (non-nan) values in the matrix
            avg = np.nanmean(cov_mat)
            # Replace any nan values with this average
            cov_mat = np.where(np.isnan(cov_mat), avg, cov_mat)
        v[k2]['cov_mat'] = cov_mat


def rearraging_count_matrix(full_obs, copula_results, counts, cell_type_prop, n_cells):
    rearranged_counts = {}
    for k, v in full_obs.items():
        cell_type_mask_counter = 0
        cell_type_mask = {}
        for idx, cp in enumerate(cell_type_prop[k]):
            cell_type_mask[list(copula_results[k].keys())[idx]] = cell_type_mask_counter
            cell_type_mask_counter += cp
        temp_counter = 0
        rearranged_counts[k] = np.zeros((n_cells[k], 5000))
        for idx, cell in enumerate(v.values):
            for k2 in cell_type_mask.keys():
                if cell[2] == 'malignant':
                    if '{}_{}'.format(cell[1], cell[0]) == k2:
                        rearranged_counts[k][idx] = counts[k][cell_type_mask[k2]]
                        cell_type_mask[k2] += 1
                        temp_counter += 1
                else:
                    if cell[1] == k2:
                        rearranged_counts[k][idx] = counts[k][cell_type_mask[k2]]
                        cell_type_mask[k2] += 1
                        temp_counter += 1
                        break
    return rearranged_counts


def simulate_dataset_scDesign2(
    config: SimCellConfig, rng: np.random.Generator, full_obs: Dict[str, pd.DataFrame], dataset: Dataset,
            healthy_celltypes: [str], programs: [str], subclones: [str], reference_dataset_path: str, pickle_path: str=None) -> Tuple[Dict[str, np.ndarray]]:
    mean_pp = get_common_mean_pc_scdesign2(config=config, full_obs=full_obs, rng=rng, programs=programs, subclones=subclones, reference_dataset_path=reference_dataset_path, pickle_path=pickle_path)
    mean_pp_short = {}
    for k, v in mean_pp.items():
        mean_pp_short[k] = {}
        for k2, x in v.items():
            result = np.zeros((len(x['gene_sel1']) + len(x['gene_sel2']) + len(x['gene_sel3']), 3))
            for i in range(len(x['gene_sel1'])):
                result[x['gene_sel1'][i]] = x['marginal_param1'][i]
            for i in range(len(x['gene_sel2'])):
                result[x['gene_sel2'][i]] = x['marginal_param2'][i]
            for i in range(len(x['gene_sel3'])):
                result[x['gene_sel3'][i]] = [0, 0, 0]
            mean_pp_short[k][k2] = result

    mean_pp_short_short = {}
    for k, v in mean_pp_short.items():
        mean_pp_short_short[k] = {}
        for k2, v2 in v.items():
            mean_pp_short_short[k][k2] = np.array(v2)[:, 2]

    transformed_means, de_facs_groups, de_facs_be, gain_expr_full, loss_expr_full = get_group_cnv_transformed_scDesign2(
        mean_pp=mean_pp_short_short, full_obs=full_obs, config=config, dataset=dataset, rng=rng
    )
    for k, v in transformed_means.items():
        for k2 in v.keys():
            mean_pp_short[k][k2][:, 2] = transformed_means[k][k2]

    for k, v in mean_pp_short.items():
        for k2, v2 in v.items():
            mean_pp[k][k2]['marginal_param1'] = [mean_pp_short[k][k2][i] for i in mean_pp[k][k2]['gene_sel1']]
            mean_pp[k][k2]['marginal_param2'] = [mean_pp_short[k][k2][i] for i in mean_pp[k][k2]['gene_sel2']]

    copula_results = mean_pp

    cell_prop = calculate_cell_type_proportions(copula_results, full_obs, subclones=subclones, programs=programs, healthy_celltypes=healthy_celltypes)

    counts = {}
    n_cells = {}
    cell_type_prop = {}
    for k, v in copula_results.items():
        cell_type_prop[k] = [cell_prop[k][k2] for k2 in v.keys()]
        n_cells[k] = sum(cell_type_prop[k])
        for k2 in v.keys():
            v[k2]['marginal_param1'] = np.array(v[k2]['marginal_param1'])
            v[k2]['marginal_param2'] = np.array(v[k2]['marginal_param2'])
        recalculate_cov_matrix(v, reference_dataset_path)
        counts[k] = simulate_count_scDesign2(v, n_cells[k], cell_type_prop=cell_type_prop[k], sim_method='copula', reseq_method='multinomial')
        counts[k] = np.array(counts[k])
        counts[k] = np.transpose(counts[k])

    counts = rearraging_count_matrix(full_obs, copula_results, counts, cell_type_prop, n_cells)
    return counts, de_facs_groups, de_facs_be, gain_expr_full, loss_expr_full


def calculate_cell_type_proportions(copula_results: Dict[str, pd.DataFrame], full_obs: Dict[str, pd.DataFrame],
        subclones=[str], programs=[str], healthy_celltypes=[str]) -> Dict[str, Dict[str, int]]:
    cell_prop = {}
    for k, v in full_obs.items():
        cell_prop[k] = {'{}_{}'.format(program, subclone): 0 for program in programs for subclone in subclones}
        for x in healthy_celltypes:
            cell_prop[k][x] = 0
        for v2 in v.values:
            if v2[2] == 'malignant':
                cell_prop[k]['{}_{}'.format(v2[1], v2[0])] += 1
            else:
                cell_prop[k][v2[1]] += 1

    cell_prop_keys = list(cell_prop.keys())
    cell_prop_keys_keys = list(cell_prop[cell_prop_keys[0]].keys())
    for k in cell_prop_keys:
        for k2 in cell_prop_keys_keys:
            if cell_prop[k][k2] == 0:
                try:
                    cell_prop[k].pop(k2)
                    copula_results[k].pop(k2)
                except Exception:
                    pass
    return cell_prop


def counts_to_adata(
    counts_pp: Dict[str, np.ndarray], observations: Dict[str, pd.DataFrame], var: pd.DataFrame
) -> ad.AnnData:
    adatas = []
    for pat in counts_pp:

        obs = observations[pat]
        sample_df = pd.DataFrame(np.array([pat] * obs.shape[0]), index=obs.index, columns=["sample_id"])
        obs = pd.concat([obs, sample_df], axis=1)

        adatas.append(ad.AnnData(counts_pp[pat], obs=obs, var=var, dtype=counts_pp[pat].dtype))
    return adatas
