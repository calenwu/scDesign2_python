from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import simul.cnv.gene_expression as gex
import simul.base.splatter as splatter

from simul.patients.dataset import Dataset
from simul.base.config import SimCellConfig

import os
import pathlib as pl
import json
import anndata as ad

######## SAVING FUNCTIONS ##############


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def save_dataset(
    adatas: List[ad.AnnData],
    ds_name: str,
    de_group: pd.DataFrame,
    de_batch: pd.DataFrame,
    gain_expr_full: Dict[str, np.ndarray],
    loss_expr_full: Dict[str, np.ndarray],
    savedir: pl.Path,
    config: SimCellConfig,
) -> None:
    os.makedirs(savedir / ds_name, exist_ok=True)

    with open(savedir / ds_name / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, default=np_encoder)

    for adata in adatas:
        patname = adata.obs.sample_id[0]
        adata.write(savedir / ds_name / f"{patname}.h5ad")

    truefacsdir = savedir / ds_name / "true_facs"
    os.makedirs(truefacsdir, exist_ok=True)
    de_group.to_csv(truefacsdir / "de-groups.csv")
    de_batch.to_csv(truefacsdir / "de-batch.csv")

    cnvdir = savedir / ds_name / "cnv_effects"
    os.makedirs(cnvdir, exist_ok=True)
    for pat in gain_expr_full:
        pd.Series(gain_expr_full[pat]).to_csv(cnvdir / f"{pat}_gain_effect.csv")
    for pat in loss_expr_full:
        pd.Series(loss_expr_full[pat]).to_csv(cnvdir / f"{pat}_loss_effect.csv")


#################### RUNNING SIMULATION #############################


def break_mean_pc(mean_pc: np.ndarray, full_obs: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    mean_pc_pp = {}
    idx = 0
    for pat in full_obs:
        mean_pc_pp[pat] = mean_pc[idx : idx + full_obs[pat].shape[0]]
        idx += full_obs[pat].shape[0]
    return mean_pc_pp


def get_facs_matrix(de_pp: Dict[str, np.ndarray], full_obs: pd.DataFrame) -> Dict[str, np.ndarray]:
    de_facs = {pat: [] for pat in full_obs}
    for pat in full_obs:
        listprogram = full_obs[pat].program.ravel()
        for pr in listprogram:
            de_facs[pat].append(de_pp[pr])
        de_facs[pat] = np.array(de_facs[pat])
    return de_facs


def get_facs_matrix_batch(de_pp: Dict[str, np.ndarray], full_obs: pd.DataFrame) -> Dict[str, np.ndarray]:
    de_facs = {}
    for pat in full_obs:
        de_facs[pat] = np.array([de_pp[pat]] * full_obs[pat].shape[0])
    # de_facs = {pat: np.array([de_pp[pat]] * full_obs[pat].shape[0]) for pat in full_obs}
    return de_facs


def get_facs_matrix_scdesign2(de_pp: Dict[str, np.ndarray], full_obs: pd.DataFrame) -> Dict[str, np.ndarray]:

    de_facs = {}
    for pat in full_obs:
        de_facs[pat] = {}
        listprogram = set(full_obs[pat].program.ravel())
        for pr in listprogram:
            de_facs[pat][pr] = np.array(de_pp[pr])
    return de_facs


def get_facs_matrix_batch_scdesign2(de_pp: Dict[str, np.ndarray], full_obs: pd.DataFrame) -> Dict[str, np.ndarray]:
    de_facs = {}
    for pat in full_obs:
        de_facs[pat] = {}
        listprogram = set(full_obs[pat].program.ravel())
        for pr in listprogram:
            de_facs[pat][pr] = np.array(de_pp[pat])
    return de_facs


def get_mask_high(means_pc: np.ndarray, quantile: float = 0.3):
    avg = means_pc.mean(axis=0)
    qt = np.quantile(avg, quantile)
    return avg > qt


def transform_means_by_facs(
    rng: np.random.Generator,
    config: SimCellConfig,
    full_obs: Dict[str, pd.DataFrame],
    mean_pc_pp: Dict[str, pd.DataFrame],
    group_or_batch: str = "group",
) -> Dict[str, pd.DataFrame]:
    if group_or_batch == "group":
        group_names = config.group_names
        p_de_list = config.p_de_list
        p_down_list = config.p_down_list
        de_location_list = config.de_location_list
        de_scale_list = config.de_scale_list
    else:
        group_names = config.batch_names
        p_de_list = config.pb_de_list
        p_down_list = config.pb_down_list
        de_location_list = config.bde_location_list
        de_scale_list = config.bde_scale_list

    groups_de = splatter.get_groups_de(
        rng=rng,
        group_names=group_names,
        n_genes=config.n_genes,
        p_de_list=p_de_list,
        p_down_list=p_down_list,
        de_location_list=de_location_list,
        de_scale_list=de_scale_list,
    )

    de_pp = {group_names[i]: groups_de[i] for i in range(len(group_names))}
    if group_or_batch == "group":
        de_facs = get_facs_matrix(de_pp=de_pp, full_obs=full_obs)
    else:
        de_facs = get_facs_matrix_batch(de_pp=de_pp, full_obs=full_obs)

    transformed_means = splatter.transform_group_means(means_pp=mean_pc_pp, de_facs=de_facs)

    return transformed_means, de_pp


def transform_means_by_facs_scdesign2(
    rng: np.random.Generator,
    config: SimCellConfig,
    full_obs: Dict[str, pd.DataFrame],
    mean_pc_pp: Dict[str, pd.DataFrame],
    group_or_batch: str = "group",
) -> Dict[str, pd.DataFrame]:
    if group_or_batch == "group":
        group_names = config.group_names
        p_de_list = config.p_de_list
        p_down_list = config.p_down_list
        de_location_list = config.de_location_list
        de_scale_list = config.de_scale_list
    else:
        group_names = config.batch_names
        p_de_list = config.pb_de_list
        p_down_list = config.pb_down_list
        de_location_list = config.bde_location_list
        de_scale_list = config.bde_scale_list

    groups_de = splatter.get_groups_de(
        rng=rng,
        group_names=group_names,
        n_genes=config.n_genes,
        p_de_list=p_de_list,
        p_down_list=p_down_list,
        de_location_list=de_location_list,
        de_scale_list=de_scale_list,
    )

    de_pp = {group_names[i]: groups_de[i] for i in range(len(group_names))}
    if group_or_batch == "group":
        de_facs = get_facs_matrix_scdesign2(de_pp=de_pp, full_obs=full_obs)
    else:
        de_facs = get_facs_matrix_batch_scdesign2(de_pp=de_pp, full_obs=full_obs)

    transformed_means = splatter.transform_group_means_scdesign2(means_pp=mean_pc_pp, de_facs=de_facs)

    return transformed_means, de_pp


def get_gain_loss_expr(means: np.ndarray, quantile: float = 0.3) -> Tuple[np.ndarray]:
    # we first select which genes belong to the highly/lowly expressed, as the effect of
    # gains/losses on gene expression depends on the original expression of the gene
    mask_high = get_mask_high(means_pc=means, quantile=quantile)
    # simulate the effect of a gain/loss for a specific gene separately for each patient
    gain_expr = gex.sample_gain_vector(mask_high=mask_high)
    loss_expr = gex.sample_loss_vector(mask_high=mask_high)

    return gain_expr, loss_expr


def transform_malignant_means(
    full_obs: Dict[str, pd.DataFrame],
    transformed_means: Dict[str, pd.DataFrame],
    dataset: Dataset,
    shared_cnv: bool = False,
) -> Tuple[Dict[str, np.ndarray]]:

    cnv_transf_means = {pat: [] for pat in full_obs}

    gain_expr_full, loss_expr_full = {}, {}

    if shared_cnv:
        # the CNV effects will be shared across all patients
        full_gex = np.concatenate(list(transformed_means.values()))
        gain_expr, loss_expr = get_gain_loss_expr(means=full_gex, quantile=0.3)
        gain_expr_full["shared"] = gain_expr
        loss_expr_full["shared"] = loss_expr

    for patient in full_obs:
        df_obs = full_obs[patient].copy()
        mask_malignant = (df_obs.malignant_key == "malignant").ravel()

        df_obs = df_obs.loc[mask_malignant]
        new_means = transformed_means[patient].copy()
        mal_means = new_means[mask_malignant]
        cell_subclones = df_obs.subclone.ravel()

        if not (shared_cnv):
            gain_expr, loss_expr = get_gain_loss_expr(means=mal_means, quantile=0.3)
            gain_expr_full[patient] = gain_expr
            loss_expr_full[patient] = loss_expr

        # retrieve the subclone profiles
        mapping_patients = dataset.name_to_patient()
        patient_subclone_profiles = {
            mapping_patients[patient].subclones[i].name: mapping_patients[patient].subclones[i].profile
            for i in range(len(mapping_patients[patient].subclones))
        }

        cnvmeans = []
        for i, sub in enumerate(cell_subclones):
            subclone_profile = patient_subclone_profiles[sub].ravel()

            mean_gex = mal_means[i]

            mean_gex = gex.change_expression(
                mean_gex,
                changes=subclone_profile,
                gain_change=gain_expr,
                loss_change=loss_expr,
            )
            # we clip the values so that 0 entries become 0.0001. This is because we
            # sample from a gamma distribution at the beginning
            # the % of 0 in the data is small enough that the approximation should be ok
            mean_gex = np.clip(mean_gex, a_min=0.0001, a_max=None)

            cnvmeans.append(mean_gex)

        new_means[mask_malignant] = cnvmeans

        cnv_transf_means[patient] = new_means

    return cnv_transf_means, gain_expr_full, loss_expr_full


def transform_malignant_means_scdesign2(
    full_obs: Dict[str, pd.DataFrame],
    transformed_means: Dict[str, Dict[str, pd.DataFrame]],
    dataset: Dataset,
    shared_cnv: bool = False
) -> Tuple[Dict[str, np.ndarray]]:
    import copy
    gain_expr_full, loss_expr_full = {}, {}

    # the CNV effects will be shared across all patients
    if shared_cnv:
        full_gex = []
        for k, v in transformed_means.items():
            for v2 in transformed_means[k].values():
                full_gex.append(v2)
        full_gex = np.array(full_gex)
        gain_expr, loss_expr = get_gain_loss_expr(means=full_gex, quantile=0.3)
        gain_expr_full["shared"] = gain_expr
        loss_expr_full["shared"] = loss_expr

    for patient in full_obs:
        all_transformed_means = []
        mean_all_transformed_means = {}
        for x in transformed_means[patient].keys():
            if 'program' in x:
                transformed_means[patient][x] = np.array(transformed_means[patient][x])
                mean_all_transformed_means[x] = transformed_means[patient][x]
                all_transformed_means.append(mean_all_transformed_means[x])
        all_transformed_means = np.array(all_transformed_means)

        if not (shared_cnv):
            gain_expr, loss_expr = get_gain_loss_expr(means=all_transformed_means, quantile=0.3)
            gain_expr_full[patient] = gain_expr
            loss_expr_full[patient] = loss_expr

        # retrieve the subclone profiles
        mapping_patients = dataset.name_to_patient()
        patient_subclone_profiles = {
            mapping_patients[patient].subclones[i].name: mapping_patients[patient].subclones[i].profile
            for i in range(len(mapping_patients[patient].subclones))
        }

        for k in patient_subclone_profiles.keys():
            subclone_profile = patient_subclone_profiles[k].ravel()
            for program in transformed_means[patient].keys():
                if k in program:
                    mal_means = copy.deepcopy(transformed_means[patient][program])
                    mean_gex = mal_means
                    mean_gex = gex.change_expression(
                        mean_gex,
                        changes=subclone_profile,
                        gain_change=gain_expr,
                        loss_change=loss_expr,
                    )
                    # we clip the values so that 0 entries become 0.0001. This is because we
                    # sample from a gamma distribution at the beginning
                    # the % of 0 in the data is small enough that the approximation should be ok
                    # mean_gex = np.clip(mean_gex, a_min=0.0001, a_max=None)
                    # print(np.sum(np.abs(transformed_means[patient][program] - mean_gex)))
                    transformed_means[patient][program] = mean_gex

    return transformed_means, gain_expr_full, loss_expr_full
