import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, List


def getLNormFactors(
    rng: np.random.Generator,
    n_genes: int = 5000,
    p_de: float = 0.1,
    p_down: float = 0.5,
    de_location: float = 0.1,
    de_scale: float = 0.4,
) -> np.ndarray:

    is_de = rng.binomial(n=1, p=p_de, size=(n_genes,)).astype(bool)
    n_de = np.sum(is_de)

    expdown = rng.binomial(n=1, p=p_down, size=(n_de,))
    downregulated = (-1) ** (expdown)

    ln = rng.lognormal(mean=de_location, sigma=de_scale, size=(n_de,))

    downregulated[ln < 1] = -1 * downregulated[ln < 1]

    facs = np.ones((n_genes,))
    idx_down = np.where(is_de)[0][(downregulated < 0)]

    facs[is_de] = ln
    facs[idx_down] = 1 / facs[idx_down]

    return facs


def transform_float_list(p, n) -> np.ndarray:
    if type(p) == np.ndarray:
        return p
    else:
        return np.array([p] * n)


def sample_mean(rng: np.random.Generator, shape: float = 0.6, scale: float = 3, size: Tuple = (5000,)) -> np.ndarray:
    return rng.gamma(shape, scale, size=size)


def sample_outlier(
    rng: np.random.Generator,
    p: float = 0.05,
    location: float = 4,
    scale: float = 0.5,
    size: Tuple = (5000,),
) -> np.ndarray:
    return (
        rng.binomial(n=1, p=p, size=size),
        rng.lognormal(mean=location, sigma=scale, size=size),
    )


def get_mean_pp(mean: np.ndarray, full_obs: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    mean_pp = {}
    for pat in full_obs:
        mean_pp[pat] = np.concatenate([[mean] * full_obs[pat].shape[0]])
    return mean_pp


def transform_mean(mean: np.ndarray, outlier: np.ndarray, outlier_factor: np.ndarray) -> np.ndarray:
    modif_mean = mean.copy()
    median_mean = np.median(mean)

    modif_mean[outlier.astype(bool)] = outlier_factor[outlier.astype(bool)] * median_mean
    return modif_mean


def get_groups_de(
    rng: np.random.Generator,
    group_names: Union[List[str], np.ndarray],
    n_genes: int = 5000,
    p_de_list: Union[float, np.ndarray] = 0.1,
    p_down_list: Union[float, np.ndarray] = 0.5,
    de_location_list: Union[float, np.ndarray] = 0.1,
    de_scale_list: Union[float, np.ndarray] = 0.4,
) -> np.ndarray:

    factors = []
    n_groups = len(group_names)

    p_de_list = transform_float_list(p=p_de_list, n=n_groups)
    p_down_list = transform_float_list(p=p_down_list, n=n_groups)
    de_location_list = transform_float_list(p=de_location_list, n=n_groups)
    de_scale_list = transform_float_list(p=de_scale_list, n=n_groups)

    for group in range(n_groups):
        facs = getLNormFactors(
            rng=rng,
            n_genes=n_genes,
            p_de=p_de_list[group],
            p_down=p_down_list[group],
            de_location=de_location_list[group],
            de_scale=de_scale_list[group],
        )
        factors.append(facs)
    return np.array(factors)


def transform_group_means(means_pp: Dict[str, np.ndarray], de_facs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    transformed_means = {}
    for pat in means_pp:
        transformed_means[pat] = means_pp[pat] * de_facs[pat]
    return transformed_means


def transform_group_means_scdesign2(means_pp: Dict[str, np.ndarray], de_facs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    transformed_means = {}
    for pat in means_pp.keys():
        transformed_means[pat] = {}
        for k, v in means_pp[pat].items():
            for k2, v2 in de_facs[pat].items():
                if k2 in k:
                    transformed_means[pat][k] = v * v2
    return transformed_means


def sample_library_size(
    rng: np.random.Generator,
    transformed_means: Dict[str, np.ndarray],
    location: float = 11,
    scale: float = 0.2,
) -> np.ndarray:
    pat_libsize = {}
    for pat in transformed_means:
        pat_libsize[pat] = rng.lognormal(mean=location, sigma=scale, size=transformed_means[pat].shape[0])
    return pat_libsize


def libsize_adjusted_means(means: Dict[str, np.ndarray], libsize: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    libsize_means = {}
    for pat in means:
        scaled_mean = means[pat] / means[pat].sum(axis=1).reshape(-1, 1)
        libsize_means[pat] = scaled_mean * libsize[pat].reshape(-1, 1)
    return libsize_means


def sample_BCV(rng: np.random.Generator, means: np.ndarray, common_disp: float = 0.1, dof: int = 60) -> np.ndarray:
    scalefact = common_disp + (1 / np.sqrt(means))
    chifact = np.sqrt(dof / rng.chisquare(df=60, size=(means.shape[1],)))
    return scalefact * chifact


def sample_trended_mean(rng: np.random.Generator, means: np.ndarray, bcv: np.ndarray) -> np.ndarray:
    shape = 1 / bcv**2
    scale = means * bcv**2
    return rng.gamma(shape, scale)


def sample_true_counts(rng: np.random.Generator, means: np.ndarray) -> np.ndarray:
    return rng.poisson(means)


def get_dropout_probability(means: np.ndarray, midpoint: float = 0, shape: float = -1) -> np.ndarray:
    zero_means = np.where(means == 0)
    # to avoid logging 0
    invprob = 1 + np.exp(-shape * (np.log(means.clip(0.0001, np.inf)) - midpoint))
    prob = 1 / invprob
    prob[zero_means] = 1
    return prob


def sample_dropout(rng: np.random.Generator, dropout_prob: np.ndarray) -> np.ndarray:
    return rng.binomial(n=1, p=dropout_prob)


def get_counts(true_counts: np.ndarray, dropout: np.ndarray) -> np.ndarray:
    return (1 - dropout) * true_counts
