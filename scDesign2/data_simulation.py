import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom, gamma, binom, poisson
from typing import Any


def simulate_count_copula(copula_result: dict[str, Any], n=100, marginal='nb'):
    """
    Simulates count data based on a given copula structure.
    
    Parameters:
    ----------
    copula_result : dict
        Contains copula from fit_model_scDesign2 in model_fitting.
    n : int, optional
        Number of samples to simulate. Defaults to 100.
    
    marginal : str, optional
        The type of marginal distribution to use for the simulation. Options are 'nb' (Negative Binomial) and 'Gamma'.
        Defaults to 'nb'.
    
    Returns:
    -------
    ndarray
        Simulated count data as a 2D array. Each row represents a gene, and each column represents a sample.
    
    Raises:
    ------
    ValueError
        If the provided marginal is not among the valid options.
    
    Notes:
    -----
    - The function simulates count data using a copula to model the joint distribution of counts across genes, with specified marginals for each gene.
    - If the provided covariance matrix is not positive definite, a regularization factor is added to make it so.
   
    """
    marginal_options = ['nb', 'Gamma']
    if marginal not in marginal_options:
        raise ValueError(f"Invalid value for 'marginal'. Expected one of {marginal_options}, got {marginal}.")

    p1 = len(copula_result['gene_sel1'])
    if p1 > 0:
        try:
            result1 = np.random.multivariate_normal(mean=np.zeros(p1), cov=copula_result['cov_mat'], size=n)
        except Exception as ex:
            reg_factor = 1e-6
            cov_mat = copula_result['cov_mat'] + np.eye(copula_result['cov_mat'].shape[0]) * reg_factor
            result1 = np.random.multivariate_normal(mean=np.zeros(p1), cov=cov_mat, size=n)
        result2 = norm.cdf(result1)

    p2 = len(copula_result['gene_sel2'])
    if marginal == 'nb':
        result31 = np.zeros((n, p1))
        for iter in range(p1):
            param = copula_result['marginal_param1'][iter]
            quantiles = np.maximum(1e-5, result2[:, iter] - param[0]) / (1 - param[0])
            if np.isinf(param[1]):
                temp = poisson.ppf(quantiles, mu=param[2])
            else:
                temp = nbinom.ppf(quantiles, n=param[1], p=param[1] / (param[2] + param[1]))
                # temp = nbinom.ppf(quantiles, n=param[1], p=1/(param[2]/param[1] + 1))
            result31[:, iter] = temp
        result32 = np.zeros((n, p2))
        for iter in range(p2):
            param = copula_result['marginal_param2'][iter]
            if np.isinf(param[1]):
                temp = binom.rvs(1, 1 - param[0], size=n) * poisson.rvs(param[2], size=n)
            else:
                temp = binom.rvs(1, 1 - param[0], size=n) * nbinom.rvs(param[1], param[1] / (param[1] + param[2]), size=n)
            result32[:, iter] = temp
    elif marginal == 'Gamma':
        result31 = [binom.rvs(1, 1-param[0], size=n) * gamma.rvs(param[2], scale=param[2]/param[1], size=n)
                    for param in copula_result['marginal_param1']]
        result32 = [binom.rvs(1, 1-param[0], size=n) * gamma.rvs(param[2], scale=param[2]/param[1], size=n)
                    for param in copula_result['marginal_param2']]
    result31 = result31.astype(int).T
    result32 = result32.astype(int).T
    result = np.zeros((p1 + p2 + len(copula_result['gene_sel3']), n))
    if p1 > 0:
        result[np.ix_(copula_result['gene_sel1'], range(n))] = result31
    if p2 > 0:
        result[np.ix_(copula_result['gene_sel2'], range(n))] = result32
    return result


def simulate_count_ind(model_params: dict[str, Any], n=100, marginal='nb'):
    """
    Simulates count data based on given parameters, independently of any copula structure.
    
    Parameters:
    ----------
    model_params : dict
        copula_result from fit_model_scDesign2 in model_fitting.
    
    n : int, optional
        Number of samples to simulate. Defaults to 100.
    
    marginal : str, optional
        The type of marginal distribution to use for the simulation. Options include 'nb' (Negative Binomial) and others.
        Defaults to 'nb'.
    
    Returns:
    -------
    np.ndarray
        Simulated count data as a 2D array. Each row represents a gene, and each column represents a sample.
    
    Raises:
    ------
    ValueError
        If the provided marginal is not in the valid list of marginals.
    
    Notes:
    -----
    - The function simulates count data based on the provided model parameters. The simulation is independent, meaning that the joint distribution of counts across genes is assumed to be a product of the marginal distributions.
    - The simulation can use different marginals for different genes, as specified in the `model_params`.
    """
    marginal = marginal.lower()
    valid_marginals = ['nb', 'gamma']
    if marginal not in valid_marginals:
        raise ValueError(f"Invalid marginal value. Expected one of: {valid_marginals}")
    if model_params['sim_method'] == 'copula' or 'gene_sel3' in model_params:
        p1 = len(model_params['gene_sel1'])
        p2 = len(model_params['gene_sel2'])
        result31 = []
        result32 = []
        if marginal == 'nb':
            for iter in range(p1):
                param = model_params['marginal_param1'][iter]
                binom_samples = binom.rvs(1, 1-param[0], size=n)
                if np.isinf(param[1]):
                    neg_binom_samples = np.random.poisson(param[2], n)
                else:
                    neg_binom_samples = nbinom.rvs(param[2], param[1] / (param[1] + param[2]), size=n)
                result31.append(
                    binom_samples *
                    neg_binom_samples
                )
            result31 = np.transpose(result31)
            for iter in range(p2):
                param = model_params['marginal_param2'][iter]
                result32.append(
                    np.random.binomial(1, 1 - param[0], n) *
                    np.random.negative_binomial(param[1], param[1] / (param[2] + param[1]), n)
                )
            result32 = np.transpose(result32)
        elif marginal == 'gamma':
            result31 = [binom.rvs(1, 1-param[0], size=n) * gamma.rvs(param[2], scale=param[2]/param[1], size=n)
                        for param in model_params['marginal_param1']]
            result32 = [binom.rvs(1, 1-param[0], size=n) * gamma.rvs(param[2], scale=param[2]/param[1], size=n)
                        for param in model_params['marginal_param2']]
        result = np.zeros((p1 + p2 + len(model_params['gene_sel3']), n))
        if p1 > 0:
            result[model_params['gene_sel1'], :] = result31
        if p2 > 0:
            result[model_params['gene_sel2'], :] = result32
    else:
        p1 = len(model_params['gene_sel1'])
        p2 = len(model_params['gene_sel2'])
        result = np.zeros((p1 + p2, n))
        result31 = []
        if p1 > 0:
            if marginal == 'nb':
                for iter in range(p1):
                    param = model_params['marginal_param1'][iter]
                    result31.append(np.random.binomial(1, 1 - param[0], n) * np.random.negative_binomial(param[1], param[1] / (param[2] + param[1]), n))
            elif marginal == 'gamma':
                for iter in range(p1):
                    param = model_params['marginal_param1'][iter]
                    result31.append(np.random.binomial(1, 1 - param[0], n) * np.random.gamma(param[1], param[2] / param[1], n))
            result31 = np.transpose(result31)
            result[model_params['gene_sel1'], :] = result31
        return result


def simulate_count_scDesign2(model_params, n_cell_new=None, cell_type_prop=1,
                             total_count_new=None, total_count_old=None,
                             n_cell_old=None, sim_method='copula',
                             reseq_method='mean_scale', cell_sample=False):
    """
    Simulates sequencing data based on given parameters and methods.
    
    Parameters:
    ----------
    model_params : dict
        copula_result from fit_model_scDesign2 in model_fitting.
    
    n_cell_new : int, optional
        The number of new cells to be simulated. Default is None.
    
    cell_type_prop : float or list or ndarray, optional
        Proportions of cell types. If a single value is provided, it will be used for all cell types.
        If a list or array is provided, it should match the number of models. Default is 1.
    
    total_count_new : int or float, optional
        Total count of new reads. If not provided, the resequencing method will default to 'mean_scale'.
    
    total_count_old : int or float, optional
        Total count of old reads. Default is None.
    
    n_cell_old : int, optional
        Number of old cells. Default is None.
    
    sim_method : str, optional
        Method to use for simulation. Options are 'copula' and 'ind'. Default is 'copula'.
    
    reseq_method : str, optional
        Method to use for resequencing. Options are 'mean_scale' and 'multinomial'. Default is 'mean_scale'.
    
    cell_sample : bool, optional
        If True, the function will sample cells. Default is False.
    
    Returns:
    -------
    pd.DataFrame
        Simulated sequencing data. The rows represent genes and the columns represent cells.
        Each entry in the dataframe represents the count of the respective gene in the respective cell.
    
    Notes:
    -----
    - Ensure that the length of `cell_type_prop` matches the number of models in `model_params` if it is provided as a list or array.
    - If `total_count_new` is not provided, the function defaults to using the 'mean_scale' method for resequencing.
    - Column names of the output dataframe will be based on the 'names' attribute in `model_params` if provided.
      Otherwise, they will be generated based on the cell type index.
    """
    sim_method = sim_method.lower()
    reseq_method = reseq_method.lower()

    n_cell_vec = [v['n_cell'] for k, v in model_params.items()]
    n_read_vec = [v['n_read'] for k, v in model_params.items()]

    if total_count_old is None:
        total_count_old = 0
        for x in n_read_vec:
            temp = np.sum(x.values)
            total_count_old += temp
    if n_cell_old is None:
        n_cell_old = 0
        for x in n_cell_vec:
            n_cell_old += x

    if total_count_new is None:
        reseq_method = 'mean_scale'

    n_cell_type = len(model_params)

    if not isinstance(cell_type_prop, (list, np.ndarray)):
        cell_type_prop = [cell_type_prop] * n_cell_type

    if len(model_params) != len(cell_type_prop):
        raise ValueError('Cell type proportion should have the same length as the number of models.')

    if cell_sample:
        n_cell_each = np.random.multinomial(1, cell_type_prop, size=n_cell_new).astype(int)
    else:
        cell_type_prop = np.array(cell_type_prop) / sum(cell_type_prop)
        n_cell_each = np.round(cell_type_prop * n_cell_new).astype(int)
        if sum(n_cell_each) != n_cell_new:
            idx = np.random.choice(n_cell_type, size=1)
            n_cell_each[idx] += n_cell_new - sum(n_cell_each)

    p = (len(model_params[list(model_params.keys())[0]]['gene_sel1']) +
        len(model_params[list(model_params.keys())[0]]['gene_sel2']) +
        len(model_params[list(model_params.keys())[0]]['gene_sel3']))
    new_count = np.zeros((p, n_cell_new))

    if reseq_method == 'mean_scale':
        if total_count_new is None:
            r = np.ones(n_cell_type)
        elif isinstance(total_count_new, (int, float)):
            r = np.repeat(total_count_new / sum((total_count_old / n_cell_old) * n_cell_each), n_cell_type)
        else:
            r = (total_count_new / n_cell_new) / (total_count_old / n_cell_old)

        for i in range(n_cell_type):
            if n_cell_each[i] > 0:
                ulim = sum(n_cell_each[:i+1])
                llim = ulim - n_cell_each[i]
                params_new = model_params[list(model_params.keys())[i]]
                if len(params_new['marginal_param1']) > 0:
                    params_new['marginal_param1'][:, 2] *= r[i]

                if sim_method == 'copula':
                    params_new['marginal_param2'][:, 2] *= r[i]
                    new_count[:, llim:ulim] = simulate_count_copula(params_new, n=n_cell_each[i], marginal='nb')
                elif sim_method == 'ind':
                    new_count[:, llim:ulim] = simulate_count_ind(params_new, n=n_cell_each[i], marginal='nb')

        if model_params[list(model_params.keys())[0]].get('names') is None:
            df = pd.DataFrame(new_count)
            df.columns = list(itertools.chain.from_iterable([[str(i)] * n_cell_each[i] for i in range(n_cell_type)]))
        else:
            df = pd.DataFrame(new_count)
            df.columns = list(itertools.chain.from_iterable([[list(model_params.keys())[i].get('names')] * n_cell_each[i] for i in range(n_cell_type)]))
        return df

    elif reseq_method == 'multinomial':
        for i in range(n_cell_type):
            ulim = sum(n_cell_each[:i+1])
            llim = ulim - n_cell_each[i] + 1
            if sim_method == 'copula':
                new_count[:, llim:ulim] = simulate_count_copula(model_params[list(model_params.keys())[i]], n=n_cell_each[i], marginal='Gamma')
            elif sim_method == 'ind':
                new_count[:, llim:ulim] = simulate_count_ind(model_params[list(model_params.keys())[i]], n=n_cell_each[i], marginal='Gamma')

        new_count[np.isinf(new_count)] = 0
        new_count[np.isnan(new_count)] = 0

        bam_file = np.random.choice(np.arange(p * n_cell_new), size=total_count_new, replace=True, p=new_count.ravel())
        hist_result, _ = np.histogram(bam_file, bins=np.arange(n_cell_new * p + 1))
        result = np.reshape(hist_result, new_count.shape)

        if model_params[list(model_params.keys())[0]].get('names') is None:
            df = pd.DataFrame(result)
            df.columns = list(itertools.chain.from_iterable([[str(i)] * n_cell_each[i] for i in range(n_cell_type)]))
        else:
            df = pd.DataFrame(result)
            df.columns = list(itertools.chain.from_iterable([[list(model_params.keys())[i].get('names')] * n_cell_each[i] for i in range(n_cell_type)]))
        return df
