
import csv
import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from functools import partial
# from multiprocessing import Pool
from scipy.stats import chi2, nbinom, norm
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
warnings.filterwarnings("ignore")


def fit_model_scDesign2(data_mat: pd.DataFrame, cell_type_sel: list[str], sim_method='copula', marginal='auto_choose',
        jitter=True, zp_cutoff=0.8, min_nonzero_num=2, ncores=1):
    """
    Fit the model for each gene.

    Parameters
    ----------
    data_mat : pandas.DataFrame
        rows are genes, columns are cells
    cell_type_sel : list[str]
        list of cell types
    sim_method : str
        'copula' or 'ind' (independent)
    marginal : str
        Specification of the types of marginal distribution.
        Default value is 'auto_choose' which chooses between ZINB, NB, ZIP
        and Poisson by a likelihood ratio test (lrt) and whether there is underdispersion.
        'zinb' will fit the ZINB model. If there is underdispersion, it will choose between ZIP and Poisson by a lrt.
        Otherwise, it will try to fit the ZINB model. If in this case, there is no zero at all or an error occurs,
        it will fit an NB model instead.
        'nb' fits the NB model that chooses between NB and Poisson depending on whether there is underdispersion.
        'poisson' simply fits the Poisson model.
    jitter : bool
        Logical, whether a random projection should be performed in the distributional transform.
    zp_cucfoff : float
        The maximum propotion of zero allowed for a gene to be included in the joint copula model.
    min_nonzero_num : int
        The minimum number of non-zero values required for a gene to be fitted a marginal model.
    ncores : int
        number of cores to use
    
    Returns
    -------
    list
        A list with the same length as cell_type_sel that contains the fitted model as each of its element.
    """
    sim_method = sim_method.lower()
    marginal = marginal.lower()
    if np.any(np.abs(data_mat - np.round(data_mat)) > 1e-5):
        print('Warning: The entries in the input matrix are not integers. Rounding is performed.')
        data_mat = np.round(data_mat)

    data_mat = pd.DataFrame(data_mat)
    if sim_method == 'copula':
        fit_func = partial(fit_gaussian_copula, marginal=marginal, jitter=jitter,
                zp_cutoff=zp_cutoff, min_nonzero_num=min_nonzero_num)
    elif sim_method == 'ind':
        fit_func = partial(fit_wo_copula, marginal=marginal, jitter=jitter,
                min_nonzero_num=min_nonzero_num)
    else:
        raise ValueError(f"Invalid sim_method: {sim_method}")

    # def fit_and_append(ct):
    #     temp = data_mat.filter(regex=ct)
    #     return ct, fit_func(temp, ct=ct)

    # param_dict = {}
    # with ThreadPoolExecutor(max_workers=ncores) as executor:
    #     for ct, param in executor.map(fit_and_append, cell_type_sel):
    #         param_dict[ct] = param

    # return param_dict
    param = []
    for ct in cell_type_sel:
        # temp = data_mat.filter(regex=ct)
        temp = data_mat.filter(regex='^{}$'.format(ct))
        param.append(fit_func(temp, ct=ct))

    param_dict = dict(zip(cell_type_sel, param))
    return param_dict


def fit_gaussian_copula(x, marginal='auto_choose', jitter=True, zp_cutoff=0.8, min_nonzero_num=2, ct='Stem'):
    """
    Fits a Gaussian copula to the given data while considering marginal distributions.

    Parameters:
    ----------
    x : pd.DataFrame
        The data for which the copula needs to be fitted. Rows represent genes and columns represent samples.
    
    marginal : str, optional
        Specifies the types of marginal distribution.
        Choices are: 'auto_choose', 'zinb', 'nb', and 'poisson'.
        Default is 'auto_choose' which chooses between ZINB, NB, ZIP, and Poisson based on certain criteria.
    
    jitter : bool, optional
        Whether a random projection (jittering) should be performed in the distributional transform. Defaults to True.
    
    zp_cutoff : float, optional
        The maximum proportion of zero allowed for a gene to be included in the joint copula model. Defaults to 0.8.
    
    min_nonzero_num : int, optional
        The minimum number of non-zero values required for a gene to be fitted with a marginal model. Defaults to 2.
    
    ct : str, optional
        Represents the cell type. Defaults to 'Stem'.

    Returns:
    -------
    dict
        A dictionary containing:
        - 'cov_mat': Covariance matrix of the fitted Gaussian copula.
        - 'marginal_param1': Parameters of the marginal distribution for the genes in gene_sel1.
        - 'marginal_param2': Parameters of the marginal distribution for the genes in gene_sel2.
        - 'gene_sel1': Genes selected based on the proportion of zeros being less than zp_cutoff.
        - 'gene_sel2': Genes selected based on having zero proportions between zp_cutoff and a specified threshold.
        - 'gene_sel3': Remaining genes not in gene_sel1 or gene_sel2.
        - 'zp_cutoff': The provided zp_cutoff value.
        - 'min_nonzero_num': The provided min_nonzero_num value.
        - 'sim_method': Fixed to 'copula' as this function fits a copula model.
        - 'n_cell': Number of cells (columns) in the input data.
        - 'n_read': Sum of counts across all genes and samples.

    Notes:
    -----
    - The function fits a Gaussian copula to model the joint distribution of counts across genes.
    - The marginal distributions can be ZINB, NB, ZIP, or Poisson based on the provided choice or determined criteria.
    - Genes are categorized based on the proportion of zero counts they have. Different marginal models are applied 
      to different categories.
    """
    marginal = marginal.lower()
    n = x.shape[1]
    p = x.shape[0]
    # Calculate the proportion of values in each row (gene) of DataFrame x that are 0.
    gene_zero_prop = np.sum(x < 1e-5, axis=1) / n
    # Get genes where the proportion of 0s is less than zp_cutoff.
    gene_sel1 = np.where(gene_zero_prop < zp_cutoff)[0]
    # Gets genes where proportion of values are less than 1e-5 between zp_cutoff (inclusive) and (1.0 - min_nonzero_num / n) (exclusive). 
    gene_sel2 = np.where((gene_zero_prop < 1.0 - min_nonzero_num / n) & (gene_zero_prop >= zp_cutoff))[0]
    gene_sel3 = np.setdiff1d(np.arange(p), np.union1d(gene_sel1, gene_sel2))
    if len(gene_sel1) > 0:
        marginal_result1 = fit_marginals(x.iloc[gene_sel1, :], marginal, jitter=jitter, DT=True)
        quantile_normal = norm.ppf(marginal_result1['u'])
        cov_mat = np.corrcoef(quantile_normal)
    else:
        cov_mat = None
        marginal_result1 = None

    if len(gene_sel2) > 0:
        marginal_result2 = fit_marginals(x.iloc[gene_sel2, :], marginal, DT=False)
    else:
        marginal_result2 = None
    return {
        'cov_mat': cov_mat,
        'marginal_param1': marginal_result1['params'] if marginal_result1 else None,
        'marginal_param2': marginal_result2['params'] if marginal_result2 else None,
        'gene_sel1': gene_sel1,
        'gene_sel2': gene_sel2,
        'gene_sel3': gene_sel3,
        'zp_cutoff': zp_cutoff,
        'min_nonzero_num': min_nonzero_num,
        'sim_method': 'copula',
        'n_cell': n,
        'n_read': np.sum(x)
    }


def fit_wo_copula(x, marginal='auto_choose', jitter=True, min_nonzero_num=2, ct='Stem'):
    """
    Fits data without using a copula, considering only marginal distributions.

    Parameters:
    ----------
    x : pd.DataFrame
        The data for which the distribution needs to be fitted. Rows represent genes and columns represent samples.
    
    marginal : str, optional
        Specifies the types of marginal distribution.
        Choices are: 'auto_choose', 'zinb', 'nb', and 'poisson'.
        Default is 'auto_choose' which chooses between ZINB, NB, ZIP, and Poisson based on certain criteria.
    
    jitter : bool, optional
        Whether a random projection (jittering) should be performed in the distributional transform. Defaults to True.
    
    min_nonzero_num : int, optional
        The minimum number of non-zero values required for a gene to be fitted with a marginal model. Defaults to 2.
    
    ct : str, optional
        Represents the cell type. Defaults to 'Stem'.

    Returns:
    -------
    dict
        A dictionary containing:
        - 'marginal_param1': Parameters of the marginal distribution for the genes in gene_sel1.
        - 'gene_sel1': Genes selected based on having zero proportions below a certain threshold.
        - 'gene_sel2': Remaining genes not in gene_sel1.
        - 'min_nonzero_num': The provided min_nonzero_num value.
        - 'sim_method': Fixed to 'ind' as this function fits without a copula model.
        - 'n_cell': Number of cells (columns) in the input data.
        - 'n_read': Sum of counts across all genes and samples.

    Notes:
    -----
    - The function fits data without using a copula, modeling only the marginal distributions.
    - The marginal distributions can be ZINB, NB, ZIP, or Poisson based on the provided choice or determined criteria.
    - Genes are categorized based on the proportion of zero counts they have.
    """
    p, n = x.shape
    marginal = marginal.lower()

    gene_zero_prop = np.sum(x < 1e-5, axis=1) / n

    gene_sel1 = np.where(gene_zero_prop < 1.0 - min_nonzero_num / n)[0]
    gene_sel2 = np.setdiff1d(np.arange(p), gene_sel1)

    if len(gene_sel1) > 0:
        marginal_result1 = fit_marginals(x.iloc[gene_sel1, :], marginal, jitter=jitter, DT=True)
    else:
        marginal_result1 = None

    return {
        'marginal_param1': marginal_result1['params'] if marginal_result1 else None,
        'gene_sel1': gene_sel1,
        'gene_sel2': gene_sel2,
        'min_nonzero_num': min_nonzero_num,
        'sim_method': 'ind',
        'n_cell': n,
        'n_read': np.sum(x)
    }


def fit_marginals(x: pd.DataFrame, marginal=('auto_choose', 'zinb', 'nb', 'poisson'), pval_cutoff=0.05, epsilon=1e-5,
        jitter=True, DT=True):
    """
    Fit marginal distributions to the data.

    Parameters
    ----------
    pval_cutoff : float
        Cutoff of p-value of the lrt that determines whether there is zero inflation.
    epsilon : float
        Threshold value for preventing the transformed quantile to collapse to 0 or 1.
    DT : bool
        Logical, whether distributional transformed should be performed.
        If set to FALSE, the returned object u will be NULL.
    
    Returns
        {
            'params': a matrix of shape p by 3. The values of each column are: the ZI proportion,
                the dispersion parameter (for Poisson, it's Inf), and the mean parameter.
            'u': NULL or a matrix of the same shape as x, which records the transformed quantiles, by DT.
        }
    """
    p, n = x.shape
    # Ensure valid marginal input
    if marginal not in ('auto_choose', 'zinb', 'nb', 'poisson'):
        raise ValueError("Invalid value for 'marginal'")

    def calc_params(gene, marginal):
        m, v = gene.mean(), gene.var()
        if marginal == 'auto_choose':
            if m >= v:
                # Poisson model
                mle_Poisson = GLM(
                    gene,
                    np.ones((len(gene), 1)),
                    family=families.Poisson(link=sm.families.links.log())
                ).fit(maxiter=1000, disp=False, tol=1e-8, scale='x2')
                try:
                    # Zero-inflated Poisson model
                    mle_ZIP = ZeroInflatedPoisson(gene, np.ones(len(gene))).fit(
                        maxiter=1000, disp=False, tol=1e-8, scale='x2')
                    chisq_val = 2 * (mle_ZIP.llf - mle_Poisson.llf)
                    pvalue = 1 - chi2.ppf(chisq_val, 1)
                    if pvalue < pval_cutoff:
                        return np.array([plogis(mle_ZIP.params[0]), np.inf, np.exp(mle_ZIP.params[1])])
                    else:
                        return np.array([0.0, np.inf, m])
                except:
                    # Poisson model fallback
                    return np.array([0.0, np.inf, m])
            else:
                # Negative binomial model
                try:
                    mle_NB = GLM(
                        gene,
                        np.ones((len(gene), 1)),
                        family=families.NegativeBinomial(link=sm.families.links.log())
                    ).fit(maxiter=1000, disp=False, scale='x2')
                except Exception as e:
                    print(e)
                mle_NB = GLM(
                    gene,
                    np.ones((len(gene), 1)),
                    family=families.NegativeBinomial(link=sm.families.links.log())
                ).fit(maxiter=1000, disp=False, scale='x2')
                if np.min(gene) > 0:
                    return np.array([0.0, 1/mle_NB.scale, np.exp(mle_NB.params[0])])
                else:
                    try:
                        # Fit the zero-inflated negative binomial model
                        mle_ZINB = sm.ZeroInflatedNegativeBinomialP(
                            gene,
                            np.ones(len(gene))
                        ).fit(maxiter=1000, disp=False, tol=1e-8, scale='x2')
                        chisq_val = 2 * (mle_ZINB.llf - mle_NB.llf)
                        pvalue = 1 - chi2.ppf(chisq_val, 1)
                        # Calculate the estimated theta parameter
                        if pvalue < pval_cutoff:
                            return np.array([
                                plogis(mle_ZINB.params[0]),
                                1/mle_ZINB.scale,
                                np.exp(mle_ZINB.params[1])
                            ])
                        else:
                            return np.array([0.0, 1/mle_NB.scale, np.exp(mle_NB.params[0])])
                    except Exception as e:
                        # Negative binomial model fallback
                        return np.array([0.0, 1/mle_NB.scale, np.exp(mle_NB.params[0])])
        elif marginal == 'zinb':
            mle_NB = smf.glm(
                    formula="gene ~ 1", data=gene, family=sm.families.NegativeBinomial()).fit(maxiter=1000, disp=False)
            if gene.min() > 0:
                return [0.0, 1/mle_NB.scale, np.exp(mle_NB.params[0])]
            else:
                try:
                    mle_ZINB = sm.ZeroInflatedNegativeBinomialP(gene, np.ones(len(gene))).fit(maxiter=1000, disp=False)
                    disp_param = mle_ZINB.pearson_chi2 / mle_ZINB.df_resid
                    theta_param = 1/disp_param
                    return np.array([
                        plogis(mle_ZINB.params[0]),
                        theta_param,
                        np.exp(mle_ZINB.params[1])
                    ])
                except:
                    return [0.0, 1/mle_NB.scale, np.exp(mle_NB.params[0])]
        elif marginal == 'nb':
            mle_NB = smf.glm(
                    formula="gene ~ 1", data=gene, family=sm.families.NegativeBinomial()).fit(maxiter=1000, disp=False)
            disp_param = mle_NB.pearson_chi2 / mle_NB.df_resid
            # Calculate the estimated theta parameter
            nb_theta_param = 1/disp_param
            return np.array([0.0, nb_theta_param, np.exp(mle_NB.params[0])])
        elif marginal == 'poisson':
            return np.array([0.0, np.inf, m])
    params = np.array([calc_params(x.iloc[i, :], marginal) for i in range(p)])
    if DT:
        u = calc_u(x, params, jitter=jitter, epsilon=epsilon)
    else:
        u = None
    return {'params': params, 'u': u}


def calc_u(x: pd.DataFrame, params, jitter=True, epsilon=1e-5):
    """
    Calculates the transformed quantiles using distributional transform for copula modeling.

    The distributional transform is applied to convert the initial gene counts, \(X_{ij}\), 
    into uniform variables \(U_{ij}\) suitable for the copula model. This transformation is 
    defined as:

    \[
    U_{ij} = V_{ij}F_i(X_{ij} - 1) + (1 - V_{ij})F_i(X_{ij})
    \]

    where \(V_{ij}\sim\)Uniform[0,1] is generated independently for \(i=1,...,p\) and \(j=1,...,n\).

    Parameters:
    ----------
    x : pd.DataFrame
        The data containing the gene counts. Rows represent genes and columns represent samples.
    
    params : array_like
        Parameters of the marginal distributions.
    
    jitter : bool, optional
        Whether to introduce a jitter (random projection) during the transformation. Defaults to True.
    
    epsilon : float, optional
        A small value to prevent the transformed quantile from collapsing to 0 or 1. Defaults to 1e-5.

    Returns:
    -------
    np.ndarray
        A matrix of transformed quantiles, of the same shape as x.
    """
    p, n = x.shape
    u = np.zeros((p, n))
    for iter in range(p):
        param = params[iter, :]
        gene = x.iloc[iter, :]
        prob0 = param[0]
        # if dispersion is infinite, use Poisson distribution
        if np.isinf(param[1]):
            u1 = prob0 + (1 - prob0) * stats.poisson.cdf(gene, mu=param[2])
            u2 = (prob0 + (1 - prob0) * stats.poisson.cdf((gene - 1), mu=param[2])) * (gene > 0)
        else:
            u1 = prob0 + (1 - prob0) * nbinom.cdf(gene, param[1], param[1] / (param[1] + param[2]))
            u2 = (prob0 + (1 - prob0) * nbinom.cdf((gene - 1), param[1], param[1] / (param[1] + param[2]))) * (gene > 0)
        if jitter:
            v = np.random.uniform(size=n)
        else:
            v = np.full(n, 0.5)
        r = u1 * v + u2 * (1 - v)
        idx_adjust = np.where(1 - r < epsilon)
        r.to_numpy()[idx_adjust] -= epsilon
        idx_adjust = np.where(r < epsilon)
        r.to_numpy()[idx_adjust] += epsilon
        u[iter, :] = r
    return u


def plogis(x):
    """
    
    Computes the logistic function, transforming values to the interval [0, 1].

    Parameters:
    ----------
    x : float or np.ndarray
        Input value or array of values.

    Returns:
    -------
    float or np.ndarray
        Transformed value(s) in the range [0, 1].
    """
    return np.exp(x) / (1 + np.exp(x))
