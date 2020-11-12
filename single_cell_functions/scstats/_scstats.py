import numpy as np
import pandas as pd
import scipy.stats as stats
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sb
import itertools

def chi_independence(df : DataFrame,
                     th=None,
                     plot : bool=True) -> [DataFrame, DataFrame]:
    """
    Calculates the chi squared test of independence for each pair-wise comparison between all the columns in the
    dataframe.

    :param df: Dataframe where rows are samples (cells) and columns are variables (genes or proteins).
    :param th: threshold value to consider genes on or off
    :return: Dataframe with p-value for independence test for each pair of genes.
    """

    if th is None:
        raise ValueError("Pass a threshold to make data categorical")

    dfbin = df > th
    df_indep = pd.DataFrame(columns=dfbin.columns, index=dfbin.columns, dtype='float64')

    # for protein in dfbin.columns:
    #     for protein2 in dfbin.columns:
    #         if protein != protein2:
    #             crosstab = pd.crosstab(dfbin[protein], dfbin[protein2])
    #             chi2, p, dof, ex = stats.chi2_contingency(crosstab)
    #             df_indep.loc[protein, protein2] = p
    #         else:
    #             df_indep.loc[protein, protein2] = 0

    for pair in itertools.combinations(dfbin.columns, r=2):
        protein = pair[0]
        protein2 = pair[1]
        crosstab = pd.crosstab(dfbin.loc[:,protein], dfbin.loc[:,protein2])
        chi2, p, dof, ex = stats.chi2_contingency(crosstab)
        df_indep.loc[protein, protein2] = p
        df_indep.loc[protein2, protein] = p # mirror image has the same value, fill it for completeness


    # Fill in diagonal
    for protein in dfbin.columns:
        df_indep.loc[protein, protein] = 0

    # Generate a mask for the upper triangle
    # mask = np.zeros_like(df_indep, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    # mask[np.diag_indices_from(mask)] = False
    if plot:
        if df.shape[1] > 20:
            annot=False
        else:
            annot=True

        plt.figure(figsize=(12, 10))
        sb.heatmap(df_indep, cmap='Oranges_r', vmax=0.05, vmin=0.049999999999, annot=annot, fmt='.2g', cbar=False,
                   xticklabels=True, yticklabels=True)
        plt.title("Chi2 Independence test. Threshold = {}".format(round(th, 1)))

    return df_indep


def cond_prob(df: DataFrame,
              th=None,
              plot : bool=True) -> DataFrame:
    """
    Calculates the conditional probability for each pair-wise comparison between all the columns in the dataframe.

    :param df: Dataframe where rows are samples (cells) and columns are variables (genes or proteins).
    :param th: threshold value to consider genes on or off

    :return: Dataframe with conditional probabilities for each pair of genes. Listed as p(A\B) where A are the rows and
    B are the columns of the DataFrame (probability of row given column).
    """

    if th is None:
        raise ValueError("Pass a threshold to make data categorical")

    dfbin = df > th

    df_cp = pd.DataFrame(columns=dfbin.columns, index=dfbin.columns, dtype='float64')
    # Change in conditional probability
    df_cpch = pd.DataFrame(columns=dfbin.columns, index=dfbin.columns, dtype='float64')

    for proteinA in dfbin.columns:
        for proteinB in dfbin.columns:
            if proteinA != proteinB:
                pAandB = np.mean(dfbin[proteinA] & dfbin[proteinB])
                pB = dfbin[proteinB].mean()
                df_cp.loc[proteinA, proteinB] = pAandB / pB
            else:
                df_cp.loc[proteinA, proteinB] = 1
        df_cpch.loc[proteinA, :] = df_cp.loc[proteinA, :] / dfbin[proteinA].mean()

    if plot:
        fig, axt = plt.subplots(1, 2, figsize=(20, 8))
        sb.heatmap(df_cp, cmap='PiYG', vmax=1, vmin=0, annot=True, fmt='.2f', cbar=True, center=0.5, ax=axt[0])
        axt[0].set_title("Conditional Probability. Threshold = {}".format(round(th, 1)))

        # For plotting only, change places where conditional probability change = 0 to the minumum of the entire frame
        df2 = df_cpch.copy()
        df2.mask(df2 == 0, inplace=True)
        min_ch = df2.min().min()
        df2 = df_cpch.copy()
        df2[df2 == 0] = min_ch
        sb.heatmap(np.log2(df2), cmap='PiYG', annot=True, fmt='.2f', vmin=-2, vmax=2, cbar=True, center=0,
                   ax=axt[1])
        axt[1].set_title("Conditional Probability Log2(Change). Threshold = {}".format(round(th, 1)))
        plt.tight_layout()


    return df_cp, df_cpch

def mask_cond_prob(df_cp : DataFrame,
                   df_ind: DataFrame,
                   *,
                   alpha = 0.05,
                   th = None,
                   plot : bool=True) -> DataFrame:
    """
    Masks conditional probablity DataFrame using the p-values of independence test to look only at significant results.
    
    :param df_cp: Conditionaly probablity dataframe
    :param df_ind: P-values of independence test, in dataframe
    :return: 
    """

    # Check that dataframe rows and columsn are the same
    if np.all(df_cp.columns != df_ind.columns):
        raise ValueError("Columns don't match")

    if np.all(df_cp.index != df_ind.index):
        raise ValueError("Rows don't match")
    if th is None:
        raise ValueError("Pass the threshold used")

    df_masked = df_cp.copy()
    df_masked.mask(df_ind > alpha, inplace=True)

    if plot:
        df2 = df_masked.copy()
        df2.mask(df2 == 0, inplace=True)
        min_ch = df2.min().min()
        df2 = df_masked.copy()
        df2[df2 == 0] = min_ch
        plt.figure(figsize=(12, 10))
        sb.heatmap(np.log2(df2), cmap='PiYG', annot=True, fmt='.2f', vmin=-2, vmax=2, cbar=True, center=0)
        plt.title("Conditional Probability Log2(Change). Threshold = {}".format(round(th, 1)))
        plt.tight_layout()


def odds_ratio(df : DataFrame,
               th = None,
               alpha = 0.05,
               control_genes = None,
               plot : bool = True) -> [DataFrame, DataFrame]:
    """
    Calculates the Odds ratio for each pair-wise comparison between all the columns in the
    dataframe.

    :param df: Dataframe where rows are samples (cells) and columns are variables (genes or proteins).
    :param th: threshold value to consider genes on or off
    :return: Dataframe with p-value for independence test for each pair of genes.
    """

    if th is None:
        raise ValueError("Pass a threshold to make data categorical")

    dfbin = df > th


    # Odds ratio is symmetric
    if control_genes:
        # Run odds ratio against a set of control genes
        not_control = ~dfbin.columns.isin(control_genes)
        test_genes = dfbin.columns[not_control]
        df_odds = pd.DataFrame(columns=control_genes, index=test_genes, dtype='float64')
        df_p = pd.DataFrame(columns=control_genes, index=test_genes, dtype='float64')

        for protein in test_genes:
            for protein2 in control_genes:
                crosstab = pd.crosstab(dfbin.loc[:,protein], dfbin.loc[:,protein2])
                if np.shape(crosstab) != (2,2):
                    continue

                oddsr, pval = stats.fisher_exact(crosstab)
                df_odds.loc[protein, protein2] = oddsr
                df_p.loc[protein, protein2] = pval

    else:
        # Run on everycombination of genes
        df_odds = pd.DataFrame(columns=dfbin.columns, index=dfbin.columns, dtype='float64')
        df_p = pd.DataFrame(columns=dfbin.columns, index=dfbin.columns, dtype='float64')

        for pair in itertools.combinations(dfbin.columns, r=2):
            protein = pair[0]
            protein2 = pair[1]
            crosstab = pd.crosstab(dfbin.loc[:,protein], dfbin.loc[:,protein2])
            if np.shape(crosstab) != (2,2):
                continue

            oddsr, pval = stats.fisher_exact(crosstab)
            df_odds.loc[protein, protein2] = oddsr
            df_p.loc[protein, protein2] = pval
            df_odds.loc[protein2, protein] = oddsr# symmetric, fill it for completeness
            df_p.loc[protein2, protein] = pval


    # Fill in diagonal
    # for protein in dfbin.columns:
    #     df_indep.loc[protein, protein] = np.nan
    #     df

    # Generate a mask for the upper triangle
    # mask = np.zeros_like(df_indep, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    # mask[np.diag_indices_from(mask)] = False
    if plot:
        if df.shape[1] > 20:
            annot=False
        else:
            annot=True
        df_masked = df_odds.copy()
        df_masked.mask(df_p > alpha, inplace=True)
        plt.figure(figsize=(12, 10))
        sb.heatmap(np.log2(df_masked), cmap='PiYG', vmax=3, vmin=-3, annot=annot, fmt='.2g', cbar=True,
                   xticklabels=True, yticklabels=True)
        plt.title("Odds Ratio with Fisher's exact Test. Threshold = {}".format(round(th, 1)))

    return df_odds, df_p