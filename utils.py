import numpy as np
import pandas as pd
from plotnine import *

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

class Dataset:
    """
    A simple Dataset class to store a DataFrame and batch size information.
    """
    def __init__(self,
                 df,
                 batch_size,
                ):
        self.df = df
        self.batch_size = batch_size

def produce_results_table(results_dict):
    """
    Produce a summary table of clustering results from a dictionary containing data set names,
    trial numbers, and model performance in terms of ARI (Adjusted Rand Index) scores.

    Parameters
    ----------
    results_dict : dict
        A dictionary with the following structure:
        {data_set_name: {trial_number: {model_name: [ARI scores]}}}

    Returns
    -------
    results_wide : pd.DataFrame
        A wide-format DataFrame containing the mean ARI scores for each model and dataset.
        Columns: 'df', 'def1', 'def2', 'hpo1', 'hpo2', 'ens', 'runtime'
    results_long : pd.DataFrame
        A long-format DataFrame containing the mean ARI scores for each model and dataset.
        Columns: 'df', 'model', 'ARI_M'
    """
    
    results_wide = pd.DataFrame(columns = ['df', 'def1', 'def2', 'hpo1', 'hpo2', 'ens', 'runtime'])

    for df_name in results_dict.keys():
        for i in results_dict[df_name].keys():

            trial = results_dict[df_name][i]

            trial_scores = [df_name]
            for k, aris in trial.items():
                trial_scores.append(np.mean(aris))

            trial_df = pd.DataFrame([trial_scores], columns = ['df', 'def1', 'def2', 'hpo1', 'hpo2', 'ens', 'runtime'])
            results_wide = pd.concat([results_wide, trial_df], ignore_index=True)
    
    results_long = results_wide.drop(columns ='runtime').melt(id_vars = 'df', var_name = 'model', value_name = 'ARI_M')
    
    # Order the factors in results table
    results_long['model'] = pd.Categorical(results_long['model'],
                                         categories=['def1', 'def2', 'hpo1', 'hpo2', 'ens'],
                                         ordered=True)
    
    results_long['model'] = results_long['model'].replace({'def1': 'DEF: KM', 
                                                           'def2': 'DEF: SKM', 
                                                           'hpo1': 'HPO Best Sil', 
                                                           'hpo2': 'HPO Best CHI', 
                                                           'ens': 'Ensemble'})
    df_rename = {'blobs_c8_f3': 'SET2',
                'blobs_c19_f3': 'SET3',
                'blobs_c6_f10': 'SET1',
                'blobs_c19_f10': 'SET4',
                'scurves_c3_f3': 'SET5',
                'scurves_c8_f3': 'SET6'}
    results_long['df'] = results_long['df'].replace(df_rename)
    results_wide['df'] = results_wide['df'].replace(df_rename)

    return results_wide, results_long

def compute_friedman_posthocs(results_wide):
    """
    Compute Friedman test and post-hoc Nemenyi tests on the given wide-format DataFrame
    containing the mean ARI scores for each model and dataset.

    Parameters
    ----------
    results_wide : pd.DataFrame
        A wide-format DataFrame containing the mean ARI scores for each model and dataset.
        Columns: 'df', 'def1', 'def2', 'hpo1', 'hpo2', 'ens', 'runtime'

    Returns
    -------
    stat_testing_results : dict
        A dictionary containing the Friedman test results, mean and standard deviation of
        ARI scores, and post-hoc Nemenyi test results for each dataset.
        Structure: {dataset_name: {'M_ARI': array, 'SD_ARI': array,
                                   'friedman_stat': float, 'friedman_p': float,
                                   'posthocs_ps': pd.DataFrame}}
    """

    stat_testing_results = {}

    for df_name in np.unique(results_wide['df']):
        stat_testing_results[df_name] = {}

        df_test = results_wide[results_wide['df'] == df_name].reset_index(drop = True)

        stat, p = friedmanchisquare(df_test['def1'], df_test['def2'], 
                                df_test['hpo1'], df_test['hpo2'], df_test['ens'])
        stat_testing_results[df_name]['M_ARI'] =  np.around(np.array([np.mean(df_test['def1']),
                                                          np.mean(df_test['def2']),
                                                          np.mean(df_test['hpo1']),
                                                          np.mean(df_test['hpo2']),
                                                          np.mean(df_test['ens'])]
                                                          ), 3)
        stat_testing_results[df_name]['SD_ARI'] =  np.around(np.array([np.std(df_test['def1']),
                                                          np.std(df_test['def2']),
                                                          np.std(df_test['hpo1']),
                                                          np.std(df_test['hpo2']),
                                                          np.std(df_test['ens'])]
                                                          ), 3)
        
        posthoc = sp.posthoc_nemenyi_friedman(df_test.drop(columns =['df', 'runtime']))

        stat_testing_results[df_name]['friedman_stat'] = round(stat,4)
        stat_testing_results[df_name]['friedman_p'] = round(p,6)
        stat_testing_results[df_name]['posthocs_ps'] = posthoc
    
    return stat_testing_results

def results_boxplot(long_df):
    """
    Generate a boxplot of the average ARI scores for each clustering model on different
    datasets using the given long-format DataFrame.

    Parameters
    ----------
    long_df : pd.DataFrame
        A long-format DataFrame containing the mean ARI scores for each model and dataset.
        Columns: 'df', 'model', 'ARI_M'

    Returns
    -------
    fig : plotnine.ggplot
        A ggplot object representing the boxplot of average ARI scores for each model
        and dataset.
    """
    
    custom_palette = ['#C9C0B8', '#7F7D7C', '#5CB5E1', '#0F2B90', '#FA8072']

    labels = ['DEF: KM', 'DEF: SKM', 'HPO Best Sil', 'HPO Best CHI', 'Ensemble']

    fig = (ggplot(long_df, aes(x='model', y='ARI_M', fill = 'model', group = 'model')) +
    #     geom_jitter(alpha=0.1) +
        geom_boxplot() +
        labs(x = None, y = 'Average ARI scores', fill='Model') +
        facet_wrap('df', nrow=2, ncol=3) +

         scale_fill_manual(values=custom_palette, 
                        labels= labels) +


        theme_classic() +
        theme(figure_size=(6, 6),
              axis_text_x=element_text(angle=90),
              axis_ticks_major_x=element_blank(),
              axis_ticks_major_y=element_blank(),
              strip_background=element_rect(fill='#F0F0F0', color='None'), # Lighter grey background and no borders
              strip_text=element_text(weight='bold'), # Make subplot titles bold
              axis_line_y=element_line(color='grey', size=0.5),
              axis_line_x=element_line(color='grey', size=0.5),
              panel_border=element_rect(fill=None, color='grey', size=0.5) # Add y-axis line on the right side
             )  
    )
    
    return fig