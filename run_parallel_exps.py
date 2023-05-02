import multiprocessing as mp
from datetime import datetime
from copy import deepcopy
import pandas as pd
import pickle

# Online learning
from river.cluster import KMeans, STREAMKMeans
from river import stream 
from river.metrics import Silhouette, AdjustedRand
from river_extra.metrics.cluster.ssq_based import CalinskiHarabasz

# Custom
from online_autoclust_hpo.hp_search_spaces import default_models, search_spaces
from online_autoclust_hpo.optimisation import ClustHyperopt
from utils import Dataset
from online_autoclust_hpo.online_autoclust_hpo import *


def run_single_experiment(inputs):
    """
    Run a single experiment for clustering algorithms.
    
    Args:
        inputs (tuple): A tuple containing the following parameters:
            j (int): Index of the current experiment.
            df_name (str): The name of the dataset.
            df (pd.DataFrame): The dataset to be used in the experiment.
            i (int): The current experiment number.
            n_experiments (int): Total number of experiments.

    Returns:
        tuple: A tuple containing the dataset name and the experiment results.
    """
    j, df_name, df, i, n_experiments = inputs

    result = {}
    
    exp_start = datetime.now()

    exp_nr = i + 1

    print(f'-- Running experiment nr {exp_nr}/{n_experiments} for {df_name}')

    result[f'{exp_nr}'] = {}

    # Setup the algo
    d1 = Dataset(df, 1001)
    algo_list = ['KMeans', 'STREAMKMeans']
    n_trials_list = [50, 50] 
    metrics_list = [deepcopy(Silhouette()), deepcopy(CalinskiHarabasz())]
    ext_metric = deepcopy(AdjustedRand())

    # Run the experiments
    oclust = deepcopy(runOnlineAutoClust(d_object = d1,
               algo_list = algo_list,
               metrics_list = metrics_list,
               ext_metric = ext_metric,
               n_trials_list = n_trials_list))

    experiments = oclust.run_Experiments()

    clusterer1 = algo_list[0]
    clusterer2 = algo_list[1]
    int1 = 'Silhouette'
    int2 = 'CalinskiHarabasz'

    # Save the results
    result[f'{exp_nr}']['def1'] = experiments['defaults'][clusterer1]['ext_scores']
    result[f'{exp_nr}']['def2'] = experiments['defaults'][clusterer2]['ext_scores']
    result[f'{exp_nr}']['hpo1'] = experiments['hpo_models'][int1]['ext_scores']
    result[f'{exp_nr}']['hpo2'] = experiments['hpo_models'][int2]['ext_scores']
    result[f'{exp_nr}']['ensemble'] = experiments['ensemble']['ext_scores']
    exp_runtime = (datetime.now() - exp_start).total_seconds()

    result[f'{exp_nr}']['runtime'] = exp_runtime 
    print(f'Experiment {i+1}/{n_experiments} duration: {exp_runtime} seconds.')
    print('-----')
    print()

    return df_name, result

def load_datasets():
    """Load and return the datasets."""
    ## Blobs
    blobs_c8_f3 = pd.read_csv('./datasets/blobs_c8_f3.csv')
    blobs_c19_f3 = pd.read_csv('./datasets/blobs_c19_f3.csv')
    blobs_c6_f10 = pd.read_csv('./datasets/blobs_c6_f10.csv')
    blobs_c19_f10 = pd.read_csv('./datasets/blobs_c19_f10.csv')

    ## S-Curves
    scurves_c3_f3 = pd.read_csv('./datasets/scurves_c3_f3.csv')
    scurves_c8_f3 = pd.read_csv('./datasets/scurves_c3_f3.csv')

    df_names = ['blobs_c8_f3', 'blobs_c19_f3', 'blobs_c6_f10', 'blobs_c19_f10', 'scurves_c3_f3', 'scurves_c8_f3']
    df_list = [blobs_c8_f3, blobs_c19_f3, blobs_c6_f10, blobs_c19_f10, scurves_c3_f3, scurves_c8_f3]

    return df_names, df_list

def main():
    """
    Main function that runs the experiment pipeline.

    The pipeline includes the following steps:
    1. Load the datasets
    2. Run the experiments using multiprocessing
    3. Combine the results into a single dictionary
    4. Calculate and print the total runtime
    5. Save the results to a file
    """
    df_names, df_list = load_datasets()

    results_dict = {}
    n_experiments = 100 # number of experiments
    n_processors = 7 # number of cores

    all_start_time = datetime.now()

    print(f'Started on {all_start_time}')

    inputs = [(j, df_name, df_list[j], i, n_experiments) for j, df_name in enumerate(df_names) for i in range(n_experiments)]

    # Use the spawn context to create the pool
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_processors) as pool:
        results = pool.map(run_single_experiment, inputs)
        pool.close()
        pool.join()

    # Combine the results into the results_dict
    for df_name in df_names:
        results_dict[f'{df_name}'] = {}

    for df_name, exp_data in results:
        results_dict[df_name].update(exp_data)

    all_end_time = datetime.now()
    total_exp_runtime = all_end_time - all_start_time

    # Save the results
    with open('results_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

    print(f'Ended on {all_end_time}')
    print(f'Total time taken: {total_exp_runtime.total_seconds()/60} minutes.')

if __name__ == '__main__':
    main()