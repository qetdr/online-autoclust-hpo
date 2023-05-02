from copy import deepcopy

import numpy as np

from river import stream

from river.cluster import KMeans, STREAMKMeans
from river.metrics import AdjustedRand
from .hp_search_spaces import search_spaces, default_models
from .optimisation import ClustHyperopt
from utils import Dataset
from .similarity_matrix import *

class InsufficientDataError(Exception):
    """Raised when there is not enough data to train the ensemble."""
    pass

class runOnlineAutoClust:
    """
    A class for running experiments with online clustering algorithms.
    """

    def __init__(self,
                d_object,
                algo_list,
                metrics_list,
                ext_metric,
                n_trials_list,
                best_optimized = [],):
        """
        Initializes the RunOnlineAutoClust object.

        :param d_object: Dataset object.
        :param algo_list: List of clustering algorithms.
        :param metrics_list: List of internal clustering metrics.
        :param ext_metric: External clustering metric (e.g. ARI).
        :param n_trials_list: List of numbers of trials for hyperparameter optimization.
        :param best_optimized: List of best optimized clustering models.
        """
        
        self.d_object = d_object
        self.algo_list = algo_list
        self.metrics_list = metrics_list
        self.ext_metric = ext_metric
        self.n_trials_list = n_trials_list
        self.best_optimized = best_optimized
        
    def run_Experiments(self):
        """
        Runs the experiments for the MSc thesis.
        Extracts (a) baseline (default) clusterers for algorithms,
        (b) optimized clusterers for internal metrics, and
        (c) ensemble of the best optimized metrics.
        
        :return: Dictionary containing the results of the experiments.
        """
        # Check if there is enough data
        if self.d_object.df.shape[0] < self.d_object.batch_size * 3:
            raise InsufficientDataError('Not enough data to train the ensemble')

        # Start the pipeline
        #------- Optimization ------#
        all_models = get_models(d_learn = self.d_object.df[:self.d_object.batch_size].reset_index(drop = True), 
                                        d_pred = self.d_object.df[self.d_object.batch_size:self.d_object.batch_size*2].reset_index(drop = True), 
                                        algo_list = self.algo_list, 
                                        metrics_list = self.metrics_list, 
                                        n_trials_list = self.n_trials_list)
        self.d_object.df = self.d_object.df[self.d_object.batch_size*2:].reset_index(drop = True) # remove the data from df
        
        #------- Start Evaluation (ARI) ------#
        print_frequency = 2000
        print('Using best ensemble.')
        print(f'Results will be printed every {print_frequency} samples.')

        all_models['ensemble']['ext_scores'] = np.empty([0])
        all_models['ensemble']['ext_metric'] = deepcopy(AdjustedRand())

        for i, (x, y_true) in enumerate(stream.iter_pandas(self.d_object.df.drop(columns = 'y'), 
                                                           self.d_object.df['y'])):
            preds_list = []
            for exp_type in all_models.keys():
                if exp_type == 'defaults':
                    for algo in all_models['defaults']:
                        def_model = all_models['defaults'][algo]['default_model']
                        y_pred = def_model.predict_one(x)
                        all_models['defaults'][algo]['ext_metric'] = all_models['defaults'][algo]['ext_metric'].update(y_true = y_true,
                                            y_pred = y_pred)
                        all_models['defaults'][algo]['ext_scores'] = np.append(all_models['defaults'][algo]['ext_scores'], all_models['defaults'][algo]['ext_metric'].get())
                
                elif exp_type == 'hpo_models':

                    for int_met in all_models['hpo_models'].keys():
                        # First, compute and save the ARIs for each metric
                        y_pred = all_models['hpo_models'][int_met]['best_model'].predict_one(x)
                        all_models['hpo_models'][int_met]['ext_metric'] = all_models['hpo_models'][int_met]['ext_metric'].update(y_true = y_true,
                                            y_pred = y_pred)
                        all_models['hpo_models'][int_met]['ext_scores'] = np.append(all_models['hpo_models'][int_met]['ext_scores'], all_models['hpo_models'][int_met]['ext_metric'].get())
                        # Second, use prediction to ensemble
                        best_preds = deepcopy(all_models['hpo_models'][int_met]['best_preds'])
                        # Append the predictions for ensembling
                        best_preds = np.append(best_preds, y_pred)
                        preds_list.append(best_preds)
            y_ens_pred = clustering_ensemble(preds_list)

                # introduce external metric
            all_models['ensemble']['ext_metric'] = all_models['ensemble']['ext_metric'].update(y_true = y_true,
                                                y_pred = y_ens_pred)
            all_models['ensemble']['ext_scores'] = np.append(all_models['ensemble']['ext_scores'], all_models['ensemble']['ext_metric'].get())
            if i % print_frequency == 0:
                print(f"id{i}; ARI = {all_models['ensemble']['ext_metric'].get()}")
        
        print(f"id{i}; ARI = {all_models['ensemble']['ext_metric'].get()}")
        print('Analysis completed')
        return all_models


def get_models(d_learn, d_pred, algo_list, metrics_list, n_trials_list):
    """
    Get models for clustering algorithms by optimizing the specified internal clustering metrics.
    
    Args:
        d_learn (pd.DataFrame): The learning dataset.
        d_pred (pd.DataFrame): The prequential evaluation dataset.
        algo_list (list): List of clustering algorithms.
        metrics_list (list): List of internal clustering metrics to optimize.
        n_trials_list (list): List of numbers of trials for each algorithm.

    Returns:
        dict: A dictionary containing optimized models for each internal metric.
    """
     
    # Lists of metrics: whether minimization or maximization
    d = {'defaults':{},
         'hpo_models':{},
         'ensemble':{}
         }
    for int_metric in metrics_list:
        internal_metric = deepcopy(int_metric)
        # Create a dictionary key with metric name
        metric_name = str(internal_metric).split(':')[0]
        print(f'Optimizing {metric_name}')
        
        d = optimize_metric(d,
                            d_learn, d_pred,
                            internal_metric, 
                            algo_list, 
                            n_trials_list)
    return d

def optimize_metric(d, 
                    d_learn, 
                    d_pred, 
                       internal_metric,
                       algo_list, 
                       n_trials):
    """
    Optimize a clustering algorithm for a specified internal clustering metric.
    
    Args:
        d (dict): Dictionary to store the optimized models.
        d_learn (pd.DataFrame): The learning dataset.
        d_pred (pd.DataFrame): The prequential evaluation dataset.
        internal_metric (object): The internal clustering metric to optimize.
        algo_list (list): List of clustering algorithms.
        n_trials (int): Number of trials for each algorithm.
    
    Returns:
        dict: Updated dictionary with optimized models for the specified internal metric.
    """
        
    min_metrics = ['Silhouette', 'DaviesBouldin']
    max_metrics = ['CalinskiHarabasz', 'AdjustedRand', 'AdjustedMutualInfo']  
    best_score_min = 1e06
    best_score_max = 0

    metric_name = str(internal_metric).split(':')[0]
    d['hpo_models'][metric_name] = {}

    # 1. Learn and save the preds of DEFAULTS
    for i, algorithm in enumerate(algo_list):
        print(f'**{algorithm}**')
        d['defaults'][algorithm] = {}
        def_model = eval(f'{algorithm}(**{default_models[algorithm]})')

        for _, (x, _) in enumerate(stream.iter_pandas(d_learn)):
            def_model = def_model.learn_one(x)
        for _, (x, _) in enumerate(stream.iter_pandas(d_pred)):
            def_model = def_model.learn_one(x)
        d['defaults'][algorithm]['default_model'] = def_model
        d['defaults'][algorithm]['ext_metric'] = deepcopy(AdjustedRand())
        d['defaults'][algorithm]['ext_scores'] = np.empty([0])
        
        # 2. Get the optimized models (and save preds for BEST)
        chpo = ClustHyperopt(df_learn = d_learn,
                        df_predict= d_pred,
                        clusterer_name = algorithm,
                        param_search_space = search_spaces[algorithm],
                        default_params = default_models[algorithm],
                        metric = internal_metric
                    )
        chpo.run_hyperopt(n_trials[i]) # n_trials from list
    
        # Check if metric is min or max, and save scores to dict
        if metric_name in min_metrics:
            if chpo.best_score < best_score_min:
                best_score_min = chpo.best_score
                d['hpo_models'][metric_name]['best_algorithm'] = chpo.clusterer_name
                d['hpo_models'][metric_name]['best_score'] = chpo.best_score
                d['hpo_models'][metric_name]['best_model'] = chpo.best_model
                d['hpo_models'][metric_name]['best_preds'] = chpo.best_predictions
                d['hpo_models'][metric_name]['best_metric'] = chpo.best_metric
                d['hpo_models'][metric_name]['ext_scores'] = np.empty([0])
                d['hpo_models'][metric_name]['ext_metric'] = deepcopy(AdjustedRand())
        elif metric_name in max_metrics:
            if chpo.best_score > best_score_max:
                best_score_max = chpo.best_score
                d['hpo_models'][metric_name]['best_algorithm'] = chpo.clusterer_name
                d['hpo_models'][metric_name]['best_score'] = chpo.best_score
                d['hpo_models'][metric_name]['best_model'] = chpo.best_model
                d['hpo_models'][metric_name]['best_preds'] = chpo.best_predictions
                d['hpo_models'][metric_name]['best_metric'] = chpo.best_metric
                d['hpo_models'][metric_name]['ext_scores'] = np.empty([0])
                d['hpo_models'][metric_name]['ext_metric'] = deepcopy(AdjustedRand())
    print()
    return d