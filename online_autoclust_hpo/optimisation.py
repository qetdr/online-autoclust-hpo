"""Hyperparameter optimizer for a clusterer
"""
from time import time
from copy import deepcopy

import numpy as np
from river.cluster import KMeans, STREAMKMeans
from river import stream
from river.metrics import Silhouette
from river_extra.metrics.cluster.ssq_based import CalinskiHarabasz
from functools import partial  # Wrapper for hyperopt function
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from .hp_search_spaces import default_models

class ClustHyperopt:
    """
    Clusterer hyperparameter optimizer.
    Works for ONE clustering algorithm with ONE metric
    to be optimized against.

    Attributes:
    df_learn : pd.DataFrame
        DataFrame on which online learning takes place.
    df_predict : pd.DataFrame
        DataFrame on which prequential evaluation happens.
    clusterer_name : str
        Name of the clusterer/model.
    param_search_space : dict
        Parameter search space (hp-objects).
    default_params : dict
        Model's default parameters.
    metric : river.metrics class
        A metric used in river.

    Additional class-specific attributes:
    default_score : float
        The default score of the model.
    last_model : river.cluster.BaseClusterer
        The last model trained.
    best_model : river.cluster.BaseClusterer
        The best model found during optimization.
    last_predictions : np.ndarray
        The last set of predictions made by the model.
    best_predictions : np.ndarray
        The best set of predictions made by the model.
    last_metric : float
        The last metric value computed.
    best_metric : float
        The best metric value computed.
    runtime : float
        The runtime of the optimization process.
    best_params : dict
        The best set of parameters found during optimization.
    internal_metrics : list
        List of internal metrics used for evaluation.
    external_metrics : list
        List of external metrics used for evaluation.
    min_metrics : list
        List of metrics where smaller values are better.
    max_metrics : list
        List of metrics where larger values are better.
    best_score : float
        The best score achieved during optimization.
    """
    def __init__(self,
                 df_learn,
                 df_predict,
                 clusterer_name,
                 param_search_space,
                 default_params,
                 metric
                 ):
        """
        Constructor for the ClustHyperopt class.

        Args:
        df_learn : pd.DataFrame
            DataFrame on which online learning takes place.
        df_predict : pd.DataFrame
            DataFrame on which prequential evaluation happens.
        clusterer_name : str
            Name of the clusterer/model.
        param_search_space : dict
            Parameter search space (hp-objects).
        default_params : dict
            Model's default parameters.
        metric : river.metrics class
            A metric used in river.
        """
        self.df_learn = df_learn
        self.df_predict = df_predict
        self.clusterer_name = clusterer_name
        self.param_search_space = param_search_space
        self.default_params = default_params
        self.metric = metric

        # Add class-specific attributes
        self.default_score = None
        self.last_model = None
        self.best_model = None
        self.last_predictions = None
        self.best_predictions = None
        self.last_metric = None
        self.best_metric = None
        self.runtime=None
        self.best_params=None
        self.internal_metrics = ['Silhouette', 'CalinskiHarabasz', 'DaviesBouldin', 'XieBeni']
        self.external_metrics = ['AdjustedRand', 'Purity', 'AdjustedMutualInfo']
        self.min_metrics = ['Silhouette', 'DaviesBouldin', 'XieBeni']
        self.max_metrics = ['CalinskiHarabasz', 'AdjustedRand', 
                            'Purity', 'AdjustedMutualInfo']

        # Dynamic variables
        if any(m in str(self.metric) for m in self.min_metrics):
            self.best_score = 1e6
        elif any(m in str(self.metric) for m in self.max_metrics):
            self.best_score = 0

    def get_score(self, params):
        """
        Saves the model and returns the metric value for online learning.

        Args:
        params : dict
            Model parameters.

        Returns:
        float
            The computed metric value for the given parameters.
        """
        clusterer = eval(f'{self.clusterer_name}(**{params})')
        temp_metric = deepcopy(self.metric)
        # Prepare the predict df
        y_array = self.df_predict['y']
        df_predict_x = self.df_predict.drop('y', axis = 'columns')
        
        y_preds = np.empty([0], dtype = int)

        if any(int_metric in str(temp_metric) for int_metric in self.internal_metrics):
            for i, (x, _) in enumerate(stream.iter_pandas(self.df_learn)):
                clusterer = clusterer.learn_one(x)
            for i, (x, _) in enumerate(stream.iter_pandas(df_predict_x)):
                y_pred = clusterer.predict_one(x)
                y_preds = np.append(y_preds, y_pred)
                if not clusterer.centers:
                    continue
                temp_metric = temp_metric.update(x=x,
                                                y_pred=y_pred,
                                                centers=clusterer.centers)
                clusterer = clusterer.learn_one(x)
        # If using external metrics (needing y_pred)
        elif any(ext_met in str(temp_metric) for ext_met in self.external_metrics):
            for i, (x, _) in enumerate(stream.iter_pandas(self.df_learn)):
                clusterer = clusterer.learn_one(x)
            for i, (x, y_true) in enumerate(stream.iter_pandas(df_predict_x, y_array)):
                y_pred = clusterer.predict_one(x)
                y_preds = np.append(y_preds, y_pred)
                if not clusterer.centers:
                    continue
                temp_metric = temp_metric.update(y_true, y_pred)
                clusterer = clusterer.learn_one(x)
        # Save the model
        self.last_model = clusterer
        self.last_predictions = y_preds
        self.last_metric = temp_metric
        return temp_metric.get()

    def obj_function_min(self, params):
        """
        An objective function for minimizing the target metric.
        Used in hyperopt.

        Args:
        params : dict
            Model parameters.

        Returns:
        dict
            A dictionary containing 'loss' and 'status' for hyperopt optimization.
        """
        clusterer_score = self.get_score(params)
        if str(clusterer_score) == "None" or str(clusterer_score) == "inf":
            clusterer_score = 10e6
        if clusterer_score < self.best_score:
            self.best_score = clusterer_score
            self.best_params = params
            self.best_model = self.last_model
            self.best_predictions = self.last_predictions
            self.best_metric = self.last_metric

        return {'loss': clusterer_score,
                'status': STATUS_OK}

    def obj_function_max(self, params):
        """
        An objective function for maximizing the target metric.
        Used in hyperopt.

        Args:
        params : dict
            Model parameters.

        Returns:
        dict
            A dictionary containing 'loss' and 'status' for hyperopt optimization.
        """
        clusterer_score = self.get_score(params)
        if str(clusterer_score) == "None" or str(clusterer_score) == "-inf":
            clusterer_score = -10e6
        if clusterer_score > self.best_score:
            self.best_score = clusterer_score
            self.best_params = params
            self.best_model = self.last_model
            self.best_predictions = self.last_predictions
            self.best_metric = self.last_metric

        return {'loss': -clusterer_score,
                'status': STATUS_OK}

    def run_hyperopt(self, n_trials=10):
        """
        The main hyperparameter optimization experiment.

        Args:
        n_trials : int, optional
            The number of hyperopt trials. Defaults to 10.
        """
        self.n_trials = n_trials

        if any(m in str(self.metric) for m in self.min_metrics):
            objective_function = partial(self.obj_function_min)

        elif any(m in str(self.metric) for m in self.max_metrics):
            objective_function = partial(self.obj_function_max)
        trials = Trials()
        start_best_params_time = time()

        best = fmin(objective_function,
                    self.param_search_space,  # search space
                    algo=tpe.suggest,
                    max_evals=self.n_trials,  # how many evaluations?
                    trials=trials)
        end_best_params_time = time()
        self.runtime = round(end_best_params_time - start_best_params_time, 5)

        # Compute the default and compare against it
        self.default_score = self.get_score(default_models[self.clusterer_name])

        if any(m in str(self.metric) for m in self.min_metrics):
            if self.default_score < self.best_score:
                self.best_params = default_models[self.clusterer_name]
                self.best_model = self.last_model
                self.best_predictions = self.last_predictions
                self.best_metric = self.last_metric
                print(f'NOTE: Default model performed better in {str(self.metric).split(":")[0]} and is considered as the best model')
        elif any(m in str(self.metric) for m in self.max_metrics):
            if self.default_score > self.best_score:
                self.best_params = default_models[self.clusterer_name]
                self.best_model = self.last_model
                self.best_predictions = self.last_predictions
                self.best_metric = self.last_metric
                print(f'NOTE: Default model performed better in {str(self.metric).split(":")[0]} and is considered as the best model')

     

