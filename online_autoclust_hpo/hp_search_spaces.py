"""This module includes the defaults and hyperparameter search
spaces for online clustering algorithms (in river 0.14.0).

The following algorithms are represented:
- KMeans
- STREAMKMeans
"""

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from river.cluster import KMeans, STREAMKMeans

default_models = {
'KMeans':{
    'n_clusters' : 5, 
       'halflife' : 0.5, 
       'mu' : 0, 
       'sigma' : 1, 
       'p' : 2, 
       'seed' : None},
'STREAMKMeans':
    {'chunk_size' : 10, 
     'n_clusters' : 5, 
     'seed' : None}
     }

search_spaces = {
'KMeans': {
          'n_clusters': hp.choice('n_clusters', range(3, 21)),
            'halflife': hp.uniform('halflife', 0.01, 1),
            'mu': hp.uniform('mu', 0.01, 2.0),
            'sigma': hp.uniform('sigma', 0.01, 2.0),
            'p': hp.choice('p', [1,2]),
            'seed': None  
        },
'STREAMKMeans': {
        'chunk_size': hp.choice('chunk_size', range(5, 50)),
        'n_clusters': hp.choice('n_clusters', range(3, 21)),
        'halflife': hp.uniform('halflife', 0.01, 1),
        'mu': hp.uniform('mu', 0.01, 2.0),
        'sigma': hp.uniform('sigma', 0.01, 2.0),
        'p': hp.choice('p', [1,2]),
        'seed': None  
     }
}