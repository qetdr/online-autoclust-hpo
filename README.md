# Online AutoClust HPO
A repository containing the codebase for the MSc thesis, titled "Towards automated machine learning: hyperparameter optimization in online clustering", authored by Dmitri Rozgonjuk.

## Summary
I wrote in my thesis the following:
> While an AutoML framework can encompass both offline and online phases, the main focus of this thesis is on the offline part, where models are trained and optimized based on internal CVIs. In the online phase, the models are evaluated using an external CVI. The general workflow of the present work is depicted in Figure 1. 


|![[Figure 1. The general workflow.]](images/fig_workflow.png)|
|:--:|
| <b>Figure 1. The general workflow overview: model training, optimization and model selection, followed by online evaluation and analytical comparison of results.</b>|


> Importantly, for both the `Learning` and `Prequential Evaluation` parts of the process, `N = 1001` data instances were used both in the learning and prequential evaluation phases (similarly to \cite{celik_online_2022}). The rest of the data (i.e., `N = 10000 - 1001 - 1001 = 7998` instances; more details on datasets are provided in Table 1) were used for the model evaluation part (computing ARI scores).

## Files and Directories
Below, the file tree is depicted:

```
online-autoclust-hpo/
│
├── datasets/
│   ├── <GENERATED DATASETS>
│   ├── create_datasets.py
│   └── README.md
├── images/
│   └── fig_workflow.png
├── online_autoclust_hpo/
│   ├── hp_search_spaces.py
│   ├── online_autoclust_hpo.py
│   ├── optimisation.py
│   └── similarity_matrix.py
├── results/
│   ├── <RESULTS PICKLE FILE>
├── .gitignore
├── experiment_results.ipynb
├── LICENSE
├── README.md
├── requirements.txt
├── run_parallel_exps.py
└── utils.py
```

Here is a brief overview of functionality:
- create_datasets.py <LINK>: script for creating the synthetic datasets.
- hp_search_spaces.py <LINK>: defines the clustering algorithms and their hyperparameter search spaces.
- optimisation.py <LINK>: model optimisation related functions. Has a class `ClustHyperopt` which includes methods to run `hyperopt` for a given algorithm and other parameters.
- online_autoclust_hpo.py <LINK>: includes the class `runOnlineAutoClust` which uses methods and other helper functions in the module to find the best optimized models. The method `run_Experiments()` computes the default, best optimized models, and the ensemble model.
- similarity_matrix.py <LINK>: a module that contains a solution for cluster ensembling. This is largely inspired by the work by Joao Pedro ([Tutorial](https://towardsdatascience.com/how-to-ensemble-clustering-algorithms-bf78d7602265); [Github Repository](https://github.com/jaumpedro214/posts/blob/main/ensamble_clustering/)).
- utils.py <LINK>: has helper functions and the class `Dataset` that deals with data partitioning and updating based on batch size. Other functions are related to producing analysis results (tabularizing the results from pickle file, Friedman test, Nemenyi post-hoc test, boxplots figure).
- run_parallel_exps.py <LINK>: this is the main script that runs the experiments with parallel computation.
- experiment_results.ipynb <LINK>: analyzing the results and presenting the output.


## How to Run
### Setup
The following steps need to be taken to set up the workflow:
1. Clone the repository to your local machine:
```bash
git clone https:/github.com/qetdr/online-autoclust-hpo
```
2. Navigate to the root directory of this repository from command line
```bash
cd <PATH-TO-DIRECTORY>/online-autoclust-hpo
```
3. Install all the necessary packages
```bash
pip install -r requirements.txt
```

### Creating Synthetic Datasets
In the present thesis, synthetic generated datasets were used. The script for the data generation can be found in the `create_datasets.py` script within the `datasets` directory. The generate all datasets, run the following code in the command line:

```bash
python3 ./datasets/create_datasets.py
```
This will generate six datasets, with the properties described in the Table 1:


Table 1. *Properties of data sets.*

| **Name** | **Type**   | **N samples** | **N Features** | **N Clusters** |
|----------|------------|---------------|----------------|----------------|
| SET1     | Blobs      | 10000         | 10             | 6              |
| SET2     | Blobs      | 10000         | 3              | 8              |
| SET3     | Blobs      | 10000         | 3              | 19             |
| SET4     | Blobs      | 10000         | 10             | 19             |
| SET5     | S-curves   | 10000         | 3              | 3              |
| SET6     | S-curves   | 10000         | 3              | 8              |

The datasets will be saved in the `datasets` directory from where they will be exported for running the experiments.

### Running the Online AutoClust HPO
Before running the experiments, it may be a good idea to review the experiment configuration with regards to (1) how many `hyperopt` trials are desired (my solution used 50 trials), (2) how many experiments should be run (in my case, N = 100), and how many cores of one's machine should be used when parallel-computing (in my case, N = 7). For the latter, it is generally a good idea not to allocate all cores, so N = max_cores - 1 should work well. 

- Number of `hyperopt` trials can be set in the `run_parallel_exps.py` module in the `run_single_experiment()` script. Find the `n_trials_list` variable. It is a list with the number of values for trials for defined algorithms.
- The number of experiments can be changed in the `run-parallel_exps.py` script in the `main()` function. Find the `n_experiments` variable.
- The number of experiments can be changed in the `run-parallel_exps.py` script in the `main()` function. Find the `n_processors` variable.

Once the setup is in place, you can start the experiment runs by executing the following command from Terminal.

```python
python3 ./run_parallel_exps.py
```
When the command is executed, informing output is produced (e.g., dataset name, experiment number, optimization status, model evaluation). Since the framework uses parallelized computation, the output is not produced in a sequential order and is rather concurrently informative.

Upon the completion of the experiment run, the user is notified and the total duration of runtime is displayed. In the case of 6 datasets, 50 trials for `hyperopt` optimization process, 100 such experiments with 7 cores, the total runtime was around 8 hours and 18 minutes. Hence, when trying out the experiment, it may be cautious to at first try the framework out with a smaller number of experiments.

### Analyzing the Results
Once the results are computed, they should be stored in the `results` directory as `results_dict.pkl` file. However, a temporary solution is also to download the data stored in [my personal Google Drive directory](https://drive.google.com/file/d/1oWoh_zCbYyndqBhLT5eT7uVN8neEtVE-/view?usp=share_link). However, I will most likely remove this option after June 2023.

Once the results are available, one can compare the model performances in a Jupyter Notebook `experiment_results.ipynb`. There, Friedman and Nemenyi post-hoc test results are computed and the boxplot of average ARI scores' distributions across models and datasets is displayed. The code for these analyses is within the `utils.py` module.

