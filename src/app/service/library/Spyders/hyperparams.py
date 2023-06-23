# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Get hyper-parameters
def Get_Hyperparams():
    """
    This function get hyperparams
    Outputs:
        - dictionary of hyperparams

    """

    hyper_parameters = dict()
    hyper_parameters["LogisticRegression"] = {
        "estimator__C": np.random.uniform(0.01, 100, 50)}

    # hyper_parameters["Support Vector Machine"] = {"estimator__C": np.random.uniform(0.01, 500, 100),
    # "estimator__gamma": np.random.uniform(1e-3, 100, 100),
    #                            }

    hyper_parameters["RandomForest"] = {"estimator__max_depth": [3, 4, 5, 6, 7],
                                        "estimator__n_estimators": np.arange(50, 200, 10),
                                        "estimator__min_samples_split": np.arange(2, 10, 1),
                                        "estimator__min_samples_leaf": np.arange(1, 30, 2),
                                        "estimator__criterion": ["gini", "entropy"],
                                        "estimator__max_features": np.random.uniform(0.01, 1, 10),
                                        }

    hyper_parameters["GradientBoost"] = {"estimator__max_depth": [2, 3, 4, 5, 6, 7],
                                        "estimator__learning_rate": np.random.uniform(0.001, 0.5, 10),
                                        "estimator__n_estimators": np.arange(50, 200, 10),
                                        #"estimator__subsample": np.random.uniform(0.01, 1., 10),
                                        #"estimator__min_samples_split": np.arange(2, 10, 1),
                                        #"estimator__max_features": np.random.uniform(0.1, 1, 10),
                                        #"estimator__min_samples_leaf": np.arange(1, 10, 1)
                                        }

    hyper_parameters["XGBoost"] =       {"estimator__n_estimators": np.arange(50, 300, 10),
                                        "estimator__max_depth": [3, 5, 6],
                                        "estimator__eta": np.random.uniform(0, 1., 10),
                                        "estimator__learning_rate": np.random.uniform(0.01, 0.4, 10),
                                        "estimator__colsample_bytree": np.random.uniform(0., 1, 10),
                                        "estimator__colsample_bylevel": np.random.uniform(0., 1, 10),
                                        "estimator__colsample_bynode": np.random.uniform(0., 1, 10),
                                        "estimator__gamma": np.random.uniform(0., 10, 100),
                                        "estimator__booster": ["gbtree"],
                                        "estimator__subsample": np.linspace(0.01, 1, 10),
                                        "estimator__num_parallel_tree": [1, 2, 3, 4, 5],
                                        }
            
    hyper_parameters["ExtraTree"] =     {"estimator__max_depth": np.arange(3, 20, 1),
                                        "estimator__criterion": ['gini', 'entropy'],
                                        "estimator__n_estimators": np.arange(100, 500, 10),
                                        "estimator__max_features": np.random.uniform(0.1, 1, 10),
                                        "estimator__bootstrap": [True, False],
                                        }

    hyper_parameters["HistGradientBoost"] = {"estimator__max_depth": np.arange(3, 7, 1),
                                            "estimator__learning_rate": np.random.uniform(0.01, 0.5, 10),
                                            "estimator__max_leaf_nodes": np.arange(20, 40, 1),
                                            "estimator__loss": ["auto", "binary_crossentropy"],
                                            }
    
    hyper_parameters["LightGBM"] ={'estimator__num_leaves': np.arange(2, 50, 2), 
                                'estimator__min_child_samples': np.arange(100, 500, 10), 
                                'estimator__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                                'estimator__subsample': np.random.uniform(0.01, 1., 10), 
                                'estimator__colsample_bytree': np.random.uniform(0.01, 1., 10),
                                'estimator__reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                                'estimator__reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                                }

    hyper_parameters["LabelSpreading"] = {
        "estimator__n_neighbors": np.arange(5, 50, 1),
        "estimator__alpha": np.random.uniform(0.1, 0.9, 10),
    }

    return hyper_parameters

if __name__ == '__main__':
    Get_Hyperparams()