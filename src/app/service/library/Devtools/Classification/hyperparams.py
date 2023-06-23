# import libraries
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Get hyper-parameters
def Get_Hyperparams():
    """
    This function get hyperparams
    Outputs:
        - dictionary of hyperparams

    """

    hyper_parameters = dict()
    
    hyper_parameters["Logistic"] = {"estimator__C": np.random.uniform(0.01, 200, 100)}

    hyper_parameters["Support Vector Machine"] = {"estimator__C": np.random.uniform(0.01, 500, 100),
                                                  #"estimator__gamma": np.random.uniform(1e-3, 100, "uniform"),
                                                  }

    hyper_parameters["RandomForest"] = {"estimator__max_depth": np.arange(3, 7, 1),
                                        "estimator__n_estimators": np.arange(100, 200, 10),
                                        "estimator__min_samples_split": np.arange(2, 10, 1),
                                        "estimator__min_samples_leaf": np.arange(1, 30, 2),
                                        "estimator__criterion": ["gini", "entropy"],
                                        "estimator__max_features": np.random.uniform(0.01, 1, 10),
                                        }

    hyper_parameters["BalancedRandomForest"] = {"estimator__max_depth": np.arange(3, 7, 1),
                                                "estimator__n_estimators": np.arange(100, 200, 10),
                                                "estimator__min_samples_split": np.arange(2, 10, 1),
                                                "estimator__min_samples_leaf": np.arange(1, 30, 2),
                                                "estimator__criterion": ["gini", "entropy"],
                                                "estimator__max_features": np.random.uniform(0.01, 1, 10),
                                                }

    hyper_parameters["GradientBoost"] = {"estimator__max_depth": np.arange(3, 7),
                                        "estimator__learning_rate": np.random.uniform(0.001, 0.5, 10),
                                        "estimator__n_estimators": np.arange(100, 200),
                                        "estimator__subsample": np.random.uniform(0.01, 1., 10),
                                        "estimator__min_samples_split": np.arange(2, 10),
                                        "estimator__max_features": np.random.uniform(0.1, 1, 10),
                                        "estimator__min_samples_leaf": np.arange(1, 10)
                                        }

    hyper_parameters["XGBoost"] =       {"estimator__n_estimators": np.arange(100, 300),
                                        "estimator__max_depth": np.arange(3,6),
                                        "estimator__eta": np.random.uniform(0, 1., 10),
                                        "estimator__learning_rate": np.random.uniform(0.01, 0.4, 10),
                                        "estimator__colsample_bytree": np.random.uniform(0., 1, 10),
                                        "estimator__colsample_bylevel": np.random.uniform(0., 1, 10),
                                        "estimator__colsample_bynode": np.random.uniform(0., 1, 10),
                                        "estimator__gamma": np.random.uniform(0., 10, 10),
                                        "estimator__reg_alpha": np.random.uniform(0., 10, 10),
                                        "estimator__reg_lambda": np.random.uniform(0., 10, 10),
                                        "estimator__booster": ["gbtree", "gblinear"],
                                        "estimator__subsample": np.random.uniform(0.01, 1, 10),
                                        "estimator__num_parallel_tree": [1, 2, 3, 4, 5],
                                        "estimator__scale_pos_weight": np.random.uniform(0.8, 1.5, 10),
                                        #"estimator__interaction_constraints": [[[0, 1], [2, 3, 4]], [[0, 2], [1, 3, 4]], [[0, 3], [1, 2, 4]], [[0, 4], [1, 2, 3]]]
                                        }
            
    hyper_parameters["ExtraTree"] =     {"estimator__max_depth": np.arange(3, 10),
                                        "estimator__criterion": ['gini', 'entropy'],
                                        "estimator__n_estimators": np.arange(100, 500),
                                        "estimator__max_features": np.random.uniform(0.1, 1, 10),
                                        "estimator__bootstrap": [True, False],
                                        }

    hyper_parameters["HistGradientBoost"] = {"estimator__max_depth": np.arange(3, 7),
                                             "estimator__learning_rate": np.random.uniform(0.01, 0.5, 10),
                                             "estimator__max_leaf_nodes": np.arange(20, 40),
                                             "estimator__loss": ["log_loss"],
                                             #'estimator__max_bin': [10, 20, 30, 50, 70, 100, 150, 200, 256],
                                             }
    
    hyper_parameters["LightGBM"] ={'estimator__num_leaves': np.arange(2, 64),
                                    'estimator__max_depth': np.arange(3, 6),
                                    "estimator__learning_rate": np.random.uniform(0.01, 0.5),
                                    'estimator__min_child_samples': np.arange(2, 500),
                                    'estimator__subsample': np.random.uniform(0.01, 1., 10),
                                    'estimator__colsample_bytree': np.random.uniform(0.01, 1., 10),
                                    'estimator__reg_alpha': np.random.uniform(0, 10),
                                    'estimator__reg_lambda': np.random.uniform(0,10),
                                    'estimator__max_bin': [10, 20, 30, 50, 70, 100, 150, 200, 256],
                                    'estimator__colsample_bytree': np.random.uniform(0.01, 1., 10)
                                    }

    hyper_parameters["LabelSpreading"] = {
        "estimator__n_neighbors": np.arange(5, 50),
        "estimator__alpha": np.random.uniform(0.1, 0.9),
    }
    
    hyper_parameters["EasyEnsemble"] = {
        "estimator__n_estimators": np.random.randint(2, 100),
        "estimator__base_estimator": [AdaBoostClassifier(), KNeighborsClassifier(n_neighbors=10)],
    }
    
    hyper_parameters["BalancedBagging"] = {
        "estimator__n_estimators": np.random.randint(2, 100),
        "estimator__max_samples": np.random.uniform(0.1, 1.),
    }
    
    hyper_parameters["Voting"] = None
    
    hyper_parameters["Stacking"] = None

    hyper_parameters["blender"] = None

    hyper_parameters["sequential"] = None

    return hyper_parameters
