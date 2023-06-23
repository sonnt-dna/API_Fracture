# import libraries
import numpy as np
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Get hyper-parameters
def Get_Hyperparams():
    """
    This function get hyperparams
    Outputs:
        - dictionary of hyperparams

    """

    hyper_parameters = dict()
    
    hyper_parameters["Logistic"] = {
        "estimator__C": Real(0.01, 200, "uniform")
    }

    hyper_parameters["Support Vector Machine"] = None#{"estimator__C": Real(0.01, 500, "uniform"),
                                                  #"estimator__gamma": Real(1e-3, 100, "uniform"),
                                                  #}

    hyper_parameters["RandomForest"] = {
        "estimator__max_depth": Integer(3, 7),
        "estimator__n_estimators": Integer(100, 200),
        "estimator__min_samples_split": Integer(2, 10),
        "estimator__min_samples_leaf": Integer(1, 30),
        "estimator__criterion": Categorical(["gini", "entropy"]),
        "estimator__max_features": Real(0.01, 1, "uniform"),
    }

    hyper_parameters["BalancedRandomForest"] = {
        "estimator__max_depth": Integer(3, 7),
        "estimator__n_estimators": Integer(100, 200),
        "estimator__min_samples_split": Integer(2, 10),
        "estimator__min_samples_leaf": Integer(1, 30),
        "estimator__criterion": Categorical(["gini", "entropy"]),
        "estimator__max_features": Real(0.01, 1, "uniform"),
    }

    hyper_parameters["GradientBoost"] = {
        "estimator__max_depth": Integer(3, 7),
        "estimator__learning_rate": Real(0.001, 0.5, "uniform"),
        "estimator__n_estimators": Integer(100, 200),
        "estimator__subsample": Real(0.01, 1., "uniform"),
        "estimator__min_samples_split": Integer(2, 10),
        "estimator__max_features": Real(0.1, 1, "uniform"),
        "estimator__min_samples_leaf": Integer(1, 10)
    }

    hyper_parameters["XGBoost"] = {
        "estimator__n_estimators": Integer(100, 300),
        "estimator__max_depth": Integer(3,6),
        "estimator__eta": Real(0, 1., "uniform"),
        "estimator__learning_rate": Real(0.01, 0.4, "uniform"),
        "estimator__colsample_bytree": Real(0., 1, "uniform"),
        "estimator__colsample_bylevel": Real(0., 1, "uniform"),
        "estimator__colsample_bynode": Real(0., 1, "uniform"),
        "estimator__gamma": Real(0., 10, "uniform"),
        "estimator__reg_alpha": Real(0., 10, "uniform"),
        "estimator__reg_lambda": Real(0., 10, "uniform"),
        "estimator__booster": Categorical(["gbtree", "gblinear"]),
        "estimator__subsample": Real(0.01, 1, "uniform"),
        "estimator__num_parallel_tree": [1, 2, 3, 4, 5],
        "estimator__scale_pos_weight": Real(0.8, 1.5, "uniform"),
        #"estimator__interaction_constraints": [[[0, 1], [2, 3, 4]], [[0, 2], [1, 3, 4]], [[0, 3], [1, 2, 4]], [[0, 4], [1, 2, 3]]]
        }
            
    hyper_parameters["ExtraTree"] = {
        "estimator__max_depth": Integer(3, 10),
        "estimator__criterion": Categorical(['gini', 'entropy']),
        "estimator__n_estimators": Integer(100, 500),
        "estimator__max_features": Real(0.1, 1, "uniform"),
        "estimator__bootstrap": Categorical([True, False]),
    }

    hyper_parameters["HistGradientBoost"] = {
        "estimator__max_depth": Integer(3, 7),
         "estimator__learning_rate": Real(0.01, 0.5, "uniform"),
         "estimator__max_leaf_nodes": Integer(20, 40),
         "estimator__loss": Categorical(["log_loss"]),
    }
    
    hyper_parameters["LightGBM"] = {
        'estimator__num_leaves': Integer(2, 64),
        'estimator__max_depth': Integer(3, 6),
        "estimator__learning_rate": Real(0.01, 0.5),
        'estimator__min_child_samples': Integer(2, 500),
        'estimator__subsample': Real(0.01, 1., "uniform"),
        'estimator__colsample_bytree': Real(0.01, 1., "uniform"),
        'estimator__reg_alpha': Real(0, 10),
        'estimator__reg_lambda': Real(0,10),
        'estimator__max_bin': [10, 20, 30, 50, 70, 100, 150, 200, 256],
        'estimator__colsample_bytree': Real(0.01, 1., "uniform")
        }

    hyper_parameters["LabelSpreading"] = {
        "estimator__n_neighbors": Integer(5, 50),
        "estimator__alpha": Real(0.1, 0.9),
    }
    
    hyper_parameters["EasyEnsemble"] = {
        "estimator__n_estimators": Integer(2, 100),
        #"estimator__base_estimator": [AdaBoostClassifier(), KNeighborsClassifier(n_neighbors=10)],
    }
    
    hyper_parameters["BalancedBagging"] = {
        "estimator__n_estimators": Integer(2, 100),
        "estimator__max_samples": Real(0.1, 1.),
    }
    
    hyper_parameters["Voting"] = None
    
    hyper_parameters["Stacking"] = None

    hyper_parameters["blender"] = None

    hyper_parameters["sequential"] = None

    return hyper_parameters
