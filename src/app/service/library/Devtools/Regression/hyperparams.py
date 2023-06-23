# import libraries
import numpy as np
from skopt.space import Real, Integer, Categorical

# Get hyper-parameters
def Get_Hyperparams():
    """
    This function get hyperparams
    Outputs:
        - dictionary of hyperparams

    """

    hyper_parameters = dict()
    
    hyper_parameters["LinearRegression"] = {
        "estimator__l1_ratio": Real(0.0, 1., "uniform"),
        "estimator__n_alphas": Integer(1, 200),
        }
    
    hyper_parameters["BayesianRidge"] = {
        "estimator__alpha_init": Real(0.01, 20, "uniform"),
        "estimator__lambda_init": Real(0.01, 20, "uniform"),
        }
    
    hyper_parameters["Poisson"] = {
        "estimator__alpha": Integer(1, 200),
        }
    
    hyper_parameters["Gamma"] = {
        "estimator__alpha": Real(0.01, 200, "uniform"),
        }
    
    hyper_parameters["Tweedie"] = {
        "estimator__power": Real(0, 3, "uniform"),
        "estimator__alpha": Real(0.01, 100, "uniform"),
        "estimator__link": Categorical(["auto", "identity", "log"]),
        }

    hyper_parameters["LinearSVR"] = None#{
        #"estimator__C": Real(0.01, 500, "uniform"),
        #}

    hyper_parameters["RandomForest"] = {
        "estimator__max_depth": [3, 4, 5, 6, 7],
        "estimator__n_estimators": Integer(100, 500),
        "estimator__min_samples_split": Integer(2, 10),
        "estimator__min_samples_leaf": Integer(1, 30),
        "estimator__criterion": Categorical(["squared_error", "absolute_error", "poisson"]),
        "estimator__max_features": Real(0.01, 1, "uniform"),
        }

    hyper_parameters["GradientBoost"] = {
        "estimator__max_depth": [2, 3, 4, 5, 6, 7],
        "estimator__learning_rate": Real(0.001, 0.5, "uniform"),
        "estimator__n_estimators": Integer(100, 200),
        "estimator__subsample": Real(0.01, 1., "uniform"),
        "estimator__min_samples_split": Integer(2, 10),
        "estimator__max_features": Real(0.1, 1, "uniform"),
        "estimator__min_samples_leaf": Integer(1, 10)
        }

    hyper_parameters["XGBoost"] = {
        "estimator__n_estimators": Integer(100, 300),
        "estimator__max_depth": [3, 4, 5, 6],
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
        "estimator__objective": Categorical(["reg:squarederror", "reg:squaredlogerror", 
                                            "reg:gamma", "reg:tweedie", "reg:pseudohubererror"])
        #"estimator__interaction_constraints": [[[0, 1], [2, 3, 4]], [[0, 2], [1, 3, 4]], [[0, 3], [1, 2, 4]], [[0, 4], [1, 2, 3]]]
    }
            
    hyper_parameters["ExtraTree"] = {
        "estimator__max_depth": Integer(3, 7),
        "estimator__criterion": Categorical(["absolute_error", "squared_error"]),
        "estimator__n_estimators": Integer(100, 500),
        "estimator__max_features": Real(0.1, 1, "uniform"),
        "estimator__bootstrap": [True, False],
    }

    hyper_parameters["HistGradientBoost"] = {
        "estimator__max_depth": Integer(3, 7),
         "estimator__learning_rate": Real(0.01, 0.5, "uniform"),
         "estimator__max_leaf_nodes": Integer(20, 40),
         "estimator__loss": Categorical(["squared_error", "absolute_error", "poisson"]),
     }
    
    hyper_parameters["LightGBM"] ={
        'estimator__num_leaves': Integer(2, 100),
        'estimator__max_depth': Integer(3, 6),
        'estimator__learning_rate': Real(0.01, 0.4, "uniform"),
        'estimator__min_child_samples': Integer(2, 500),
        'estimator__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
        'estimator__subsample': Real(0.01, 1., "uniform"),
        'estimator__colsample_bytree': Real(0.01, 1., "uniform"),
        'estimator__reg_alpha': [0.1, 2e-1, 1, 2, 5, 7, 10, 50, 100],
        'estimator__reg_lambda': [.1, 2e-1, 1, 5, 10, 20, 50, 100],
        'estimator__max_bin': [10, 20, 30, 50, 70, 100, 150, 200, 256],
        'estimator__objective': Categorical(["regression", "poisson", "regression_l1", 
                                             "gamma", "tweedie", "quantile","mape", "fair"])
        #'estimator__min_data_in_leaf': Integer(5, 50, 5),
        #'estimator__bagging_fraction': Real(0.01, 1., 40),
        #'estimator__feature_fraction': Real(0.01, 1., 40),
        #'estimator__min_gain_to_split': Real(0.01, 1., 20),
    }

    
    hyper_parameters["Voting"] = None
    
    hyper_parameters["Stacking"] = None
    
    hyper_parameters["blender"] = None

    return hyper_parameters
