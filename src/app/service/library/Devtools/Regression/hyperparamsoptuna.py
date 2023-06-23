# import libraries
import numpy as np
from optuna.distributions import IntUniformDistribution, LogUniformDistribution, CategoricalDistribution

# Get hyper-parameters
def Get_HyperparamsTuna():
    """
    This function get hyperparams
    Outputs:
        - dictionary of hyperparams

    """

    hyper_parameters = dict()

    hyper_parameters["LinearRegression"] = {
        "estimator__l1_ratio": LogUniformDistribution(0.001, 1.),
        "estimator__n_alphas": IntUniformDistribution(1, 200),
    }

    hyper_parameters["BayesianRidge"] = {
        "estimator__alpha_init": LogUniformDistribution(0.01, 20),
        "estimator__lambda_init": LogUniformDistribution(0.01, 20),
    }

    hyper_parameters["Poisson"] = {
        "estimator__alpha": IntUniformDistribution(1, 200),
    }

    hyper_parameters["Gamma"] = {
        "estimator__alpha": LogUniformDistribution(0.01, 200),
    }

    hyper_parameters["Tweedie"] = {
        "estimator__power": LogUniformDistribution(1e-8, 3),
        "estimator__alpha": LogUniformDistribution(0.01, 100),
        "estimator__link": CategoricalDistribution(["auto", "identity", "log"]),
    }

    hyper_parameters["Support Vector Machine"] = {
        "estimator__C": LogUniformDistribution(0.01, 500),
        "estimator__gamma": LogUniformDistribution(1e-3, 100),
        "estimator__degree": IntUniformDistribution(1, 5),
        "estimator__kernel": CategoricalDistribution(["rbf", "poly", "linear"]),
    }

    hyper_parameters["RandomForest"] = {
        "estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__n_estimators": IntUniformDistribution(100, 500),
        "estimator__min_samples_split": IntUniformDistribution(2, 10),
        "estimator__min_samples_leaf": IntUniformDistribution(1, 30),
        "estimator__criterion": CategoricalDistribution(["squared_error", "absolute_error", "poisson"]),
        "estimator__max_features": LogUniformDistribution(0.01, 1),
    }

    hyper_parameters["GradientBoost"] = {
        "estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__learning_rate": LogUniformDistribution(0.001, 0.5),
        "estimator__n_estimators": IntUniformDistribution(100, 200),
        "estimator__subsample": LogUniformDistribution(0.01, 1.),
        "estimator__min_samples_split": IntUniformDistribution(2, 10),
        "estimator__max_features": LogUniformDistribution(0.01, 1.),
        "estimator__min_samples_leaf": IntUniformDistribution(1, 10)
    }

    hyper_parameters["XGBoost"] = {
        "estimator__n_estimators": IntUniformDistribution(100, 300),
        "estimator__max_depth": IntUniformDistribution(3,6),
        "estimator__eta": LogUniformDistribution(1e-8, 1.),
        "estimator__learning_rate": LogUniformDistribution(0.01, 0.5),
        "estimator__colsample_bytree": LogUniformDistribution(1e-8, 1),
        "estimator__colsample_bylevel": LogUniformDistribution(1e-8, 1),
        "estimator__colsample_bynode": LogUniformDistribution(1e-8, 1),
        "estimator__gamma": LogUniformDistribution(1e-8, 10),
        "estimator__reg_alpha": LogUniformDistribution(1e-8, 10),
        "estimator__reg_lambda": LogUniformDistribution(1e-8, 10),
        "estimator__booster": CategoricalDistribution(["gbtree", "gblinear"]),
        "estimator__subsample": LogUniformDistribution(0.01, 1),
        "estimator__num_parallel_tree": IntUniformDistribution(1, 5),
        "estimator__scale_pos_weight": LogUniformDistribution(0.8, 1.5),
        "estimator__objective": CategoricalDistribution(["reg:squarederror", "reg:squaredlogerror",
                                "reg:gamma", "reg:tweedie", "reg:pseudohubererror"])
        # "estimator__interaction_constraints": [[[0, 1], [2, 3, 4]], [[0, 2], [1, 3, 4]], [[0, 3], [1, 2, 4]], [[0, 4], [1, 2, 3]]]
    }

    hyper_parameters["ExtraTree"] = {
        "estimator__max_depth": IntUniformDistribution(3, 7),
        "estimator__criterion": CategoricalDistribution(["absolute_error", "squared_error"]),
        "estimator__n_estimators": IntUniformDistribution(100, 500),
        "estimator__max_features": LogUniformDistribution(0.1, 1),
        "estimator__bootstrap": CategoricalDistribution([True, False]),
    }

    hyper_parameters["HistGradientBoost"] = {
        "estimator__max_depth": IntUniformDistribution(3, 7),
        "estimator__learning_rate": LogUniformDistribution(0.01, 0.5),
        "estimator__max_leaf_nodes": IntUniformDistribution(20, 40),
        "estimator__loss": CategoricalDistribution(["squared_error", "absolute_error", "poisson"]),
    }

    hyper_parameters["LightGBM"] = {
        'estimator__num_leaves': IntUniformDistribution(2, 100),
        'estimator__max_depth': IntUniformDistribution(3, 6),
        'estimator__learning_rate': LogUniformDistribution(0.01, 0.5),
        'estimator__bagging_freq': IntUniformDistribution(1, 7),
        'estimator__min_child_samples': IntUniformDistribution(5, 500),
        'estimator__feature_fraction': LogUniformDistribution(0.01, 1.),
        'estimator__bagging_fraction': LogUniformDistribution(0.01, 1.),
        'estimator__lambda_l2': LogUniformDistribution(0.1, 100),
        'estimator__lambda_l1': LogUniformDistribution(0.1, 100),
        'estimator__max_bin': IntUniformDistribution(10, 256),
        'estimator__objective': CategoricalDistribution(["regression", "poisson", "regression_l1",
                                "gamma", "tweedie", "quantile", "mape", "fair"])
        # 'estimator__min_data_in_leaf': Integer(5, 50, 5),
        # 'estimator__bagging_fraction': Real(0.01, 1., 40),
        # 'estimator__feature_fraction': Real(0.01, 1., 40),
        # 'estimator__min_gain_to_split': Real(0.01, 1., 20),
    }

    
    hyper_parameters["Voting"] = None
    
    hyper_parameters["Stacking"] = None

    hyper_parameters["blender"] = None

    hyper_parameters["sequential"] = None

    return hyper_parameters
