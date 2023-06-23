# import libraries
from optuna import trial
from optuna.distributions import IntUniformDistribution, LogUniformDistribution, CategoricalDistribution

# Get hyper-parameters
def Get_HyperparamsTuna():
    """
    This function get hyperparams
    Outputs:
        - dictionary of hyperparams

    """

    hyper_parameters = dict()

    hyper_parameters["Logistic"] = {
        "estimator__C": LogUniformDistribution(0.01, 200)
    }

    #hyper_parameters["Support Vector Machine"] = None#{
        #"estimator__C": LogUniformDistribution(0.01, 100),
        #"estimator__gamma": LogUniformDistribution(1e-3, 10),
        #"estimator__kernel": CategoricalDistribution(["rbf", "poly", "linear", "sigmoid"]),
        #"estimator__degree": IntUniformDistribution(2, 3)
    #}

    hyper_parameters["RandomForest"] = {
        #"estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__n_estimators": IntUniformDistribution(100, 300),
        "estimator__min_samples_split": IntUniformDistribution(10, 50),
        "estimator__min_samples_leaf": IntUniformDistribution(1, 30),
        "estimator__criterion": CategoricalDistribution(["gini", "entropy"]),
        "estimator__max_features": LogUniformDistribution(0.4, 1),
    }

    hyper_parameters["BalancedRandomForest"] = {
        #"estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__n_estimators": IntUniformDistribution(100, 300),
        "estimator__min_samples_split": IntUniformDistribution(4, 50),
        "estimator__min_samples_leaf": IntUniformDistribution(1, 30),
        "estimator__criterion": CategoricalDistribution(["gini", "entropy"]),
        "estimator__max_features": LogUniformDistribution(0.4, 1),
    }

    hyper_parameters["GradientBoost"] = {
        #"estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__learning_rate": LogUniformDistribution(0.001, 0.5),
        "estimator__n_estimators": IntUniformDistribution(100, 300),
        "estimator__subsample": LogUniformDistribution(0.01, 1.),
        "estimator__min_samples_split": IntUniformDistribution(4, 50),
        "estimator__max_features": LogUniformDistribution(0.4, 1),
        "estimator__min_samples_leaf": IntUniformDistribution(1, 10)
    }

    hyper_parameters["XGBoost"] = {
        "estimator__n_estimators": IntUniformDistribution(100, 300),
        "estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__eta": LogUniformDistribution(0.001, 1.),
        "estimator__learning_rate": LogUniformDistribution(.01, 0.4),
        "estimator__colsample_bytree": LogUniformDistribution(0.001, 1),
        "estimator__colsample_bylevel": LogUniformDistribution(0.001, 1),
        "estimator__colsample_bynode": LogUniformDistribution(0.001, 1),
        "estimator__gamma": LogUniformDistribution(0.001, 10),
        "estimator__reg_alpha": LogUniformDistribution(0.001, 10),
        "estimator__reg_lambda": LogUniformDistribution(0.001, 10),
        #"estimator__booster": CategoricalDistribution(["gbtree", "gblinear"]),
        "estimator__subsample": LogUniformDistribution(0.01, 1),
        "estimator__num_parallel_tree": IntUniformDistribution(1, 5),
        # "estimator__interaction_constraints": [[[0, 1], [2, 3, 4]], [[0, 2], [1, 3, 4]], [[0, 3], [1, 2, 4]], [[0, 4], [1, 2, 3]]]
    }

    hyper_parameters["ExtraTree"] = {
        #"estimator__max_depth": IntUniformDistribution(3, 10),
        "estimator__criterion": CategoricalDistribution(['gini', 'entropy']),
        "estimator__n_estimators": IntUniformDistribution(100, 300),
        "estimator__max_features": LogUniformDistribution(0.4, 1),
        "estimator__min_samples_split": IntUniformDistribution(10, 50),
        "estimator__min_samples_leaf": IntUniformDistribution(1, 30),
        "estimator__bootstrap": CategoricalDistribution([True, False]),
    }

    hyper_parameters["HistGradientBoost"] = {
        #"estimator__max_depth": IntUniformDistribution(3, 7),
        "estimator__learning_rate": LogUniformDistribution(0.01, 0.5),
        "estimator__max_leaf_nodes": IntUniformDistribution(20, 50),
        "estimator__min_samples_leaf": IntUniformDistribution(10, 40),
        "estimator__max_bins": IntUniformDistribution(100, 255)
        #"estimator__loss": CategoricalDistribution(["log_loss","binary_crossentropy"])
    }

    hyper_parameters["LightGBM"] = {
        #"estimator__boosting_type": CategoricalDistribution(["gbdt", "rf"]),
        #"estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__reg_alpha": LogUniformDistribution(1e-8, 10.0),
        "estimator__reg_lambda": LogUniformDistribution(1e-8, 10.0),
        "estimator__num_leaves": IntUniformDistribution(2, 512),
        "estimator__colsample_bytree": LogUniformDistribution(0.25, 1.0),
        "estimator__subsample": LogUniformDistribution(0.2, 1.0),
        "estimator__subsample_freq": IntUniformDistribution(1, 10),
        "estimator__min_child_samples": IntUniformDistribution(5, 100),
    }

    hyper_parameters["CatBoost"] = {
        "estimator__objective": CategoricalDistribution(["Logloss", "CrossEntropy"]),
        "estimator__boosting_type": CategoricalDistribution(["Ordered", "Plain"]),
        #"estimator__max_depth": IntUniformDistribution(3, 6),
        "estimator__colsample_bylevel": LogUniformDistribution(1e-2, 1.0),
        "estimator__bootstrap_type": CategoricalDistribution(["Bayesian"]),
        "estimator__bagging_temperature": LogUniformDistribution(1e-2, 10),
        "estimator__learning_rate": LogUniformDistribution(0.1, 0.5),
    }

    hyper_parameters["LabelSpreading"] = {
        "estimator__n_neighbors": IntUniformDistribution(5, 50),
        "estimator__alpha": LogUniformDistribution(0.01, 0.9),
    }

    hyper_parameters["EasyEnsemble"] = {
        "estimator__n_estimators": IntUniformDistribution(2, 100),
        #"estimator__base_estimator": [AdaBoostClassifier(), KNeighborsClassifier(n_neighbors=10)],
    }

    hyper_parameters["BalancedBagging"] = {
        "estimator__n_estimators": IntUniformDistribution(2, 100),
        "estimator__max_samples": LogUniformDistribution(0.001, 1.),
    }

    hyper_parameters["Voting"] = None

    hyper_parameters["Stacking"] = None

    hyper_parameters["blender"] = None

    hyper_parameters["sequential"] = None

    return hyper_parameters
