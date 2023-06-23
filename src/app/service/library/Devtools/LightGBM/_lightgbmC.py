"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.
In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.
"""

import numpy as np
import optuna
import lightgbm as lgb
from .score_cal import Score
from sklearn.model_selection import train_test_split

seed=42
def Train_LGBM(features:any=None,
               labels:any=None,
               iterations:int=50,
               scoring:str=None,
               validation_size:float=0.2,
               task:str="binary",
               objectives:any="valid_score", #{0: "valid_score", 1: "train_test_drop"}
               favor_class:any=1, #{0: "min false negative", 1: "min false positive", 2: "balanced"}
               start:any=1, # 1 or True
               show_shap:bool=True,
               ):
    """Construct a function to train a booster.

    Parameters
    ----------
    features : any, {dataframe, array} of predictors
        features use to build model.
    labels : any, {array, Series} of labels
        labels to be predicted.
    iterations : int,
        number of optimization rounds.
    scoring : str,
        scoring method for optimization.
        Available scoring method is sklearn regression metrics: balanced_accuracy, f1_weighted, precision, recall,
                roc_auc, accuracy.
    validation_size: float,
        size of validation set use to evaluate model.
    task : str,
        internal method for xgboost to optimize gradient. Available tasks of lightgbm classification are:
        binary, cross_entropy, multiclass, multiclassova.
    objectives : any, {int or str}
        the way to perform optimization.
        "valid_score" or 0: mean to optimize valid score
        "train_test_drop" or 1: mean to optimize different between train and test score.
    start : any,
        default=1
        is internal param for starting optimize.
    show_shap : bool,
        default=False, show shap chart after tuning if True other while save charts as pictures.

    Returns
    -------
    estimator : object
        The trained model.
    """
    train_x, valid_x, train_labels, valid_labels = train_test_split(features, labels, test_size=validation_size, random_state=seed)
    dtrain = lgb.Dataset(train_x, label=train_labels)
    dvalid = lgb.Dataset(valid_x, label=valid_labels)

    def objective(trial):
        params = {
            "objective": task,
            "verbosity": -1,
            "feature_pre_filter": False,
            #"boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            #"max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 0.5),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 10, 512),
            "feature_fraction": trial.suggest_loguniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_loguniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 4, 10000),
            "scale_pos_weight": trial.suggest_loguniform("scale_pos_weight", 1., 20),
            "tree_learner": trial.suggest_categorical("tree_learner", ["serial", "feature", "data", "voting"]),
            "pos_bagging_fraction": trial.suggest_loguniform("pos_bagging_fraction", 1e-2, 1.),
            # "neg_bagging_fraction": trial.suggest_loguniform("neg_bagging_fraction", 1e-2, 1.),
        }

        model = lgb.train(params, dtrain, num_boost_round=100)
        #model.fit(train_x, train_labels)#, valid_sets=dvalid, num_boost_round=100)
        preds = model.predict(valid_x)
        pred_labels = preds>=0.5
        pred_trains = model.predict(train_x)
        pred_trains = pred_trains>=0.5
        valid_score = Score(valid_labels, pred_labels, scoring=scoring, favor_class=favor_class)
        train_score = Score(train_labels, pred_trains, scoring=scoring, favor_class=favor_class)
        diff = train_score-valid_score
        if objectives==0 or objectives=='valid_score':
            print(f"train-test differs: {diff}")
            return valid_score
        else:
            print(valid_score)
            return np.abs(diff)

    guides = ["balanced_accuracy", "recall", "f1_weighted", "precision", "roc_auc", "R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"]

    if objectives==0 or objectives=='valid_score':
        direction="maximize"
    else:
        direction="minimize"

    if start==1 or start==True:
        study = optuna.create_study(direction=direction, study_name="find best model")
        study.optimize(objective, n_trials=iterations)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    best_params = {
            "objective": task,
            "metric": "binary_error",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
            "early_stopping_rounds": 20,
        }
    best_params.update(study.best_params)
    model = lgb.train(params=best_params, train_set=lgb.Dataset(features, label=labels), valid_sets=(dvalid,), verbose_eval=False)
    preds = model.predict(valid_x)
    pred_labels = preds >= 0.5
    pred_trains = model.predict(train_x)
    pred_trains = pred_trains >= 0.5
    valid_score = Score(valid_labels, pred_labels, scoring=scoring, favor_class=favor_class)
    train_score = Score(train_labels, pred_trains, scoring=scoring, favor_class=favor_class)
    print(f"Train score: {train_score}")
    print(f"Valid score: {valid_score}")
    feature_names = features.columns
    # if show_shap:
    #     import sys
    #     sys.path.append('../Shapley')
    #     from Devtools.Shapley.Cshapley import shapley_importances
    #     shapley_importances(model=model, X=features, feature_names=feature_names, shap_sample_size=1, show_plot=show_shap)
    #
    # else:
    #     import sys
    #     sys.path.append('../Shapley')
    #     from Devtools.Shapley.Cshapley import shapley_importances
    #     shapley_importances(model=model, X=features, feature_names=feature_names, shap_sample_size=1, show_plot=False)

    return model