"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.
In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.
"""

import numpy as np
import optuna
import catboost as cat
from catboost import Pool
from score_cal import Score
from sklearn.model_selection import train_test_split

seed=42
def Train_CAT(features:any=None,
               labels:any=None,
               iterations:int=50,
               scoring:str=None,
               validation_size:float=0.2,
               task:str="Logloss",
               objectives:any="valid_score", #{0: "valid_score", 1: "train_test_drop"}
               favor_class:any=1, #{0: "min false negative", 1: "min false positive", 2: "balanced"}
               start:any=1, # 1 or True
               show_shap:bool=False,
               ):
    """Construct a function to train a booster.

    Parameters
    ----------
    features : any, {dataframe, array} of predictors
        features use to build model.
    labels : any, {array, Series} of labels
        labels to be predicted.
    iterations : int,
        default = 50, number of optimization rounds.
    scoring : str,
        scoring method for optimization.
        Available scoring method is sklearn regression metrics: balanced_accuracy, f1_weighted, precision, recall,
                roc_auc, accuracy.
    validation_size: float,
        size of validation set use to evaluate model.
    task : str,
        internal method for CatBoost to optimize gradient. Available tasks of CatBoost classification are:
        Logloss, Accuracy, AUC, BalancedAccuracy, F, F1, Precision, Recall, CrossEntropy.
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
    dtrain = Pool(data=train_x, label=train_labels, feature_names=features.columns.to_list())
    dvalid = Pool(data=valid_x, label=valid_labels, feature_names=features.columns.to_list())

    def objective(trial):
        param = {
            "objective": "Logloss",
            "eval_metric": "Accuracy",
            #"bootstrap_type": "Bayesian",
            #"bootstrap_type": trial.suggest_categorical("booststrap_type", ["Bayesian", "MVS"]),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 1.0),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.5),
            "colsample_bylevel": trial.suggest_loguniform("colsample_bylevel", 0.1, 1),
            "mvs_reg": trial.suggest_float("mvs_reg", 0, 50),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 10000),
            "subsample": trial.suggest_loguniform("subsample", 1e-2, 1.),
            "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 1e-2, 50),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.)
        }

        model = cat.train(pool=dtrain, params=param, iterations=100, logging_level='Silent')
        #model.fit(train_x, train_labels)#, valid_sets=dvalid, num_boost_round=100)
        preds = model.predict(dvalid)
        pred_labels = preds>=0.5
        pred_trains = model.predict(dtrain)
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
            "objective": "Logloss",
            "eval_metric": "AUC",
        }
    best_params.update(study.best_params)
    model = cat.train(pool=Pool(features, label=labels), params=best_params, eval_set=dvalid, verbose=0, early_stopping_rounds=10)
    preds = model.predict(dvalid)
    pred_labels = preds >= 0.5
    pred_trains = model.predict(dtrain)
    pred_trains = pred_trains >= 0.5
    valid_score = Score(valid_labels, pred_labels, scoring=scoring, favor_class=favor_class)
    train_score = Score(train_labels, pred_trains, scoring=scoring, favor_class=favor_class)
    print(f"Train score: {train_score}")
    print(f"Valid score: {valid_score}")

    if show_shap:
        import sys
        sys.path.append('../Shapley')
        from Shapley.CShapley import shapley_importances
        shapley_importances(model=model, X=features, feature_names=features.columns, shap_sample_size=1, show_plot=show_shap)
    else:
        import sys
        sys.path.append('../Shapley')
        from Shapley.CShapley import shapley_importances
        shapley_importances(model=model, X=features, feature_names=features.columns, shap_sample_size=1,
                            show_plot=False)
    return model