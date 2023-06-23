"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.
In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.
"""

import numpy as np
import optuna
import catboost as cat
from catboost import Pool
from .score_cal import RScore
import fasttreeshap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

seed=42
def Train_CATR(features:any=None,
               target:any=None,
               iterations:int=50,
               scoring:str=None,
               validation_size:float=0.2,
               task:str="MAE",
               objectives:any="valid_score", #{0: "valid_score", 1: "train_test_drop"}
               show_shap:bool=False,
               ):
    """Construct a function to train a booster.

        Parameters
        ----------
        features : any, {dataframe, array} of predictors
            features use to build model.
        target : any, {array, Series} of target
            continuous target to be predicted.
        iterations : int,
            number of optimization rounds.
        scoring : str,
            scoring method for optimization.
            Available scoring method is sklearn regression metrics: R2 (r2), MAE, MSE, RMSE, MAPE, Poisson, Tweedie, MeAE,
                ExVS, MSLE, ME, Gama, D2T, D2Pi, D2A.
        validation_size: float,
            size of validation set use to evaluate model.
        task : str,
            internal method for catboost to optimize gradient. Available tasks of CatBoost regression are:
            MAE, MAPE, Poisson, Quantile, RMSE, Tweedie, R2, MSLE, MedianAbsoluteError, Huber.
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
    train_x, valid_x, train_target, valid_target = train_test_split(features, target, test_size=validation_size, random_state=seed)
    dtrain = Pool(data=train_x, label=train_target, feature_names=features.columns.to_list())
    dvalid = Pool(data=valid_x, label=valid_target, feature_names=features.columns.to_list())

    def objective(trial):
        param = {
            "objective": task,
            "eval_metric": task,
            "bootstrap_type": "Bayesian",
            #"bootstrap_type": trial.suggest_categorical("booststrap_type", ["Bayesian", "MVS"]),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 1.0),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.5),
            #"colsample_bylevel": trial.suggest_loguniform("colsample_bylevel", 0.1, 1),
            "mvs_reg": trial.suggest_float("mvs_reg", 0, 50),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 10000),
            #"subsample": trial.suggest_loguniform("subsample", 1e-2, 1.),
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 0, 12),
        }

        model = cat.train(pool=dtrain, params=param, iterations=100, logging_level='Silent')
        #model.fit(train_x, train_labels)#, valid_sets=dvalid, num_boost_round=100)
        preds = model.predict(dvalid)
        pred_trains = model.predict(dtrain)
        valid_score = RScore(valid_target, preds, scoring=scoring)
        train_score = RScore(train_target, pred_trains, scoring=scoring)
        diff = train_score-valid_score
        #scores = ["R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"]
        if objectives==0 or objectives=='valid_score':
            print(f"train-test differs: {diff}")
            return valid_score
        else:
            print(valid_score)
            return np.abs(diff)

    guides = ["R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"]

    if scoring in guides:
        if objectives==0 or objectives=='valid_score':
            direction="maximize"
    else:
        direction="minimize"

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
            "eval_metric": task,
        }
    best_params.update(study.best_params)
    model = cat.train(pool=Pool(features, label=target), params=best_params, eval_set=dvalid, verbose=0, early_stopping_rounds=5)
    preds = model.predict(dvalid)
    pred_trains = model.predict(dtrain)
    valid_score = RScore(valid_target, preds, scoring=scoring)
    train_score = RScore(train_target, pred_trains, scoring=scoring)
    print(f"Train score: {train_score}")
    print(f"Valid score: {valid_score}")


    shap_values = model.get_feature_importance(Pool(features, label=target), type='ShapValues')

    #expected_value = shap_values[0, -1]
    shap_values = shap_values[:, :-1]
    fasttreeshap.summary_plot(shap_values, features)
    if show_shap:
        f = plt.gcf()
        plt.title(f"Feature importances of Catboost model")
        plt.savefig(f"Catboost_summary_plot.png", bbox_inches='tight')
        plt.close()
        # visualize the first prediction's explanation
        #fasttreeshap.force_plot(expected_value, shap_values[-1,:], features[-1,:])
    for feature in features.columns:
        fasttreeshap.dependence_plot(feature, shap_values, features)
        if show_shap:
            f = plt.gcf()
            plt.title(f"Catboost_{feature}_features dependency plot")
            plt.savefig(f"Catboost_{feature}_features dependency plot.png", bbox_inches='tight')
            plt.close()
    return model