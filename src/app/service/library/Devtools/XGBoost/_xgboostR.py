import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from .score_cal import RScore
from sklearn.model_selection import train_test_split
import os
seed = 42


def Train_XGBR(features: any = None,
               target: any = None,
               iterations: int = 50,
               scoring: str = None,
               validation_size: float = 0.2,
               test_size: float = 0.1,
               validation_set: tuple = None,
               test_set: tuple = None,
               task: str = "reg:squarederror",
               base_score:float=None,
               # {0: "valid_score", 1: "train_valid_drop"}
               objectives: any = "valid_score",
               start: any = 1,  # 1 or True
               show_shap: bool = False,
               saved_dir:any=None,
               refit: bool = False,
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
        test_size: float,
            size of test set.
        validation_set: tuple,
            tuple of (X_valid, y_valid)
        test_set: tuple,
            tuple of (X_test, y_test)
        task : str,
            internal method for xgboost to optimize gradient. Available tasks of xgboost regression are:
            reg:squarederror, reg:squaredlogerror, reg:pseudohubererror, count:poisson, survival:cox, survival:aft,
            reg:gamma, reg:tweedie, rank:ndcg, rank:pairwise, rank:map.
        objectives : any, {int or str}
            the way to perform optimization.
            "valid_score" or 0: mean to optimize valid score
            "train_test_drop" or 1: mean to optimize different between train and valid score
        base_score: float
            control under fitting when apply objective is train_test_drop / 1
        start : any,
            default=1
            is internal param for starting optimize
        show_shap : bool,
            default=False, show shap chart after tuning if True other while save charts as pictures
        Returns
        -------
        estimator : object
            The trained model.
        """

    if validation_set and test_set:
        train_x, train_target = features, target
        valid_x, valid_target = validation_set
        test_x, test_target = test_set

    elif validation_set and test_size:
        valid_x, valid_target = validation_set
        train_x, test_x, train_target, test_target = train_test_split(features,
                                                                      target,
                                                                      test_size=test_size,
                                                                      random_state=seed)

    elif test_set and validation_size:
        train_x, valid_x, train_target, valid_target = train_test_split(features,
                                                                        target,
                                                                        test_size=validation_size,
                                                                        random_state=seed)
        test_x, test_target = test_set

    else:
        train_x, valid_x, train_target, valid_target = train_test_split(features,
                                                                        target,
                                                                        test_size=validation_size,
                                                                        random_state=seed)

        train_x, test_x, train_target, test_target = train_test_split(features,
                                                                      target,
                                                                      test_size=test_size,
                                                                      random_state=seed)

    # if validation_set:
        # train_x, train_target = features, target
        # valid_x, valid_target = validation_set
    # else:
        # train_x, valid_x, train_target, valid_target = train_test_split(
        # features, target, test_size=validation_size, random_state=seed)
    dtrain = xgb.DMatrix(train_x, label=train_target)
    dvalid = xgb.DMatrix(valid_x, label=valid_target)
    dtest = xgb.DMatrix(test_x, label=test_target)

    def objective(trial):
        param = {
            "objective": task,
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-2, 5.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-2, 5.0),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-1, 0.10),
        }

        if param["booster"] == "gbtree":
            param["colsample_bytree"] = trial.suggest_loguniform(
                "colsample_bytree", 0.5, 0.9)
            param["colsample_bynode"] = trial.suggest_loguniform(
                "colsample_bynode", 0.5, 0.9)
            param["colsample_bylevel"] = trial.suggest_loguniform(
                "colsample_bylevel", 0.5, 0.9)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-2, 10.0)
            param["num_parallel_tree"] = trial.suggest_int(
                "num_parallel_tree", 1, 5)
            param["subsample"] = trial.suggest_loguniform(
                "subsample", 0.5, 0.9)
            param["tree_method"] = trial.suggest_categorical(
                "tree_method", ["exact", "hist"])
            param["min_child_weight"] = trial.suggest_int(
                "min_child_weight", 5, 10)
            if param["tree_method"] != "exact":
                param["max_leaves"] = trial.suggest_categorical(
                    "max_leaves", [21, 31, 41, 51, 61])
                param["max_bin"] = trial.suggest_categorical(
                    "max_bin", [50, 100, 150, 200, 256])
                param["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"])
        if param["objective"] == "reg:tweedie":
            param["tweedie_variance_power"] = trial.suggest_loguniform(
                "tweedie_variance_power", 1., 1.99)
        if param["objective"] == "reg:pseudohubererror":
            param["huber_slope"] = trial.suggest_loguniform(
                "huber_slope", 0.5, 2)
        if param["booster"] == "gblinear":
            param["updater"] = trial.suggest_categorical(
                "updater", ["shotgun", "coord_descent"])
            param["feature_selector"] = trial.suggest_categorical(
                "feature_selector", ["cyclic", "shuffle"])

        model = xgb.train(
            param,
            dtrain,
            # evals=[(dtrain, "train"), (dvalid, "valid")],
            num_boost_round=100,
            # early_stopping_rounds=5,
            verbose_eval=0
        )
        preds = model.predict(dvalid)
        pred_trains = model.predict(dtrain)
        valid_score = RScore(valid_target, preds, scoring=scoring)
        train_score = RScore(train_target, pred_trains, scoring=scoring)
        diff = train_score-valid_score
        if objectives == 0 or objectives == 'valid_score':
            print(f"train-test differs: {diff}")
            return valid_score
        else:
            if valid_score >= base_score:
                print(valid_score)
                return np.abs(diff)
            else:
                return np.exp(np.abs(diff)) * 10

    guides = ["R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"]

    if scoring in guides:
        if objectives == 0 or objectives == 'valid_score':
            direction = "maximize"
        else:
            direction='minimize'
    else:
        direction = "minimize"

    if start == 1 or start == True:
        study = optuna.create_study(
            direction=direction, study_name="find best model")
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
        "eval_metric": "mae",
    }

    best_params.update(study.best_params)
    if refit:
        x_ = pd.concat([train_x, valid_x])
        target_ = pd.concat([train_target, valid_target])
        model = xgb.train(
            best_params,
            xgb.DMatrix(x_, label=target_),
            num_boost_round=500,
            early_stopping_rounds=10,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            verbose_eval=False
        )
    else:
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=500,
            early_stopping_rounds=10,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            verbose_eval=False
        )

    preds = model.predict(dvalid)
    pred_trains = model.predict(dtrain)
    pred_tests = model.predict(dtest)
    valid_score = RScore(valid_target, preds, scoring=scoring)
    train_score = RScore(train_target, pred_trains, scoring=scoring)
    test_score = RScore(test_target, pred_tests, scoring=scoring)
    print(f"Train score: {train_score}")
    if not refit:
        print(f"Valid score: {valid_score}")
    print(f"Test score: {test_score}")

    if saved_dir is None:
        list_dirs = os.scandir(os.getcwd())
        gate_ = [i for i in list_dirs if i.name=="shap_figures"]
        if len(gate_) == 1:
            saved_dir = os.path.join(os.getcwd(), "shap_figures")
        else:
            os.mkdir(os.path.join(os.getcwd(), "shap_figures"))
            saved_dir = os.path.join(os.getcwd(), "shap_figures")

    # if best_params["booster"] != "gblinear":
    #     if show_shap:
    #         import sys
    #         sys.path.append('../Shapley')
    #         from app.api.fracture.Devtools.Shapley.CShapley import shapley_importances
    #         shapley_importances(model=model, X=features,
    #                             feature_names=features.columns,
    #                             shap_sample_size=1., show_plot=show_shap, saved_dir=saved_dir)
    #
    #     else:
    #         import sys
    #         sys.path.append('../Shapley')
    #         from app.api.fracture.Devtools.Shapley.CShapley import shapley_importances
    #         shapley_importances(
    #             model=model, X=features,
    #             feature_names=features.columns,
    #             shap_sample_size=1., show_plot=False, saved_dir=saved_dir)
    # else:
    #     pass
    return model