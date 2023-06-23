import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from .score_cal import Score
from sklearn.model_selection import train_test_split
import os
seed = 42


def Train_XGBC(features: any = None,
               labels: any = None,
               iterations: int = 50,
               scoring: str = None,
               validation_size: float = 0.2,
               test_size: float = 0.1,
               validation_set: tuple = None,
               test_set: tuple = None,
               task: str = "binary:logistic",
               # {0: "valid_score", 1: "train_valid_drop"}
               objectives: any = "valid_score",
               # {0: "min false negative", 1: "min false positive", 2: "balanced"}
               favor_class: any = 1,
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
        test_size: float,
            size of test set.
        validation_set: tuple,
            tuple of valid data (X_valid, y_valid)
        test_set: tuple,
            tuple of test data (X_test, y_test)
        task : str,
            internal method for xgboost to optimize loss. Available tasks of xgboost classification are:
            binary:logistic, reg:logistic, binary:logitraw, binary:hinge, multi:softmax, multi:softprob,
            reg:gamma, reg:tweedie.
        objectives : any, {int or str}
            the way to perform optimization.
            "valid_score" or 0: mean to optimize valid score
            "train_test_drop" or 1: mean to optimize different between train and valid score
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
        train_x, train_labels = features, labels
        valid_x, valid_labels = validation_set
        test_x, test_labels = test_set

    elif validation_set and test_size:
        valid_x, valid_labels = validation_set
        train_x, test_x, train_labels, test_labels = train_test_split(features,
                                                                      labels,
                                                                      test_size=test_size,
                                                                      random_state=seed)
    elif test_set and validation_size:
        train_x, valid_x, train_labels, valid_labels = train_test_split(features,
                                                                        labels,
                                                                        test_size=validation_size,
                                                                        random_state=seed)
        test_x, test_labels = test_set

    else:
        train_x, valid_x, train_labels, valid_labels = train_test_split(features,
                                                                        labels,
                                                                        test_size=validation_size,
                                                                        random_state=seed)

        train_x, test_x, train_labels, test_labels = train_test_split(features,
                                                                      labels,
                                                                      test_size=test_size,
                                                                      random_state=seed)

    # if validation_set:
       # train_x, train_labels = features, labels
       # valid_x, valid_labels = validation_set
    # else:
       # train_x, valid_x, train_labels, valid_labels = train_test_split(features, labels, test_size=validation_size, random_state=seed)
    dtrain = xgb.DMatrix(train_x, label=train_labels)
    dvalid = xgb.DMatrix(valid_x, label=valid_labels)
    dtest = xgb.DMatrix(test_x, label=test_labels)

    def objective(trial):
        param = {
            "objective": task,
            "booster": "gbtree",
            "eval_metric": "logloss",
            "lambda": trial.suggest_loguniform("lambda", 1e-2, 10.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-2, 10.0),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.5, 1.),
            "colsample_bynode": trial.suggest_loguniform("colsample_bynode", 0.5, 1.),
            "colsample_bylevel": trial.suggest_loguniform("colsample_bylevel", 0.5, 1.),
            "subsample": trial.suggest_loguniform("subsample", 0.5, 1.),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 10),
            "scale_pos_weight": trial.suggest_loguniform("scale_pos_weight", 1., 20),
            "gamma": trial.suggest_loguniform("gamma", 1e-2, 10.0),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-1, .1),
            "tree_method": trial.suggest_categorical("tree_method", ["exact", "hist"]),
            "num_parallel_tree": trial.suggest_int("num_parallel_tree", 1, 3),
            # "max_leaves": trial.suggest_int("max_leaves", 10, 64),
            # "max_bin": trial.suggest_int("max_bin", 10, 256)
            # "sampling_method": trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"])
        }

        if param["tree_method"] != "exact":
            param["max_leaves"] = trial.suggest_categorical(
                "max_leaves", [21, 31, 41, 51, 61])
            param["max_bin"] = trial.suggest_categorical(
                "max_bin", [50, 100, 150, 200, 256])
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"])

        model = xgb.train(
            param,
            dtrain,
            # evals=[(dtrain, "train"), (dvalid, "valid")],
            # num_boost_round=500,
            # early_stopping_rounds=5,
            verbose_eval=0
        )
        preds = model.predict(dvalid)
        pred_labels = preds > 0.5
        pred_trains = model.predict(dtrain)
        pred_trains = pred_trains > 0.5
        valid_score = Score(valid_labels, pred_labels,
                            scoring=scoring, favor_class=favor_class)
        train_score = Score(train_labels, pred_trains,
                            scoring=scoring, favor_class=favor_class)
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

    guides = ["balanced_accuracy", "recall", "f1_weighted", "precision",
              "roc_auc", "R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"]

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
        "eval_metric": "auc",
        "booster": "gbtree",
    }

    best_params.update(study.best_params)

    if refit:
        x_ = pd.concat([train_x, valid_x])
        label_ = pd.concat([train_labels, valid_labels])
        model = xgb.train(
            best_params,
            xgb.DMatrix(x_, label=label_),
            num_boost_round=500,
            early_stopping_rounds=20,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            verbose_eval=False
        )

    else:
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=500,
            early_stopping_rounds=20,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            verbose_eval=False
        )

    preds = model.predict(dvalid)
    pred_labels = preds > 0.5

    pred_trains = model.predict(dtrain)
    pred_trains = pred_trains > 0.5

    pred_tests = model.predict(dtest)
    pred_tests = pred_tests > 0.5

    valid_score = Score(valid_labels,
                        pred_labels,
                        scoring=scoring,
                        favor_class=favor_class)

    train_score = Score(train_labels,
                        pred_trains,
                        scoring=scoring,
                        favor_class=favor_class)
    test_score = Score(test_labels,
                       pred_tests,
                       scoring=scoring,
                       favor_class=favor_class)

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

    if show_shap:
        import sys
        sys.path.append('../Shapley')
        from Shapley.CShapley import shapley_importances
        shapley_importances(model=model, X=features,
                            feature_names=features.columns,
                            shap_sample_size=1., show_plot=show_shap, saved_dir=saved_dir)

    else:
        import sys
        sys.path.append('../Shapley')
        from Shapley.CShapley import shapley_importances
        shapley_importances(model=model, X=features,
                            feature_names=features.columns,
                            shap_sample_size=1., show_plot=False, saved_dir=saved_dir)

    return model
