# import libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from optuna.integration import OptunaSearchCV
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from .func_score import My_Score
from .hyperparamsoptuna import Get_HyperparamsTuna
from .get_modeloptuna import Get_ModelTuna
import numpy as np

# set seed
seed = 42


def RegressorTuna(
        features: any = None,
        target: any = None,
        validation_size: float = 0.2,
        algorithm: str = 'XGBoost',
        scoring: str = 'MAE',
):
    """
    This function use to train a specific supervised regression algorithm addressed by user.

    1. Inputs
        features :  Input data.
                a dataframe or numpy array of input features

        target :   target vector
                a series or numpy array of target

        algorithm : an algorithm will be trained. The support algorithms are:
                - ElasticnetCV
                - BayesianRidge
                - Poisson
                - Gamma
                - Tweedie
                - LinearSVR
                - Extratree
                - RandomForest
                - Voting
                - Stacking
                - XGBoost (default)
                - LightGBM

        scoring: objective to be optimized. Support scoring are:
                - R2
                - MAE (default): mean absolute error
                - MSE: mean squared error
                - ExV: "explained_variance",
                - MSLE: "neg_mean_squared_log_error",
                - Poisson: "neg_mean_poisson_deviance",
                - MAPE: "neg_mean_absolute_percentage_error",
                - MeAE: "neg_median_absolute_error",
                - ME: "max_error",
                - Gama: "neg_mean_gamma_deviance",
                - Tweedie: "neg_mean_tweedie_deviance",
                - D2T: "d2_tweedie_score",
                - D2Pi: "d2_pinball_score",
                - D2A: "d2_absolute_error_score"

        validation_size: size of data to validate model,
                        default is 0.2


    2. Returns

        - model : trained model
        - train_score :   training score
        - valid_score :   validation score

    3. EXAMPLE
    - To use default parameter, regressor can be called by:

            model, train_score, valid_score = Regressor(features=X, target=y)

    - To use other setting like: train a LightGBM, with scoring is "MSE" and validation_size of 0.3:

            model, train_score, valid_score = binary_classifier(features=X,
                                                                target=y,
                                                                algorithm="LightGBM",
                                                                scoring="MSE",
                                                                validation_size=0.3)
    """

    def warn(*args, **kwargs):
        pass

    assert target.isna().sum() == 0, "Target contain nan value. Please apply dropna with subset is labels column to remove it before feeding to model"

    # split data
    x_train, x_valid, y_train, y_valid, = train_test_split(
        features,
        target,
        test_size=validation_size,
        random_state=seed,
    )

    # Get model and hypers
    hyperparams = Get_HyperparamsTuna()
    hypers = hyperparams[algorithm]
    Models = Get_ModelTuna()
    model = Models[algorithm]

    # build model
    reg = Pipeline(
        steps=[
            ("imputer", IterativeImputer(estimator=
                                         ElasticNetCV(max_iter=100000, tol=1e-2),
                                         random_state=seed,
                                         )),
            ("scaler", MinMaxScaler()),
            ("estimator", model)
        ]
    )

    # Get score dict
    score_dict = {
        "R2": "r2",
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "ExV": "explained_variance",
        "MSLE": "neg_mean_squared_log_error",
        "Poisson": "neg_mean_poisson_deviance",
        "MAPE": "neg_mean_absolute_percentage_error",
        "MeAE": "neg_median_absolute_error",
        "ME": "max_error",
        "Gama": "neg_mean_gamma_deviance",
        "Tweedie": "neg_mean_tweedie_deviance",
        "D2T": "d2_tweedie_score",
        "D2Pi": "d2_pinball_score",
        "D2A": "d2_absolute_error_score"
    }

    # define search
    if hypers:
        search = OptunaSearchCV(
            estimator=reg,
            param_distributions=hypers,
            cv=5,
            max_iter=1000,
            n_trials=100,
            timeout=600,
            verbose=0,
            scoring=score_dict[scoring],
            n_jobs=4,
            random_state=seed,
        )

        # fitting
        search.fit(x_train, y_train)

        reg = search.best_estimator_

        train_score = My_Score(reg, x_train, y_train, scoring)
        valid_score = My_Score(reg, x_valid, y_valid, scoring)

    else:
        reg.fit(x_train, y_train)
        train_score = My_Score(reg, x_train, y_train, scoring)
        valid_score = My_Score(reg, x_valid, y_valid, scoring)

    print(
        f"\t- Status\t: Done!\n\t- Train score\t: {np.round(train_score, 3)}\n\t- Valid score\t: {np.round(valid_score, 3)}")
    return reg, train_score, valid_score