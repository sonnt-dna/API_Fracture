# import libraries
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from .score_cal import Score
from .hyperparams import Get_Hyperparams
from .get_model_opt import Get_Model
import numpy as np

def OptClassifier(
        features:any = None,
        labels:any = None,
        algorithm:str = "XGBoost",
        scoring:str = "f1_weighted",
        validation_size:float = 0.2,
        imbalanced:any=None,
    ):
    """
    This function use to train a specific supervised classifier algorithm addressed by user.

    1. Inputs
        features :  Input data.
                a dataframe or numpy array of input features

        labels :   Labels vector
                a series or numpy array of labels to be classified

        algorithm : an algorithm will be trained. The support algorithms are:
                - Logistic
                - Support Vector Machine
                - RandomForest
                - GradientBoost
                - ExtraTree
                - HistGradientBoost
                - Voting
                - Stacking
                - XGBoost (default)
                - LightGBM
                - BalancedRandomForest
                - EasyEnsemble
                - BalancedBagging

        scoring: objective to be optimized. Support scoring are:
                - balanced_accuracy
                - recall_score
                - f1_weighted (default)
                - precision_score
                - roc_auc

        validation_size: size of data to validate model,
                        default is 0.2

        imbalanced: True when classes are imbalanced,
                            default is None

    2. Returns

        - model : trained model
        - train_score :   training score
        - valid_score :   validation score

    3. EXAMPLE
    - To use default parameter, binary_classifier can be called by:

            model, train_score, valid_score = binary_classifier(features=X, labels=y)

    - To use other setting like: train a LightGBM, with scoring is "roc_auc" and test_size of 0.3:

            model, train_score, valid_score = binary_classifier(features=X,
                                                                labels=y,
                                                                algorithm="LightGBM",
                                                                scoring="roc_auc",
                                                                test_size=0.3)
    """
    # set seed
    seed = 42

    def warn(*args, **kwargs):
        pass
    
    assert labels.isna().sum() == 0, "Labels contain nan value. Please apply dropna with subset is labels column to remove it before feeding to model"

    # split data
    x_train, x_valid, y_train, y_valid, = train_test_split(features,
                                                         labels.astype(int),
                                                         test_size=validation_size,
                                                         random_state=seed,
                                                         stratify=labels.astype(int),
                                                         )

    # Get model and hypers
    hyperparams = Get_Hyperparams()
    hypers = hyperparams[algorithm]
    Models = Get_Model()
    model = Models[algorithm]

    # build model
    def build_model():
        if imbalanced:
            if algorithm not in ["BalancedRandomForest", "EasyEnsemble", "BalancedBagging"]:
                clf = Pipeline(
                    steps=[
                        ("imputer", IterativeImputer(estimator=ElasticNetCV(l1_ratio=0.55, tol=1e-2, max_iter=int(10e6)), random_state=seed)),
                        ("over", SMOTE(sampling_strategy=0.35)),
                        ("under", RandomUnderSampler(sampling_strategy=0.5)),
                        ("scaler", MinMaxScaler()),
                        ("estimator", model)
                    ]
                )
            else:
                clf = Pipeline(
                    steps=[
                        ("imputer", IterativeImputer(estimator=ElasticNetCV(l1_ratio=0.55, tol=1e-2, max_iter=int(10e6)),
                                                     random_state=seed)),
                        ("scaler", MinMaxScaler()),
                        ("estimator", model)
                    ]
                )
        else:
            clf = Pipeline(
                steps=[
                    ("imputer", IterativeImputer(estimator=ElasticNetCV(l1_ratio=0.55, tol=1e-2, max_iter=int(10e6)),
                                                 random_state=seed)),
                    ("scaler", MinMaxScaler()),
                    ("estimator", model)
                ]
            )
        return clf

    # get clf
    clf = build_model()

    # define search
    best_score = 0.5
    max_train_valid_drop = 0.1
    best_train = 0
    best_clf = 0
    try:
        for i in range(50):
            search = RandomizedSearchCV(
                estimator=clf,
                param_distributions=hypers,
                cv=5,
                n_iter=1,
                scoring=scoring,
                n_jobs=4,
                #random_state=seed,
                #search_optimization="bayesian",
                #use_gpu=True,
                #time_budget_s=600,
                #local_dir='../logs'
            )

            # fitting
            search.fit(x_train, y_train)

            clf = search.best_estimator_

            train_score = Score(clf, x_train, y_train, scoring)
            valid_score = Score(clf, x_valid, y_valid, scoring)
            if valid_score >= best_score and abs(train_score - valid_score) <= max_train_valid_drop:
                max_train_valid_drop = abs(train_score - valid_score)
                best_score = valid_score
                best_train = train_score
                best_clf = clf
        if best_train:
            print("\t- Status\t: Done!")
            print(f"\t- Train score\t: {np.round(best_train, 3)}\n\t- Valid score\t: {np.round(best_score, 2)}")

            return best_clf, best_train, best_score
        
        else:
            print(f"\t- Status\t: Failed to tune {algorithm}! Try other algorithm!")
            return clf, 0, 0

    except:
        clf.fit(x_train, y_train)
        train_score = Score(clf, x_train, y_train, scoring)
        valid_score = Score(clf, x_valid, y_valid, scoring)
        if valid_score >= best_score and abs(train_score - valid_score) <= max_train_valid_drop:
            max_train_valid_drop = abs(train_score - valid_score)
            best_score = valid_score
            best_train = train_score
            best_clf = clf
        if best_train:
            print("\t- Status\t: Done!")
            print(f"\t- Train score\t: {np.round(best_train, 3)}\n\t- Valid score\t: {np.round(best_score, 2)}")

            return best_clf, best_train, best_score
        
        else:
            print(f"\t- Status\t: Failed to tune {algorithm}! Try other algorithm!")
            #return clf, train_score, valid_score
        #print("\t- Status\t: Done!")
        #print(f"\t- Train score\t: {np.round(train_score, 3)}\n\t- Valid score\t: {np.round(valid_score, 2)}")

            return clf, 0, 0
