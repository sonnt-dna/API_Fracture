# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.utils.fixes import loguniform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.semi_supervised import LabelSpreading
import xgboost as xgb
#import lightgbm as lgb

# Get hyper-parameters
def get_hyperparams():
    """
    This function get hyperparams
    Outputs:
        - dictionary of hyperparams

    """

    hyper_parameters = dict()
    hyper_parameters["LogisticRegression"] = {
        "estimator__C": np.random.uniform(0.01, 100, 50)}

    # hyper_parameters["Support Vector Machine"] = {"estimator__C": np.random.uniform(0.01, 500, 100),
    # "estimator__gamma": np.random.uniform(1e-3, 100, 100),
    #                            }

    hyper_parameters["RandomForest"] = {"estimator__max_depth": [3, 4, 5, 6, 7],
                                        "estimator__n_estimators": np.arange(50, 200, 10),
                                        "estimator__min_samples_split": np.arange(2, 10, 1),
                                        "estimator__min_samples_leaf": np.arange(1, 30, 2),
                                        "estimator__criterion": ["gini", "entropy"],
                                        "estimator__max_features": np.random.uniform(0.01, 1, 10),
                                        }

    hyper_parameters["GradientBoost"] = {"estimator__max_depth": [2, 3, 4, 5, 6, 7],
                                        "estimator__learning_rate": np.random.uniform(0.001, 0.5, 10),
                                        "estimator__n_estimators": np.arange(50, 200, 10),
                                        #"estimator__subsample": np.random.uniform(0.01, 1., 10),
                                        #"estimator__min_samples_split": np.arange(2, 10, 1),
                                        #"estimator__max_features": np.random.uniform(0.1, 1, 10),
                                        #"estimator__min_samples_leaf": np.arange(1, 10, 1)
                                        }

    hyper_parameters["XGBoost"] =       {"estimator__n_estimators": np.arange(50, 300, 10),
                                        "estimator__max_depth": [3, 5, 6],
                                        "estimator__eta": np.random.uniform(0, 1., 10),
                                        "estimator__learning_rate": np.random.uniform(0.01, 0.4, 10),
                                        "estimator__colsample_bytree": np.random.uniform(0., 1, 10),
                                        "estimator__colsample_bylevel": np.random.uniform(0., 1, 10),
                                        "estimator__colsample_bynode": np.random.uniform(0., 1, 10),
                                        "estimator__gamma": np.random.uniform(0., 10, 100),
                                        "estimator__booster": ["gbtree"],
                                        "estimator__subsample": np.linspace(0.01, 1, 10),
                                        "estimator__num_parallel_tree": [1, 2, 3, 4, 5],
                                        }
            
    hyper_parameters["ExtraTree"] =     {"estimator__max_depth": np.arange(3, 20, 1),
                                        "estimator__criterion": ['gini', 'entropy'],
                                        "estimator__n_estimators": np.arange(100, 500, 10),
                                        "estimator__max_features": np.random.uniform(0.1, 1, 10),
                                        "estimator__bootstrap": [True, False],
                                        }

    hyper_parameters["HistGradientBoost"] = {"estimator__max_depth": np.arange(3, 7, 1),
                                            "estimator__learning_rate": np.random.uniform(0.01, 0.5, 10),
                                            "estimator__max_leaf_nodes": np.arange(20, 40, 1),
                                            "estimator__loss": ["log_loss", "auto", "binary_crossentropy"],
                                            }
    
    hyper_parameters["LightGBM"] ={'estimator__num_leaves': np.arange(2, 50, 2), 
                                'estimator__min_child_samples': np.arange(100, 500, 10), 
                                'estimator__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                                'estimator__subsample': np.random.uniform(0.01, 1., 10), 
                                'estimator__colsample_bytree': np.random.uniform(0.01, 1., 10),
                                'estimator__reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                                'estimator__reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                                }

    hyper_parameters["LabelSpreading"] = {
        "estimator__n_neighbors": np.arange(5, 50, 1),
        "estimator__alpha": np.random.uniform(0.1, 0.9, 10),
    }

    return hyper_parameters

def training(
    data:pd.DataFrame=None, feature_names:list=None, labels:str=None, 
    model_name:str="ExtraTree", hypers:dict=None, 
    scoring:str="balanced_accuracy", unbalanced:bool=None):
    """
    scoring: "balanced_accuracy", "recall_score", "f1_weighted", "precision_score", "roc_auc"

    """
    # set seed
    seed=42

    # get model
    def get_model():
        # define model list
        models = {}
        models["ExtraTree"] = ExtraTreesClassifier(random_state=42)
        models["RandomForest"] = RandomForestClassifier(class_weight="balanced", random_state=42)
        models["GradientBoost"] = GradientBoostingClassifier(random_state=42)
        models["Voting"] = VotingClassifier(estimators=[("ExtraTree", ExtraTreesClassifier()), 
                                                        ("RandomForest", RandomForestClassifier()), 
                                                        ("GradientBoost", GradientBoostingClassifier())], 
                                            voting="soft")
        models["Stacking"] = StackingClassifier(estimators=[("gbt", GradientBoostingClassifier()), 
                                                            ("rf", RandomForestClassifier())],
                                                final_estimator=ExtraTreesClassifier(random_state=42))
        models["HistGradientBoost"] = HistGradientBoostingClassifier()
        models["LabelSpreading"] = LabelSpreading(kernel='knn')
        models["XGBoost"] = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")
        models["LightGBM"] = lgb.LGBMClassifier(verbose=-1, metric=None, early_stopping_round=None)

        return models

    # redefine data
    def get_ready_data(data):
        data1 = data.dropna(how="any", subset=labels)
        if unbalanced==True:# or len(data1[data1[labels]==1])/len(data1)<=0.3 or len(data1[data1[labels]==0])/len(data1)<=0.3:
            print("Inbalanced dataset!")
            if len(data1[data1[labels]==1])/len(data1)<=0.25:
                data_1 = data1[data1[labels]==1].sample(frac=0.9)
                data_0 = data1[data1[labels]==0].sample(frac=len(data_1)/len(data1)+0.05)
                df_resamples = pd.concat([data_1, data_0], axis=0)
                if model_name == "LabelSpreading":
                    data_neg = data1.drop(df_resamples.index)
                    data_neg[label] = -1
                    data1 = pd.concat([df_resamples, data_neg], axis=0)
                else:
                    data1 = df_resamples
            else: # len(data1[data1[labels]==0])/len(data1)<=0.3:
                data_0 = data1[data1[labels]==0].sample(frac=0.9)
                data_1 = data1[data1[labels]==1].sample(frac=len(data_0)/len(data1)+0.05)
                df_resamples = pd.concat([data_1, data_0], axis=0)
                if model_name == "LabelSpreading":
                    data_neg = data1.drop(df_resamples.index)
                    data_neg[label] = -1
                    data1 = pd.concat([df_resamples, data_neg], axis=0)
                else:
                    data1 = df_resamples
        
        else:
            data1 = data1

        return data1
        
    # define X and y
    df_ml = get_ready_data(data)
    X = df_ml[feature_names]
    y = df_ml[labels]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # get model
    models = get_model()
    model = models[model_name]

    # build model
    clf = Pipeline(
        steps=[
            ("imputer", IterativeImputer(estimator=BayesianRidge(), random_state=seed, max_iter=100)),
            ("scaler", MinMaxScaler()),
            ("estimator", model)
        ]
    )

    # define search
    if hypers:
        if model_name not in ["Voting", "Stacking"]:
            if len(x_train) >=5000:
                cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
            else:
                cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
                
            randomsearchcv = GridSearchCV(estimator=clf, 
                                        param_grid=hypers[model_name], 
                                        cv=cv, 
                                        #n_iter=50, 
                                        scoring=scoring,
                                        random_state=2022)
            
            # fitting 
            randomsearchcv.fit(x_train, y_train)

            best_clf = randomsearchcv.best_estimator_

            # printout the accuracy
            print(f"Training Score: {randomsearchcv.best_score_}")
        else:
            clf.fit(x_train, y_train)
            best_clf = clf
            print(f"Training Score: {balanced_accuracy_score(y_train, best_clf.predict(x_train))}")
            
    else:
        clf.fit(x_train, y_train)
        best_clf = clf

    def print_test_score(best_clf, scoring):
        if scoring=="balanced_accuracy":
            print(f"Model testset {scoring} is: {balanced_accuracy_score(y_test, best_clf.predict(x_test))}")

        if scoring=="f1_weighted":
            print(f"Model testset {scoring} is: {f1_score(y_test, best_clf.predict(x_test))}")
    
        if scoring=="recall_score":
            print(f"Model testset {scoring} is: {recall_score(y_test, best_clf.predict(x_test))}")
        
        if scoring=="precision_score":
            print(f"Model testset {scoring} is: {precision_score(y_test, best_clf.predict(x_test))}")
        
        if scoring=="roc_auc":
            print(f"Model testset {scoring} is: {roc_auc_score(y_test, best_clf.predict_proba(x_test)[:, 1], average='weighted')}")

    # print out test set
    print_test_score(best_clf, scoring)

    from sklearn.metrics import plot_confusion_matrix
    print("Cofunsion maxtrix of full dataset:")
    data = data.dropna(subset=labels)
    X_ = data[feature_names]
    y_ = data[labels]
    plot_confusion_matrix(best_clf, X_, y_)
    plt.show()
    
    from sklearn.metrics import classification_report
    print(classification_report(best_clf["FRACTURE_ZONE"].astype(int), classifier_1.predict(best_clf[features])))

    return best_clf

    

def Prediction(trained_models:list=None, data:pd.DataFrame=None, feature_names:list=None, mode:str="predict"):
    """
    mode: "predict", "predict_proba"
    """
    if mode == "predict":
        for i, model in enumerate(trained_models):
            y_preds = model.predict(data[feature_names])
            data[f"model_{i}"] = y_preds

    else:
        for i, model in enumerate(trained_models):
            y_preds = model.predict_proba(data[feature_names])
            data[f"model_{i}"] = y_preds
    
    return data