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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.semi_supervised import LabelSpreading
import xgboost as xgb
#import lightgbm as lgb

def Training(
    data:pd.DataFrame=None, feature_names:list=None, labels:str=None, 
    model_name:str="ExtraTree", hypers:dict=None, 
    scoring:str="balanced_accuracy", imbalanced:bool=None):
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
#        models["LabelSpreading"] = LabelSpreading(kernel='knn')
        models["XGBoost"] = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")
#        models["LightGBM"] = lgb.LGBMClassifier(verbose=-1, metric=None, early_stopping_round=None)

        return models

    # make scoring
    def my_score(scoring):
        score_dict = {
            "balanced_accuracy": balanced_accuracy_score, 
            "recall_score": recall_score, 
            "f1_weighted": f1_score, 
            "precision_score": precision_score, 
            "roc_auc": roc_auc_score,
        }
        if scoring != "balanced_accuracy":
            my_score_ = make_scorer(score_func=score_dict[scoring], greater_is_better=True, average="weighted")
        else:
            my_score_ = "balanced_accuracy"

        return my_score_

    # redefine data
    def get_ready_data(data):
        data1 = data.dropna(how="any", subset=labels)
        if imbalanced:# or len(data1[data1[labels]==1])/len(data1)<=0.3 or len(data1[data1[labels]==0])/len(data1)<=0.3:
            print("Imbalanced dataset!")
            if len(data1[data1[labels]==1])/len(data1)<=0.25:
                data_1 = data1[data1[labels]==1].sample(frac=0.9)
                data_0 = data1[data1[labels]==0].sample(frac=len(data_1)/len(data1)+0.05)
                df_resamples = pd.concat([data_1, data_0], axis=0)
                if model_name == "LabelSpreading":
                    data_neg = data1.drop(df_resamples.index)
                    data_neg[labels] = -1
                    data1 = pd.concat([df_resamples, data_neg], axis=0)
                else:
                    data1 = df_resamples
            else: # len(data1[data1[labels]==0])/len(data1)<=0.3:
                data_0 = data1[data1[labels]==0].sample(frac=0.9)
                data_1 = data1[data1[labels]==1].sample(frac=len(data_0)/len(data1)+0.05)
                df_resamples = pd.concat([data_1, data_0], axis=0)
                if model_name == "LabelSpreading":
                    data_neg = data1.drop(df_resamples.index)
                    data_neg[labels] = -1
                    data1 = pd.concat([df_resamples, data_neg], axis=0)
                else:
                    data1 = df_resamples
        
        elif model_name=="LabelSpreading":# or len(data1[data1[labels]==1])/len(data1)<=0.3 or len(data1[data1[labels]==0])/len(data1)<=0.3:
            if len(data1[data1[labels]==1])/len(data1)<0.5:
                data_1 = data1[data1[labels]==1].sample(frac=0.9)
                data_0 = data1[data1[labels]==0].sample(frac=len(data_1)/len(data1)+0.05)
                df_resamples = pd.concat([data_1, data_0], axis=0)
                data_neg = data1.drop(df_resamples.index)
                data_neg[labels] = -1
                data1 = pd.concat([df_resamples, data_neg], axis=0)
                
            else:
                data_0 = data1[data1[labels]==0].sample(frac=0.9)
                data_1 = data1[data1[labels]==1].sample(frac=len(data_0)/len(data1)+0.05)
                df_resamples = pd.concat([data_1, data_0], axis=0)
                data_neg = data1.drop(df_resamples.index)
                data_neg[labels] = -1
                data1 = pd.concat([df_resamples, data_neg], axis=0)

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

    # get score
    score_ = my_score(scoring)

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
                cv = 5#RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
            else:
                cv = 3#RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
                
            randomsearchcv = HalvingRandomSearchCV(estimator=clf, 
                                        param_distributions=hypers[model_name], 
                                        cv=cv, 
                                        #n_iter=50,
                                        factor=3, 
                                        scoring=score_,
                                        random_state=2022)
            
            # fitting 
            randomsearchcv.fit(x_train, y_train)

            clf = randomsearchcv.best_estimator_

            # printout the accuracy
            print(f"Training Score: {randomsearchcv.best_score_}")
        else:
            clf.fit(x_train, y_train)
            #best_clf = clf
            print(f"Training Score: {balanced_accuracy_score(y_train, clf.predict(x_train))}")
            
    else:
        clf.fit(x_train, y_train)
        #best_clf = clf

    def print_test_score(clf, scoring):
        if scoring=="balanced_accuracy":
            print(f"Model testset {scoring} is: {balanced_accuracy_score(y_test, clf.predict(x_test))}\n")

        if scoring=="f1_weighted":
            print(f"Model testset {scoring} is: {f1_score(y_test, clf.predict(x_test), average='micro')}\n")
    
        if scoring=="recall_score":
            print(f"Model testset {scoring} is: {recall_score(y_test, clf.predict(x_test))}\n")
        
        if scoring=="precision_score":
            print(f"Model testset {scoring} is: {precision_score(y_test, clf.predict(x_test))}\n")
        
        if scoring=="roc_auc":
            print(f"Model testset {scoring} is: {roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1], average='weighted')}\n")

    # print out test set
    print_test_score(clf, scoring)

    from sklearn.metrics import plot_confusion_matrix
    print("Cofunsion maxtrix of full dataset:")
    data = data.dropna(subset=labels)
    X_ = data[feature_names]
    y_ = data[labels]
    plot_confusion_matrix(clf, X_, y_)
    plt.show()
    print("Cofunsion maxtrix of Test dataset:")
    plot_confusion_matrix(clf, x_test, y_test)


    return clf

if __name__ == '__main__':
    Training()