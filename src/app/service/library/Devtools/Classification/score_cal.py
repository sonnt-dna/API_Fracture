#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:16:45 2022

@author: nguyens
"""
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, fbeta_score
from sklearn.metrics import recall_score, precision_score, roc_auc_score

def Score(model=None, X:pd.DataFrame=None, labels:pd.Series=None, scoring:str=None, favor_class:any=None):
    
    """
    This function use to calculate score of model based on input scoring method
     - Input: 
             + model: Classification model 
             + X : Input dataframe
             + labels : a label series
             + scoring: Scoring method [balanced_accuracy, f1_weighted, 
                                        recall_score, precision_score, roc_auc]
     - Output: 
         + model score
    Example:
        #>>> from score_cal import Score
        #>>> score = Score(model, X, y, scoring='f1_weighted')
        
    """
    assert X.shape[1] == model.n_features_in_, f"Model required these features: {model.feature_names_in_}, Input features are: {X.columns}"
    
    if scoring=="balanced_accuracy":
        score = balanced_accuracy_score(labels, model.predict(X))

    elif scoring=="f1_weighted":
        if favor_class==1:
            score = fbeta_score(labels, model.predict(X), beta=0.5, average='weighted')

        elif favor_class==0:
            score = fbeta_score(labels, model.predict(X), beta=2, average='weighted')

        else:
            score = fbeta_score(labels, model.predict(X), beta=1, average='weighted')

    elif scoring=="recall":
        score = recall_score(labels, model.predict(X), average='weighted')

    elif scoring=="precision":
        score = precision_score(labels, model.predict(X), average='weighted')

    else:
        score = roc_auc_score(labels, model.predict_proba(X)[:, 1], average='weighted')
        
    return score

if __name__=="__main__":
	Score()
