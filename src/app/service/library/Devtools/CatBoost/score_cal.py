#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:16:45 2022

@author: nguyens
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, fbeta_score, accuracy_score
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.metrics import mean_squared_log_error, mean_poisson_deviance, mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error, max_error, mean_gamma_deviance, mean_tweedie_deviance
from sklearn.metrics import mean_pinball_loss, d2_tweedie_score, d2_pinball_score, d2_absolute_error_score

def Score(y_true:any=None, y_pred:any=None, scoring:str=None, favor_class:any=None):
    
    """
    This function use to calculate score of model based on input scoring method
     - Input:
             + y_true : a label series/array
             + y_pred: a predict series/array
             + scoring: Scoring method [balanced_accuracy, f1_weighted, 
                                        recall_score, precision_score, roc_auc]
             + favor_class: when using f1_weighted as scoring
     - Output: 
         + model score
    Example:
        #>>> from score_cal import Score
        #>>> score = Score(y_true, y_pred, scoring='f1_weighted', favor_class=1)
        
    """
    
    if scoring=="balanced_accuracy":
        score = balanced_accuracy_score(y_true, y_pred)

    elif scoring=="f1_weighted":
        if favor_class==1 or favor_class=="min false positive":
            score = fbeta_score(y_true, y_pred, beta=0.5, average='weighted')

        elif favor_class==0 or favor_class=="min false negative":
            score = fbeta_score(y_true, y_pred, beta=2, average='weighted')

        else:
            score = fbeta_score(y_true, y_pred, beta=1, average='weighted')

    elif scoring=="recall":
        score = recall_score(y_true, y_pred, average='weighted')

    elif scoring=="precision":
        score = precision_score(y_true, y_pred, average='weighted')

    elif scoring=="accuracy":
        score = accuracy_score(y_true, y_pred)

    else:
        score = roc_auc_score(y_true, y_pred, average='weighted')
        
    return score

# make scoring
def RScore(y_true, y_pred, scoring):
    score_dict = {
        "R2": r2_score,
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": root_mean_squared_error,
        "ExVS": explained_variance_score,
        "MSLE": mean_squared_log_error,
        "Poisson": mean_poisson_deviance,
        "MAPE": mean_absolute_percentage_error,
        "MeAE": median_absolute_error,
        "ME": max_error,
        "Gama": mean_gamma_deviance,
        "Tweedie": mean_tweedie_deviance,
        "Pinball": mean_pinball_loss,
        "D2T": d2_tweedie_score,
        "D2Pi": d2_pinball_score,
        "D2A": d2_absolute_error_score
    }

    score = score_dict[scoring]

    return score(y_true, y_pred)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))