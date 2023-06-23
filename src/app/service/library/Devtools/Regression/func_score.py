#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:14:18 2022

@author: nguyens
"""
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.metrics import mean_squared_log_error, mean_poisson_deviance, mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error, max_error, mean_gamma_deviance, mean_tweedie_deviance
from sklearn.metrics import mean_pinball_loss, d2_tweedie_score, d2_pinball_score, d2_absolute_error_score

# make scoring
def My_Score(model, X, y_true, scoring):
    score_dict = {
                "R2": r2_score,
                "MAE": mean_absolute_error,
                "MSE": mean_squared_error,
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
    
    y_preds = model.predict(X)
    score = score_dict[scoring]
    
    return score(y_true, y_preds)
