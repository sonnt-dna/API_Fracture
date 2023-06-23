#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:27:39 2022

@author: nguyens
"""
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
from bayes_opt import BayesianOptimization
from .hyperparams import Get_Hyperparams
from .get_model import Get_Model

seed = 42

def base_func(features:any = None,
              labels:any = None,
              validation_size:float=0.2,
              algorithm:str = "XGBoost",
              ):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import numpy as np
    # split data
    x_train, x_valid, y_train, y_valid, = train_test_split(features,
                                                           labels.astype(int),
                                                           test_size=validation_size,
                                                           random_state=seed,
                                                           stratify=labels.astype(int),
                                                           )

    estimator_function = Get_Model()[algorithm]
    hypers = Get_Hyperparams()[algorithm]

    # Fit the estimator
    estimator_function.fit(x_train, y_train)

    # calculate out-of-the-box roc_score using validation set 1
    probs = estimator_function.predict_proba(x_valid)
    probs = probs[:, 1]
    val1_roc = roc_auc_score(y_valid, probs)

    # return the mean validation score to be maximized
    return np.array([val1_roc]).mean()

