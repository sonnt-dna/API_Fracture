#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:27:39 2022

@author: nguyens
"""

import pandas as pd
import pickle

def Prediction(model_path:str=None, X:pd.DataFrame=None, mode:str="predict_proba"):
    """
    mode: "predict", "predict_proba"
    """
    # check model path is not none
    assert model_path != None, "Invalid model path"

    # load model
    model = pickle.load(open(model_path, 'rb'))
    model_name = type(model).__name__

    if model_name=='Pipeline':
        model_name = type(model[-1]).__name__
    else:
        model_name = model_name

    feature_names = model[-1].feature_names_in_

    print(f"Trained model is: {model_name}")
    print(f"Model trained with features: {feature_names}")

    # check input
    if X.shape[1] < len(feature_names):
        print(f"The number of features you feed into model is less than model trained: {feature_names}")
    elif X.shape[1] >= model[-1].n_features_in_:
        print(f"The number of features is higher than model trained: {feature_names}")
        if set(model.feature_names_in_) <= set(X.columns):
            X = X[feature_names]
            # make prediction
            if mode == "predict":
                return model.predict(X)
            else:
                return model.predict_proba(X)[:, 1]
    else:
        return print(f"Model was fitted with features: {feature_names}, Please make sure you passed the right features!")
