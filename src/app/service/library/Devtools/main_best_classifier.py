#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:27:39 2022

@author: nguyens
"""
# =============================================================================
import numpy as np
from Classification.best_classification import ClassifierFinder
import pandas as pd
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings('ignore')

seed = 42
data_path = 'data/220718-newdata.csv'
data = pd.read_csv(data_path)

features = ["SPP", "TORQUE", "FLWPMPS", "ROP", "RPM", "TGAS", "DCALI_FINAL"]
label = 'FRACTURE_ZONE'
data = data.dropna(subset=[label, 'RPM', 'TORQUE'])
#data = data.loc[:10000, :]
features.remove("RPM")
X = data[features]
labels = data[label]

Model, train_score, validation_score, all_scores = ClassifierFinder(
    features=X,
    labels=labels,
    validation_size=0.2,
    scoring='f1_weighted',
    max_train_valid_drop=0.1,
    imbalanced=True,
)

# import joblib
# from Classification.prediction import Prediction
# #Test combine2Models.py
# from Classification.combine2Models import Combine_2Models
# model_path1 = '/Users/nguyens/Documents/Devtools/models/model_LightGBM.joblib'
# model_path2 = '/Users/nguyens/Documents/Devtools/models/well1_RandomForest.pkl'
#
# #call model
#
# model = joblib.load(model_path1)
# #model_2 = joblib.load(model_path2)
#
# # combine 2 model
# preds = model.predict(X)#Combine_2Models(model_1=model_1, model_2=model_2, features=X, threshold=.4, alpha=0.56)
# print(preds[:100])
