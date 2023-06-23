#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nguyens
"""

# =============================================================================
import numpy as np
from Regression.regressionfinderoptuna import RegressorFinderTuna
import pandas as pd
#from Shapley.RShapley import shapley_importances
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings('ignore')

seed = 42
data_path = 'data/data_reg.csv'
data = pd.read_csv(data_path)
features = ["CALI", "DCALI_FINAL", "DTC", "GR", "LLD", "LLS", "NPHI", "VP"]

data = data.dropna(subset=["RHOB"])
data = data.iloc[:5000, :]
X = pd.DataFrame(np.log1p(data[features]), columns=features)
target = data['RHOB']

Model, train_score, valid_score, all_scores = RegressorFinderTuna(
    features=X,
    target=target,
    validation_size=0.2,
    scoring='MAE',
    special_tag="Reg",
)

# printout model
print(Model)

#shapley_importances(model=Model, X=X, shap_sample_size=1, show_plot=False)

# import pickle
#from Regression.prediction import Prediction
# Test combine2Models.py
# from Classification.combine2Models import Combine_2Models
# model_path1 = '/Users/nguyens/Documents/Devtools/models/well1_GradientBoost.pkl'
# model_path2 = '/Users/nguyens/Documents/Devtools/models/well1_RandomForest.pkl'

# call model
# import pickle
# model_1 = joblib.load(model_path1)
# model_2 = joblib.load(model_path2)
#
# # combine 2 model
# preds = Combine_2Models(model_1=model_1, model_2=model_2, features=X, threshold=.4, alpha=0.56)
# print(preds.head())
