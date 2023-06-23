#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nguyens
"""

# =============================================================================
import numpy as np
from Classification.binary_optuna import Optuna_Classifier
from Classification.score_cal import Score
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings('ignore')

seed = 42
data_path = 'data/220718-newdata.csv'
data = pd.read_csv(data_path)

features = ["DXC", "SPP", "TORQUE", "FLWPMPS", "ROP", "RPM"]

label = 'FRACTURE_ZONE'
data = data.dropna(subset=[label])
X = np.log1p(data[features])
X = pd.DataFrame(X, columns=features)
labels = data[label]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.95, random_state=seed, shuffle=True, stratify=labels)

Model, train_score, valid_score = Optuna_Classifier(
    features=X_train,
    labels=y_train,
    validation_size=0.05,
    scoring='f1_weighted',
    favor_class=1, # {1: 'min false positive', 0: 'min false negative', 'balanced': 'balance both class'}
    algorithm='Support Vector Machine',
    imbalanced=False,
)

print(type(Model[-1]).__name__)
print(Model)
print(f"Test score: {Score(model=Model, X=X_test, labels=y_test, scoring='f1_weighted', favor_class='positive')}")
