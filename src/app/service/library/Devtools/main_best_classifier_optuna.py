#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nguyens
"""
# =============================================================================
import numpy as np
from Classification.best_optuna import OptunaFinder
from Classification.score_cal import Score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
import pandas as pd
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
impute = IterativeImputer(estimator=ElasticNetCV(l1_ratio=0.55, tol=1e-2, max_iter=int(10e6)), random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.95, random_state=seed, shuffle=True, stratify=labels)
X_train, X_test = impute.fit_transform(X_train), impute.transform(X_test)

Model, train_score, validation_score, all_scores = OptunaFinder(
    features=X_train,
    labels=y_train,
    validation_size=0.05,
    scoring='f1_weighted',
    favor_class=1, # {1: 'min false positive', 0: 'min false negative', 'balanced': 'balance both class'}
    base_score=0.8,
    max_train_valid_drop=0.1,
    imbalanced=None,
)

print(type(Model[-1]).__name__)
print(Model)
print(f"Test score: {Score(model=Model, X=X_test, labels=y_test, scoring='f1_weighted', favor_class=1)}")
