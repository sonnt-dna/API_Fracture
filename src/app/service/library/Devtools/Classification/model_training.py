#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:14:18 2022
@author: nguyens
"""

import os
import numpy as np
import pandas as pd
from .training import Training
from .hyperparams import Get_Hyperparams
from .get_model import Get_Model
import joblib


# from func_score import My_Score

def auto_training(X:any = None, labels:any = None, model_dir:str = None,
                  test_size:float = None, scoring:str = None, base_score:float=None,
                  model_name:str = None, special_tag:str = 'model'):
    # ==========================================================================
    """
    This function use to train ML models.

    The model required inputs:
        - X: a dataframe that contains features for train model
        - labels: a pandas series or numpy array contains labels
        - model_dir: directory to save model
        - model_name: a specific model name user want to train. Without specified
        saved directory, program will save model in the folder 'models' in the same directory
        - special_tag: This tag will be used to save model as prefix. For example, 
        special_tag='model1' and model_name is 'XGBoost', model will be saved as 'model_1_XGBoost.pkl'

    The model will return a best model, train score and test score
    
    """
    # ==========================================================================
    # get current working directory
    def warn(*args, **kwargs):
        pass

    cwd = os.getcwd()
    if model_dir:
        path = model_dir
    else:
        if'models' not in os.listdir():
            path = cwd + '/models'
            os.makedirs(path)
        else:
            path = cwd + '/models'

    # get hypers
    hypers = Get_Hyperparams()

    # get models
    models = Get_Model()

    # empty dict to store results
    scores_dict = {}
    trained_models = {}
    # =============================================================================
    if model_name:
        clf, train_score, test_score = Training(data=X,
                                                labels=labels,
                                                test_size=test_size,
                                                model=models[model_name],
                                                scoring=scoring,
                                                hypers=hypers[model_name]
                                                )

        # save model
        joblib.dump(clf, path + "/" + special_tag + "_" + model_name + ".joblib", compress=True)

        return clf, train_score, test_score

    else:
        print("Auto Tuning to select the best model:")
        for i, (name, model) in enumerate(models.items()):
            print(f'{i+1}. Tuning {name} model')
            clf, train_score, test_score = Training(data=X,
                                                    labels=labels,
                                                    model=model,
                                                    test_size=test_size,
                                                    scoring=scoring,
                                                    hypers=hypers[name]
                                                    )

            scores_dict[name] = [train_score, test_score, np.abs(train_score - test_score)]
            trained_models[name] = clf
    # ==============================================================================
    # select best model
    best_score_max = 0.1
    best_score_min = 0.0
    best_score = 0.
    best_name = ''
    if base_score:
        for name, score in scores_dict.items():
            if score[1] >= base_score:
                if score[-1] <= best_score_max and score[-1] >= best_score_min:
                    best_score = score[1]
                    best_name = name
    else:
        for name, score in scores_dict.items():
            if score[1] >= best_score:
                if score[-1] <= best_score_max and score[-1] >= best_score_min:
                    best_score = score[1]
                    best_name = name

    # ==============================================================================
    # save model        
    if special_tag:
        joblib.dump(trained_models[best_name], path + "/" + special_tag + "_" + best_name + ".joblib", compress=True)

    return trained_models[best_name], best_score, pd.DataFrame(scores_dict, index=["Train", "Test", "Delta"])

if __name__=='__main__':
    auto_training()
