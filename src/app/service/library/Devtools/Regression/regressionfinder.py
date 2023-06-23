#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:14:18 2022
@author: nguyens
"""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from .regression import Regressor
from .hyperparams import Get_Hyperparams
from .get_model import Get_Model
import joblib


# from func_score import My_Score

def RegressorFinder(features:any=None, 
                     target:any=None, 
                     validation_size:float=0.2, 
                     scoring:str="MAE", 
                     base_score:float=None,
                     max_train_valid_drop:float=None,
                     saved_dir:any=None,
                     special_tag:str =None):
    # ==========================================================================
    """
    This function use to train and find a best regression algorithms.

    1. Inputs
        features :  Input data.
                a dataframe or numpy array of input features

        target :   target vector
                a series or numpy array of target

        scoring: objective to be optimized. Support scoring are:
                - R2
                - MAE (default): mean absolute error
                - MSE: mean squared error
                - ExV: "explained_variance",
                - MSLE: "neg_mean_squared_log_error",
                - Poisson: "neg_mean_poisson_deviance",
                - MAPE: "neg_mean_absolute_percentage_error",
                - MeAE: "neg_median_absolute_error",
                - ME: "max_error",
                - Gama: "neg_mean_gamma_deviance",
                - Tweedie: "neg_mean_tweedie_deviance",
                - D2T: "d2_tweedie_score",
                - D2Pi: "d2_pinball_score",
                - D2A: "d2_absolute_error_score"

        validation_size: size of data to validate model,
                        default is 0.2
                        
        base_score: desired score to be achived. If base_score to be supplied, finder will prioritize it.
        
        max_train_valid_drop: desired drop between train and validation score. This param will be a constraint in case base_score exists, ortherwise be objective
        
        saved_dir: directory to save model
        
        special_tag: Special tag to save model


    2. Returns

        - model : trained model
        - train_score :   training score
        - valid_score :   validation score
        - scores: score of all trained model

    3. EXAMPLE
    - To use default parameter, regressorfinder can be called by:

            model, train_score, valid_score = RegressorFinder(features=X, target=y)

    - To use other setting like: train a LightGBM, with scoring is "MSE" and validation_size of 0.3:

            model, train_score, valid_score = RegressorFinder(features=X,
                                                              target=y,
                                                              scoring="MSE",
                                                              validation_size=0.3)
    
    """
    # ==========================================================================
    # get current working directory
    def warn(*args, **kwargs):
        pass

    cwd = os.getcwd()
    if saved_dir:
        path = saved_dir
    else:
        if'saved_models' not in os.listdir():
            path = cwd + '/saved_models'
            os.makedirs(path)
        else:
            path = cwd + '/saved_models'

    # Get model and hypers
    hyperparams = Get_Hyperparams()
    models = Get_Model()
    
    # empty dict to store results
    scores_dict = {}
    trained_models = {}
    # =============================================================================
    print("Auto Tuning to select the best model has been started. It might take a while, please wait a while!")
    start = time.time()
    for i, (name, model) in enumerate(models.items()):
        start1 = time.time()
        print(f'{i+1}. Tuning {name} model')
        reg, train_score, valid_score = Regressor(features=features,
                                                target=target,
                                                algorithm=name,
                                                validation_size=validation_size,
                                                scoring=scoring,
                                                )

        scores_dict[name] = [train_score, valid_score, np.abs(train_score - valid_score)]
        trained_models[name] = reg
        print(f"\t- Time consumed\t: {np.round((time.time()-start1)/60, 2)} minutes")
    print(f"\nFine Tune has finished in {np.round((time.time()-start)/60, 2)} minutes. Please wait moment for finding the best model")
    # ==============================================================================
    # create a dataframe of scores_dict
    scores = pd.DataFrame(scores_dict, index=["Train", "Validation", "Drop (Train-Valid)"]).T
    scores = scores.sort_values(by="Validation", ascending=False)
    
    # select best model
    best_valid_score = 0.
    best_train_score = 0.
    best_name = ''
    
    try:
        for name in scores.index:
            if scoring in ["R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"] \
            and scores.loc[name, "Validation"] >= base_score \
            and scores.loc[name, "Drop (Train-Valid)"] <= max_train_valid_drop:
                base_score = scores.loc[name, "Validation"]
                best_train_score = scores.loc[name, "Train"]
                best_valid_score = scores.loc[name, "Validation"]
                best_name = name
            elif scores.loc[name, "Validation"] <= base_score \
            and scores.loc[name, "Drop (Train-Valid)"] <= max_train_valid_drop:
                    base_score = scores.loc[name, "Validation"]
                    best_train_score = scores.loc[name, "Train"]
                    best_valid_score = scores.loc[name, "Validation"]
                    best_name = name
    except:
        try:
            for name in scores.index:
                if scoring in ["R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"] \
                and scores.loc[name, "Validation"] >= base_score:
                        base_score = scores.loc[name, "Validation"]
                        best_train_score = scores.loc[name, "Train"]
                        best_valid_score = scores.loc[name, "Validation"]
                        best_name = name
                elif scores.loc[name, "Validation"] <= base_score:
                        base_score = scores.loc[name, "Validation"]
                        best_train_score = scores.loc[name, "Train"]
                        best_valid_score = scores.loc[name, "Validation"]
                        best_name = name
        except:
            try:
                for name in scores.index:
                    if scoring in ["R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"] \
                    and scores.loc[name, "Drop (Train-Valid)"] >= max_train_valid_drop:
                            max_train_valid_drop = scores.loc[name, "Drop (Train-Valid)"]
                            best_train_score = scores.loc[name, "Train"]
                            best_valid_score = scores.loc[name, "Validation"]
                            best_name = name
                    elif scores.loc[name, "Drop (Train-Valid)"] <= max_train_valid_drop:
                            max_train_valid_drop = scores.loc[name, "Drop (Train-Valid)"]
                            best_train_score = scores.loc[name, "Train"]
                            best_valid_score = scores.loc[name, "Validation"]
                            best_name = name
            except:
                for name in scores.index:
                    if scoring in ["R2", "ExV", "Poisson", "D2T", "D2Pi", "D2A"]:
                        best_train_score = scores.iloc[0, 0]
                        best_valid_score = scores.iloc[0, 1]
                        best_name = name
                    else:
                        best_train_score = scores.iloc[-1, 0]
                        best_valid_score = scores.iloc[-1, 1]
                        best_name = name
                            
    # ==============================================================================
    # save model        
    try:
        
        if special_tag:
            joblib.dump(trained_models[best_name], path + f'/{special_tag}_{best_name}.joblib')

        else:
            joblib.dump(trained_models[best_name], path + f'/{best_name}.joblib')
        
        # summary results
        print(f"\nBest model summary: \n- Best model: {type(trained_models[best_name][-1]).__name__}\n- Best train_score: {best_train_score}\n- Best valid score: {best_valid_score}")
        sns.heatmap(scores, annot=True)
        return trained_models[best_name], best_train_score, best_valid_score, scores

    except: 
        print(f"Try to extend base score: {base_score} or max_train_valid_drop: {max_train_valid_drop}")
        sns.heatmap(scores, annot=True)
        return print(f'Summary results:\n {scores}')