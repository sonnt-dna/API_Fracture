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
from .binary_opt import OptClassifier
from .get_model_opt import Get_Model
import joblib

def OptFinder(features: any = None,
                    labels: any = None,
                    validation_size: float = None,
                    scoring: str = None,
                    base_score: float = 0.5,
                    max_train_valid_drop: float = 0.1,
                    imbalanced: bool=False,
                    special_tag: str=None,
                    saved_dir: str = None,
                    ):
    # ==========================================================================
    """
        This function use to train a specific supervised classifier algorithm addressed by user.

        1. Inputs
            features :  Input data.
                        a dataframe or numpy array of input features

            labels  :   Labels vector
                        a series or numpy array of labels to be classified

                        
            scoring: objective to be optimized. Support scoring are:
                    - balanced_accuracy
                    - recall_score
                    - f1_weighted (default)
                    - precision_score
                    - roc_auc

            validation_size: size of data to validate model,
                        default is 0.2

            base_score: desired validation score that trained model should beat, finder will prioritize.
                        default is 0.5 (choose the best possible)

            max_train_valid_drop: absolute different between train and validation score, This param will be a constraint in case base_score exists, ortherwise be objective
                                  default is 0.1

            imbalanced: True when classes are imbalanced,
                                  default is False
                                  
            saved_dir:   Directory to save model
            
            special_tag:   A special tag to save model

        2. Returns

            - model : the best trained model
            - train_score :   training score of best model
            - valid_score :   validation score of best model
            - trained models score: scores of all models

        3. EXAMPLE

        - To use default parameter, binary_classifier can be called by:

                model, train_score, valid_score = binary_classifier(features=X, labels=y)

        - To use other setting like: train a LightGBM, with scoring is "roc_auc" and validation_size of 0.3:

                model, train_score, valid_score = binary_classifier(features=X,
                                                                    labels=y,
                                                                    algorithm="LightGBM",
                                                                    scoring="roc_auc",
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
        if 'saved_models' not in os.listdir():
            path = cwd + '/saved_models'
            os.makedirs(path)
        else:
            path = cwd + '/saved_models'

    # get models
    models = Get_Model()

    # empty dict to store results
    scores_dict = {}
    trained_models = {}
    # =============================================================================
    print("Auto Tuning to select the best model has been started. It might take a while!")
    start = time.time()
   
    for i, name in enumerate(models.keys()):
        start1 = time.time()
        print(f'{i + 1}. Tuning {name} model')
        clf, train_score, valid_score = OptClassifier(
            features=features,
            labels=labels,
            algorithm=name,
            validation_size=validation_size,
            scoring=scoring,
            imbalanced=imbalanced,
        )

        scores_dict[name] = [train_score, valid_score, np.abs(train_score - valid_score)]
        trained_models[name] = clf
        print(f"\t- Time consumed\t: {np.round((time.time()-start1)/60, 2)} minutes")
        
        #return scores_dict, trained_models

    print(f"\nFine Tune has finished in {np.round((time.time()-start)/60, 2)} minutes. \
        Please wait moment for finding the best model")
    # ==============================================================================

    # select best model
    best_valid_score = base_score
    best_train_score = base_score
    best_name = ''

    # create a dataframe of scores_dict
    scores = pd.DataFrame(scores_dict, index=["Train", "Validation", "Drop (Train-Valid)"]).T
    scores = scores.sort_values(by="Validation", ascending=False)
    
    try:
        for inx in scores.index:
            if scores.loc[inx, "Validation"]>= base_score and scores.loc[inx, "Drop (Train-Valid)"] <= max_train_valid_drop:
                base_score = scores.loc[inx, "Validation"]
                best_valid_score = scores.loc[inx, "Validation"]
                best_train_score = scores.loc[inx, "Train"]
                best_name = inx

        # elif max_train_valid_drop:
        #     if scores.loc[inx, "Drop (Train-Valid)"] <= best_drop and scores.loc[inx, "Validation"]>= best_valid_score:
        #         best_drop = scores.loc[inx, "Drop (Train-Valid)"]
        #         best_valid_score = scores.loc[inx, "Validation"]
        #         best_train_score = scores.loc[inx, "Train"]
        #         best_name = inx
    except:
        try:
            for inx in scores.index:
                if scores.loc[inx, "Validation"]>= base_score and scores.loc[inx, "Drop (Train-Valid)"] <= max_train_valid_drop:
                    base_score = scores.loc[inx, "Validation"]
                    best_valid_score = scores.loc[inx, "Validation"]
                    best_train_score = scores.loc[inx, "Train"]
                    best_name = inx
        except:
            best_valid_score = scores.iloc[0, 1]
            best_train_score = scores.iloc[0, 0]
            best_name = scores.index[0]

    # for name, score in scores_dict.items():
    #     if score[1] >= best_valid_score:
    #         if score[-1] <= best_score_drop_max:
    #             best_valid_score = score[1]
    #             best_train_score = score[0]
    #             best_name = name

    # ==============================================================================

    # printout the best model
    if best_name:
        # save model
        if special_tag:
            joblib.dump(trained_models[best_name], path + "/" + special_tag + "_" + best_name + ".joblib", compress=9)

        else:
            joblib.dump(trained_models[best_name], path + "/" + best_name + ".joblib", compress=9)
        
        # summary results
        print(f"\nBest model summary: \n- Best model: {type(trained_models[best_name][-1]).__name__}\n- Best train_score: {best_train_score}\n- Best valid score: {best_valid_score}")
    
    else:
        print(f"No model satisfies the constraints: {base_score} and {max_train_valid_drop}")
    
    sns.heatmap(scores, annot=True)
        
    return trained_models[best_name], best_train_score, best_valid_score, scores
