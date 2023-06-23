#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:14:18 2022

@author: nguyens
"""
import numpy as np
from sklearn.metrics import make_scorer, precision_score, roc_auc_score
from sklearn.metrics import balanced_accuracy_score, recall_score, fbeta_score

# make scoring
def My_Score(scoring:str=None, favor_class:any=1):
    score_dict = {
                    "balanced_accuracy": balanced_accuracy_score,
                    "recall": recall_score,
                    "f1_weighted": fbeta_score,
                    "precision": precision_score,
                    "roc_auc": roc_auc_score,
                    }

    if scoring == "roc_auc":
        my_score_ = make_scorer(score_func=score_dict[scoring], needs_proba=True, labels=[0, 1], average="weighted")

    elif scoring == "f1_weighted":
        if favor_class==1 or favor_class=="min false positive":
            my_score_ = make_scorer(score_func=score_dict[scoring], beta=0.5, average='weighted')

        elif favor_class==0 or favor_class=="min false negative":
            my_score_ = make_scorer(score_func=score_dict[scoring], beta=2, average='weighted')

        else:
            my_score_ = make_scorer(score_func=score_dict[scoring], beta=1, average='weighted')

    elif scoring in ["precision", "recall"]:
        my_score_ = make_scorer(score_func=score_dict[scoring])

    else:
        my_score_ = score_dict["balanced_accuracy"]

    return my_score_
