import pandas as pd
import numpy as np
from numpy import asarray, linspace
from sklearn.metrics import confusion_matrix

def Optimize_Weights(df:pd.DataFrame=None, threshold:float=0.5, alpha:float=0.5,
                    beta:float=0.5, well_name:str=None, labels:str=None, fnr:float=0.2, fpr:float=0.2, tfr:float=None, strategy:int=2):
    """
    strategy: {0: class 0, 1: class 1, 2: auto (default)}
    """
    assert len(df) != 0, 'Invalid dataframe'
    
    def matrix_calculation(data=df, threshold=threshold, alpha=alpha, beta=beta, well_name=well_name, labels=labels):
        columns = [col for col in df.columns if col.startswith("model")]
        data = df.loc[:, columns]
        df_new = df.copy()
        try:
            assert alpha+beta==1
            df_new["predictions"] = data.apply(lambda x: 1 if sum(x[:]*np.asarray([alpha, beta])) >= threshold else 0, axis=1)
        except:
            beta = 1 - alpha
            df_new["predictions"] = data.apply(lambda x: 1 if sum(x[:]*np.asarray([alpha, beta])) >= threshold else 0, axis=1)

        # confusion matrix
        confusion_mat = confusion_matrix(df_new[labels].astype(int), df_new["predictions"].astype(int)).ravel()
        FNR = confusion_mat[2] / (confusion_mat[2]+confusion_mat[3]) # class 1 false ratio
        FPR = confusion_mat[1] / (confusion_mat[1]+confusion_mat[0]) # class 0 false ratio

        return FNR, FPR, df_new

    if alpha != None:
        if beta != None:
            assert alpha+beta==1, "Invalid Input of alpha and beta. Condition didn't match: alpha + beta =1"
            FNR, FPR, df = matrix_calculation()
            
            print(f"FN/(FN + TP) is: {FNR}")
            print(f"FP/(FP + TN) is: {FPR}")
            return FNR, FPR, df_new

    else:
        # define gate
        fnr = fnr # false negative ratio (FN/(FN+TP))
        fpr = fpr # false positive ratio (FP/(FP+TN))
        tfr = tfr # total false rate
        #threshold = 0.56 # > threshold will be class 1 ortherwise 0

        # empty list to store values
        possible_alpha_list = {}

        # loop
        best_fnr_score = fnr
        best_fpr_score = fpr
        best_alpha = 0.
        best_beta = 0.
        best_threshold = 0.
        balanced_delta=0.05
        for alpha1 in linspace(0., 1., 20):
            try:
                beta1 = 1 - alpha1
                FNR, FPR, df = matrix_calculation(alpha=alpha1, beta=beta1, threshold=threshold)
                if FNR <= fnr and FPR <= fpr:
                    possible_alpha_list[threshold] = [alpha1, FNR, FPR]
                    if tfr:
                        if FNR + FPR <= tfr:
                            if strategy==2:
                                if abs(FNR-FPR) <= balanced_delta:
                                    balanced_delta = abs(FNR-FPR)
                                    best_fnr_score = FNR
                                    best_fpr_score = FPR
                                    best_alpha = alpha1
                                    best_beta = beta1
                                    best_threshold = threshold
                                    df_best = df
                            elif strategy==1:
                                if FNR <= best_fnr_score:
                                    best_fnr_score = FNR
                                    best_fpr_score = FPR
                                    best_alpha = alpha1
                                    best_beta = beta1
                                    best_threshold = threshold
                                    df_best = df
                            else:
                                if FPR <= best_fpr_score:
                                    best_fnr_score = FNR
                                    best_fpr_score = FPR
                                    best_alpha = alpha1
                                    best_beta = beta1
                                    best_threshold = threshold
                                    df_best = df
                                
                    else:
                        if strategy==2:
                            if abs(FNR-FPR) <= balanced_delta:
                                balanced_delta = abs(FNR-FPR)
                                best_fnr_score = FNR
                                best_fpr_score = FPR
                                best_alpha = alpha1
                                best_beta = beta1
                                best_threshold = threshold
                                df_best = df
                        elif strategy==1:
                            if FNR <= best_fnr_score:
                                best_fnr_score = FNR
                                best_fpr_score = FPR
                                best_alpha = alpha1
                                best_beta = beta1
                                best_threshold = threshold
                                df_best = df
                        else:
                            if FPR <= best_fpr_score:
                                best_fnr_score = FNR
                                best_fpr_score = FPR
                                best_alpha = alpha1
                                best_beta = beta1
                                best_threshold = threshold
                                df_best = df
            except:
                for threshold1 in linspace(0.2, 0.8, 30):
                    beta1 = 1 - alpha1
                    FNR, FPR, df = matrix_calculation(alpha=alpha1, beta=beta1, threshold=threshold1)
                    if FNR <= fnr and FPR <= fpr:
                        possible_alpha_list[threshold1] = [alpha1, FNR, FPR]
                        if tfr:
                            if FNR + FPR <= tfr:
                                if FNR + FPR <= tfr:
                                    if strategy==2:
                                        if abs(FNR-FPR) <= balanced_delta:
                                            balanced_delta = abs(FNR-FPR)
                                            best_fnr_score = FNR
                                            best_fpr_score = FPR
                                            best_alpha = alpha1
                                            best_beta = beta1
                                            best_threshold = threshold1
                                            df_best = df
                                    elif strategy==1:
                                        if FNR <= best_fnr_score:
                                            best_fnr_score = FNR
                                            best_fpr_score = FPR
                                            best_alpha = alpha1
                                            best_beta = beta1
                                            best_threshold = threshold1
                                            df_best = df
                                    else:
                                        if FPR <= best_fpr_score:
                                            best_fnr_score = FNR
                                            best_fpr_score = FPR
                                            best_alpha = alpha1
                                            best_beta = beta1
                                            best_threshold = threshold1
                                            df_best = df
                                
                        else:
                            if strategy==2:
                                if abs(FNR-FPR) <= balanced_delta:
                                    balanced_delta = abs(FNR-FPR)
                                    best_fnr_score = FNR
                                    best_fpr_score = FPR
                                    best_alpha = alpha1
                                    best_beta = beta1
                                    best_threshold = threshold1
                                    df_best = df

                            elif strategy==1:
                                if FNR <= best_fnr_score:
                                    best_fnr_score = FNR
                                    best_fpr_score = FPR
                                    best_alpha = alpha1
                                    best_beta = beta1
                                    best_threshold = threshold1
                                    df_best = df

                            elif strategy==0:
                                if FPR <= best_fpr_score:
                                    best_fnr_score = FNR
                                    best_fpr_score = FPR
                                    best_alpha = alpha1
                                    best_beta = beta1
                                    best_threshold = threshold1
                                    df_best = df
                            
        if len(possible_alpha_list)==0:
            print("Failed to optimize, Please increase fnr and fpr and try again!")
            return fnr, fpr

        else:
            if threshold:
                print(f"Optimization Success!\nBest parameters are: \n - Best alpha is: {best_alpha} \n - Best beta is: {best_beta}")
                print(f" - Best FNR is: {best_fnr_score}\n - Best FPR is: {best_fpr_score}")
            else:
                print(f"Optimization Success!\nBest parameters are: \n - Best alpha is: {best_alpha} \
                      \n - Best beta is: {best_beta} \n - Best threshold is: {best_threshold}")
                print(f" - Best FNR is: {best_fnr_score}\n - Best FPR is: {best_fpr_score}")
                 
            df_possible_alpha = pd.DataFrame(possible_alpha_list,
                                            index=["Alpha", "FN/(FN+TP)", "FP/(FP+TN)"]).T.reset_index()
            df_possible_alpha.columns = ["Threshold", "Alpha", "FN/(FN+TP)", "FP/(FP+TN)"]

            return df_possible_alpha, df_best
