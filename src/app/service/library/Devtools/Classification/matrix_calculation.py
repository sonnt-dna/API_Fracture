import pandas as pd
import numpy as np
from numpy import asarray, linspace
from sklearn.metrics import confusion_matrix

def Matrix_Calculation(model_1=None, model_2=None, data:pd.DataFrame=None, feature_names:list=None, 
                       labels:str=None, threshold:float=None, alpha:float=None, well_name:list=None):

    assert len(data) != 0, 'Invalid dataframe'
    df_new = data.copy()

    # Make prediction
    y_preds_1 = model_1.predict_proba(df_new[feature_names])[:, 1]
    df_new["model_1"] = y_preds_1

    y_preds_2 = model_2.predict_proba(df_new[feature_names])[:, 1]
    df_new["model_2"] = y_preds_2

    columns = [col for col in df_new.columns if col.startswith("model")]
    data1 = df_new.loc[:, columns]

    beta = 1 - alpha
    df_new["predictions"] = data1.apply(lambda x: 1 if sum(x[:]*np.asarray([alpha, beta])) >= threshold else 0, axis=1)

    df_new_1 = df_new[df_new.WELL==well_name[0]]
    df_new_2 = df_new[df_new.WELL==well_name[1]]

    # confusion matrix well 1
    confusion_mat_1 = confusion_matrix(df_new_1[labels].astype(int), df_new_1["predictions"].astype(int)).ravel()
    FNR1 = confusion_mat_1[2] / (confusion_mat_1[2]+confusion_mat_1[3]) # class 1 false ratio
    FPR1 = confusion_mat_1[1] / (confusion_mat_1[1]+confusion_mat_1[0]) # class 0 false ratio

    # confusion matrix well 2
    confusion_mat_2 = confusion_matrix(df_new_2[labels].astype(int), df_new_2["predictions"].astype(int)).ravel()
    FNR2 = confusion_mat_2[2] / (confusion_mat_2[2]+confusion_mat_2[3]) # class 1 false ratio
    FPR2 = confusion_mat_2[1] / (confusion_mat_2[1]+confusion_mat_2[0]) # class 0 false ratio

    # Obj
    Obj = FNR1 + FPR1 + FNR2 + FPR2

    return df_new, Obj, FNR1, FPR1, FNR2, FPR2

if __name__=='__main__':
    Matrix_Calculation()
