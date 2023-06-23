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

        return FNR, FPR, df

    if alpha != None:
        if beta != None:
            assert alpha+beta==1, "Invalid Input of alpha and beta. Condition didn't match: alpha + beta =1"
            FNR, FPR, df = matrix_calculation()
            
            print(f"FN/(FN + TP) is: {FNR}")
            print(f"FP/(FP + TN) is: {FPR}")
            return FNR, FPR, df
