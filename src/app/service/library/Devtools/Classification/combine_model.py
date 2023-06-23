import pandas as pd
import numpy as np

def Combine_Model(model_1=None, model_2=None, data:pd.DataFrame=None, feature_names:list=None, 
                       threshold:float=None, alpha:float=None):

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

    return df_new.drop(columns=["model_1", "model_2"])

if __name__=='__main__':
    Combine_Model()
