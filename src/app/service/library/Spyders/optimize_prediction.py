import pandas as pd
import numpy as np

def Optimize_Models(df:pd.DataFrame=None, strategy="max"):
    columns = [col for col in df.columns if col.startswith("model")]
    data = df.loc[:, columns]
    if strategy=="max":
        df["predictions"] = data.apply(lambda x: max(x[:]), axis=1)
        df["predictions"] = df["predictions"].astype(int)

    else: # strategy="min"
        df["predictions"] = df.apply(lambda x: min(x[:]), axis=1)
        df["predictions"] = df["predictions"].astype(int)

    return df
