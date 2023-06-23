import pandas as pd
import numpy as np
from numpy import mean

def Optimize_Threshold(df:pd.DataFrame=None, threshold=0.5, strategy="max"):
    columns = [col for col in df.columns if col.startswith("model")]
    data = df.loc[:, columns]
    if strategy=="max":
        df["predictions"] = data.apply(lambda x: 1 if max(x[:]) >= threshold else 0, axis=1)

    elif strategy=="min":
        df["predictions"] = data.apply(lambda x: 1 if min(x[:]) >= threshold else 0, axis=1)

    else: # strategy is mean
        df["predictions"] = data.apply(lambda x: 1 if mean(x[:]) >= threshold else 0, axis=1)

    return df