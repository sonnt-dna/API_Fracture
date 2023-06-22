
# ---------------------- Define schemas of Data -------------------------

from typing import List, Optional
from pydantic import BaseModel

class input_csv_schema(BaseModel):
    DEPT:           float
    BASEMENT:       float
    DCALI_FINAL:    float
    DXC:            float
    FLWPMPS:        float
    FRACTURE_ZONE:  float
    ROP:            float
    RPM:            float
    SPP:            float
    TGAS:           float
    TORQUE:         float
    WELL:           str
    WOB:            float

class input_params_schema(BaseModel):
    Modeling_result:    str
    Predicted_Results:  str
    algorithm:          str
    scoring:            str
    objective:          str
    show_shap:          str
    iteration:          str
    feature:            list[str]
    Testing_score:      str
    target:             str
    target2:            str
    
# ------------------------ Processing --------------------------------

def predict(full_df, parameter):

    import pandas as pd
    from sklearn.model_selection import train_test_split
    import warnings
    import json

    pd.set_option('display.max_columns', 100)
    pd.set_option('use_inf_as_na',True)
    warnings.filterwarnings('ignore')

    import joblib
    from datetime import datetime

    seed = 42
    df = full_df
    col = list(df.columns)
    if 'DEPT' in col:
        df['DEPTH']=df['DEPT'].copy()
        df = df.drop(['DEPT'], axis=1)
        
    well_list = list(df['WELL'].unique())
    if len(well_list) !=1:
        well_list.append('all data')
    else:
        well_list = well_list
        
    well_list_in_json = json.dumps(well_list)
    
    return well_list_in_json