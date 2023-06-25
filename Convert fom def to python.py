# def predict_fracture(full_df, param):
def predict_fracture(full_df, parameter):
    #import libraries
    import sys
    sys.path.append('/lakehouse/default/Files')
    sys.path.append('../')

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import warnings
"""" NOTE: Import packages which customized by VPI (therefore can't be installed with "pip") """
    from Devtools.LightGBM._ligthgbmR import Train_LGBM
    from Devtools.LightGBM.score_cal import RScore
    from Devtools.XGBoost._xgboostR import Train_XGBR
    from Devtools.XGBoost.score_cal import RScore

    pd.set_option('display.max_columns', 100)
    pd.set_option('use_inf_as_na',True)
    warnings.filterwarnings('ignore')

    import joblib
    from datetime import datetime
     seed = 42
    df = full_df
    print(df)
    col = list(df.columns)
    if 'DEPT' in col:
        df['DEPTH']=df['DEPT'].copy()
        df = df.drop(['DEPT'], axis=1)
    target_list = ['NPHI', 'RHOB', 'DTS', 'DTC']
    scoring_list = ['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'Poisson', 'Tweedie', 'MeAE', 'ExVS', 'MSLE', 'ME', 'Gamma', 'D2T', 'D2Pi', 'D2A']
    obj_list =['valid_score', 'train_valid_drop']
    algorithm_list = ["lightgbm", "xgboost", "catboost"]
    well_list = list(df['WELL'].unique())
    if len(well_list) !=1:
        well_list.append('all data')
    else:
        well_list = well_list
        
# Convert def to python and notebook

import inspect
import nbformat as nbf
import os

# Giả sử hàm predict_fracture đã tạo lập và được định nghĩa trước đó

# def predict_fracture(full_df, parameter): # Ví dụ Hàm sử dụng để đóng API

#Put the name of file
output_filename =  "predict_handlers"       
output_dir = '/lakehouse/default/Files/API_Fracture/src/app/service/sample_predict'

def save_function_to_file(function, filename):
    source_code = inspect.getsource(function)
    with open(filename, 'w') as f:
        f.write(source_code)

def save_function_to_notebook(function, filename):
    source_code = inspect.getsource(function)
    nb = nbf.v4.new_notebook()

    code_cell = nbf.v4.new_code_cell(source_code)
    nb.cells.append(code_cell)

    with open(filename, 'w') as f:
        nbf.write(nb, f)

# Đặt tên cho file Python
output_filename_python = f'{output_filename}.py'
output_path = os.path.join(output_dir, output_filename_python)

# Sử dụng hàm save_function_to_file()
save_function_to_file(predict_fracture, output_path)
print(f'File Python {output_filename_python} đã được tạo thành công.')
print ("------------------------------------------------------------")

# Đặt tên cho file python
output_filename_notebook = f'{output_filename}.ipynb'
output_path = os.path.join(output_dir, output_filename_notebook)

# Sử dụng hàm save_function_to_notebook()
save_function_to_notebook(predict_fracture, output_path)
print(f'File Notebook {output_filename_notebook} đã được tạo thành công.')
