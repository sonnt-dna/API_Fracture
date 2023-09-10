import pandas as pd
def predict_fracture(full_df: pd.DataFrame,
                    feature_list: str,
                    scoring: float,
                    objective: str,
                    algorithm: str,
                    iteration: int,
                    target: str,
                    ):
    #import libraries
    import sys
    sys.path.append('../')
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import warnings
    import catboost as cat

    print(sys.path)
    """" NOTE: Import packages which customized by VPI (therefore can't be installed with "pip") """
    from app.service.library.Devtools.LightGBM._ligthgbmR import Train_LGBM
    from app.service.library.Devtools.LightGBM.score_cal import RScore
    from app.service.library.Devtools.XGBoost._xgboostR import Train_XGBR
    from app.service.library.Devtools.XGBoost.score_cal import RScore
    from app.service.library.Devtools.CatBoost._catboostR import Train_CATR
    
    pd.set_option('display.max_columns', 100)
    pd.set_option('use_inf_as_na',True)
    warnings.filterwarnings('ignore')

    import joblib
    from datetime import datetime

    seed = 42
    df = full_df
    # print(df)
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

    # ## 1. Exploratory Data Analysis

    # In[4]:
    well_view = 'all data'
    # ### 1.1. Curve missing percentage
    if well_view=='all data':
        data_view=df
    else:
        data_view = df.loc[df['WELL'].astype(str) == well_view]
        #replace -999 in dataframe
    def replace_999(df,col):
        df[col]=df[col].replace(-999, np.nan)
        return df

    col = [col for col in data_view.columns if col not in ['WELL', 'DEPTH']]
    replace_999(data_view, col)


    # ## 2. Missing log model

    # You can choose wells from list below. In case choosing all well, please set well =['all data']


    well = ['01-97-HXS-1X','15-1-SN-1X', '15-1-SN-2X','15-1-SN-4X', '15-1-SNN-1P', '15-1-SNN-2P','15-1-SNN-3P','15-1-SNN-4P','15-1-SNS-7P','15-1-SNS-4P','15-1-SNS-2P']


    # ### 2.1. Preprocessing

    # If you choose single well, you can choose depth interval for training. Otherwise, please type 'none'.
    data = df
    if len(well)!=1:
        print("Please type 'none' in from_training and to_training")
    else:
        data = data.sort_values(by=['DEPTH'])
        print('Min dept:',data['DEPTH'].min())
        print('Max dept:', data['DEPTH'].max())

    target_use = target
    scoring_use = scoring
    objective = objective
    algorithm= algorithm #algorithm #'xgboost' #'catboost', #'xgboost' #'lightgbm', 'catboost'
    iteration_use = iteration
    print('Target:', target_use)
    print("interation:", iteration_use)

    good_data = 'True'
    upper_interval = '2.5'
    lower_interval = '1.5'
    from_training = 'none'
    to_training = 'none'

    if from_training == 'none':
        data=data
    else:
        data= data.loc[(data['DEPTH'] <= float(to_training))&(data['DEPTH'] >= float(from_training))]
    #replace -999 in dataframe
    def replace_999(df,col):
        df[col]=df[col].replace(-999, np.nan)
        return df
    #replace negative in columns
    def repl_negative(df,col):
        df[col]= np.where(df[col] <0,np.nan, df[col])
        return df

    col = [col for col in data.columns if col not in ['WELL', 'DEPTH']]
    replace_999(data, col)
    if 'BS' in col:
        data['DCALI_FINAL'] = np.where(data['CALI'].isnull(), np.nan, (data['CALI']-data['BS']))
    else:
        data['DCALI_FINAL'] = data['DCALI_FINAL']

    check_negative = ['RHOB', 'LLD', 'LLS', 'DTC', 'DTS']

    repl_negative(data, check_negative)

    if good_data == 'True':
        data = data.loc[(data['DCALI_FINAL'] <= float(upper_interval))&(data['DCALI_FINAL'] >= float(lower_interval))]
    else:
        data=data

    feature_list_raw = [col for col in data.columns if col not in [target, 'WELL']]

    print('Done processing!')
    print('You can choose features in this list:',feature_list_raw)
    feature = feature_list
    print('You can choose features using for Building model in this list:',feature)

    # ### 2.2 Model building



    # feature = [feature] 
    drop = feature.copy()

    print ('target features:',target_use)
    drop.append(target_use)

    print('Drop features:', drop)
    print ("Dữ liệu đầu vào")
    
    data = data.dropna(how ='any', subset=drop)
    print (data)
    if objective == 'valid_score':
        objective = 0
    else:
        objective = 1
    if target in check_negative:
        if algorithm == 'xgboost':
            task = 'reg:gamma'
        elif algorithm == 'lightgbm':
            task = 'gamma'
        else:
            task ='MAE'
    else:
        if algorithm == 'xgboost':
            task = 'reg:squarederror'
        elif algorithm == 'lightgbm':
            task = 'regression'
        else:
            task = 'RMSE'

    
    
    y=data[target_use]
    X=data[feature]
    print("Data after preprocessing")
    print(X)
    #print(X.isna().sum())
    #split data into sets
    X_use, X_test, y_use, y_test = train_test_split(X, y, train_size=0.9, random_state=seed, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_use, y_use, train_size=0.8, random_state=seed, shuffle=True)
    preprocessors = Pipeline(steps=
            [
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("scaling", MinMaxScaler())
        ]
    )

    X_train, X_valid = preprocessors.fit_transform(X_train), preprocessors.transform(X_valid)
    X_test = preprocessors.transform(X_test)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    X_train = pd.DataFrame(X_train, columns=feature)
    X_valid = pd.DataFrame(X_valid, columns=feature)
    X_test  = pd.DataFrame(X_test, columns=feature)

    if algorithm=='lightgbm':
        model = Train_LGBM(
            features = X_train,
            target = y_train,
            iterations = int(iteration_use),
            scoring = scoring_use,
            validation_size = 0.1,
            task = task,
            )
        y_pred = model.predict(X_test)
        score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring_use)
        print(score)
    elif algorithm =='xgboost':
        model = Train_XGBR(
            features = X_train,
            target = y_train,
            iterations = int(iteration_use),
            scoring = scoring_use,
            validation_size = 0.1,
            task = task,
            )
        y_pred=model.predict(xgb.DMatrix(data=X_test, label=y_test))
        score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring_use)
        score_Train = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring_use)
        print(f"Test score: {score}")
    else:
        model= Train_CATR(
            features=X_train,
            target=y_train,
            iterations = int(iteration_use),
            scoring=scoring_use,
            validation_size = 0.1,
            task=task,
            )
        y_pred=model.predict(cat.Pool(data=X_test, label=y_test))
        score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring_use)
        score_Train = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring_use)
        print(f"Test score: {score}")


    if algorithm=='lightgbm':
        full_predict = model.predict(data=X)
    elif algorithm =='xgboost':
        full_predict = model.predict(xgb.DMatrix(data=X, label=y))
    else:
        full_predict = model.predict(cat.Pool(data=X, label=y))

    # result_in_json_1 = json_str.to_json(orient='index')
#API return for output

    result_in_json_full = data
    result_in_json_full["PREDICTED"] = full_predict

    import pyodbc
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.sql import text
    import sqlalchemy as sa
    import sys
    import re
    import os
    import numpy as np
    import pandas as pd

    driver = 'ODBC Driver 18 for SQL Server'  

    server = "xznozrobo3funm76yoyaoh75wm-qhke725ydcietlngqiyrfxa75u.datawarehouse.pbidedicated.windows.net"
    database = 'testpushdata'
    username = 'api@oilgas.ai'
    password = 'Xuw72090'
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}&TrustServerCertificate=no&Authentication=ActiveDirectoryPassword'
    engine = create_engine(connection_string, echo=True, connect_args={'auto_commit': True}, fast_executemany=True)

    df = pd.DataFrame(result_in_json_full)

    df.to_sql(name='Prediction_results', con=engine, if_exists='replace', dtype={
        'BASEMENT': sa.Integer(),
        'CALI': sa.Integer(),
        'DCALI_FINAL': sa.Float(),
        'DEPTH': sa.Float(),
        'DTC': sa.Float(),
        'DTS': sa.Float(),
        'GR': sa.Float(),
        'LLD': sa.Float(),
        'LLS': sa.Float(),
        'NPHI': sa.Float(),
        'RHOB': sa.Float(),
        'SP': sa.Float(),
        'WELL': sa.String(50),
        'PREDICTED': sa.Float(),
        'ROP': sa.Float(),
        'RPM': sa.Float(),
        'TORQUE': sa.Float(),
        'WOB': sa.Float(),
        'FLWPMPS': sa.Float()



    }, index = False, chunksize=200, method='multi')

    import neptune.new as neptune
    from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary

    run = neptune.init_run(
    project = "ProductionManagement/DNA-Missing-Log-Tool",
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjgwMDYzYi0zZWI4LTQ4Y2YtODEzYS05ODQ0MzlhN2FiOTIifQ==",
    name = "training",
    tags = ["FractureGoal2", "train", "balanced splitting data by Well"
            # "Dropna all feature"
            # "rel_depth_Top_Weathering"
            ],
)
# create callbacks
    _callbacks = NeptuneCallback(run=run)
    run['input'] = full_df
    run["chosen well"] = well
    run["target"] = target

    run["feature"] = feature
    run["algorithm"]= algorithm
    params = {
        "scoring": scoring,
        "ojective": objective,
        "task": task,
    }
    run["parameters"] = params
    run["test_score"] = score
    if algorithm=='lightgbm':
        import json
        import tempfile
        model_json = model.dump_model()
        # Convert the model output to a JSON string
        model_json_str = json.dumps(model_json)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(model_json_str.encode('utf-8'))
            temp_name = tmp.name
        run["model"].upload(temp_name)
    elif algorithm =='xgboost':
        import json
        import tempfile
        model_json = model.get_dump(dump_format='json')
        # Convert the model output to a JSON string
        model_json_str = json.dumps(model_json)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(model_json_str.encode('utf-8'))
            temp_name = tmp.name
        run["model"].upload(temp_name)
    elif algorithm == 'catboost':
        import json
        import tempfile
        model_json = model.dump_model()
        # Convert the model output to a JSON string
        model_json_str = json.dumps(model_json)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(model_json_str.encode('utf-8'))
            temp_name = tmp.name
        run["model"].upload(temp_name)
    run.stop()

    import json
    arr = np.array(full_predict)
    serializable_list = arr.tolist()

    json_str = json.dumps({"array": serializable_list})
    result_in_json = json.dumps(json_str)
    return result_in_json
