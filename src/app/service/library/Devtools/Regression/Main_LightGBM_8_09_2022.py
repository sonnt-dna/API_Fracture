import numpy as np
import pandas as pd
import lightgbm as lgb
from Devtools.LightGBM.score_cal import Score

pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings('ignore')

seed = 42
data_path = '../data/220718-newdata.csv'
data = pd.read_csv(data_path)
features = ["DEPT", "DXC", "SPP", "TORQUE", "FLWPMPS", "ROP", "RPM", "TGAS", "WOB"]
cols = list(data.columns)
cols = cols + ['spp_', 'dept_']
dataset = pd.DataFrame(data=None, columns=cols)
for i, well in enumerate(data.WELL.unique()):
    data_ = data[data.WELL==well].sort_values(by='DEPT').reset_index(drop=True)
    data_.loc[:,'spp_'] = data_.loc[:, "SPP"] - data_.loc[0, "SPP"]
    data_.loc[:, 'dept_'] = data_.loc[:, 'DEPT'] - data_.loc[0, "DEPT"]
    dataset = pd.concat([dataset, data_])
label = 'FRACTURE_ZONE'
dataset = dataset.dropna(subset=[label])
dataset['torque_per_rpm'] = dataset.loc[:, "TORQUE"] / dataset.loc[:, "RPM"]
dataset['rop_per_rpm'] = dataset.loc[:, "ROP"] / dataset.loc[:, "RPM"]
dataset['tgas_per_flwpmps'] = dataset.loc[:, "TGAS"] / dataset.loc[:, "FLWPMPS"]
#datase["spp_per_depth"] = data.SPP.values / data.DEPT.values
features = features + ["spp_", "dept_", "tgas_per_flwpmps", "torque_per_rpm", "rop_per_rpm"]
# features.remove("DEPT")
# features.remove("SPP")
# features.remove("TGAS")
# features.remove("RPM")
dataset = dataset.drop(dataset[dataset.RPM==0].index)
X = dataset[features]
#X = np.log1p(data[features])
#X = pd.DataFrame(X, columns=features)
labels = dataset[label]
# print(dataset.head())
# print(dataset.info())
# # processing pipeline
# preprocessors = make_pipeline(IterativeImputer(estimator=ElasticNetCV(max_iter=int(1e6)), random_state=seed),
#                               MinMaxScaler(),
#                               )
# split data
#X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.9, random_state=seed, #shuffle=True, stratify=labels)
#X_train, X_test = preprocessors.fit_transform(X_train), preprocessors.transform(X_test)
# set scoring
scoring='f1_weighted'
#model = Train_LGBMC(
#    features=X_train,
#    labels=y_train,
#    iterations=100,
#    scoring=scoring,
#    base_score=0.83, # applied when objective is 1 (train_test_drop)
#    validation_size=0.1,
#    objectives=0, #{0: "valid_score", 1: "train_test_drop"}
#    favor_class=1, #{0: "min false negative", 1: "min false positive", 2: "balanced"}
#    show_shap=True, # flag to show shap True or False
#   )

model = lgb.Booster(model_file='../saved_models/lgb_model.json')
# model = lgb.Booster(model_file='../saved_models/lgb_model_no_dept_v0.1.1.json')
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import time
for well in dataset.WELL.unique():
    test_ = dataset[dataset.WELL==well]
    X = test_[features]
    y = test_[label]
    y_pred=model.predict(X) >= 0.5
    test_['prediction'] = y_pred.astype(int)
    test_=test_.drop(columns=["spp_", "dept_", "tgas_per_flwpmps", "torque_per_rpm", "rop_per_rpm"])
    test_.to_csv(f'../Study_results/{time.time()}prediction_{well}_no_dept.csv')
    score = Score(y_true=y, y_pred=y_pred, scoring=scoring, favor_class=1)
    print(f"Score of {well}: {score}")

    # y_pred_full = model.predict(X) >=0.5
    conf_ = confusion_matrix(y, y_pred)
    confd_ = ConfusionMatrixDisplay(conf_)
    confd_.plot()
    plt.title(f'Confusion matrix of {well}')
    plt.savefig(f'../imgs/confusion-matrix-wells/{time.time()}_well_cm_no_dept.png')
    plt.show();