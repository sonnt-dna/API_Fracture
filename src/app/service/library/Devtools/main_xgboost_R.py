from XGBoost._xgboostR import Train_XGBR
from XGBoost.score_cal import RScore
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings('ignore')

seed = 42
data_path = 'data/data_reg.csv'
data = pd.read_csv(data_path)
features = ["DCALI_FINAL", "DTC", "GR", "LLD", "LLS", "NPHI", "VP"]

data = data.dropna(subset=["RHOB"])

#X = pd.DataFrame(np.log1p(data[features]), columns=features)
target = data.loc[:,'RHOB']
data = data[features]

# split data
X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.9, random_state=seed, shuffle=True)

# set scoring
scoring='MAE'
model = Train_XGBR(features=X_train,
                   target=y_train,
                   iterations=100, # number of iterations
                   scoring=scoring,
                   validation_size=0.1,
                   objectives=0, #{0: "valid_score", 1: "train_test_drop"}
                   task='reg:tweedie',
                   show_shap=True, # flag to show shap True or False (or save)
                   )

y_pred=model.predict(xgb.DMatrix(data=X_test, label=y_test))
score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring)
print(f"Test score: {score}")
