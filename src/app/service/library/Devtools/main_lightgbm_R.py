from LightGBM._ligthgbmR import Train_LGBM
from LightGBM.score_cal import RScore
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
model = Train_LGBM(features=X_train,
               target=y_train,
               iterations=25, # number of iterations
               scoring=scoring,
               validation_size=0.1,
               objectives=0, #{0: "valid_score", 1: "train_test_drop"}
               show_shap=True, # flag to show shap True or False
               )

y_pred=model.predict(X_test)
score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring)
print(f"Test score: {score}")
