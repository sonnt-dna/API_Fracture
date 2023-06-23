from LightGBM._lightgbmC import Train_LGBM
from LightGBM.score_cal import Score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings('ignore')

seed = 42
data_path = '../data/220718-newdata.csv'
data = pd.read_csv(data_path)

features = ["DEPT", "DXC", "TORQUE", "FLWPMPS", "ROP", "RPM"]

label = 'FRACTURE_ZONE'
data = data.dropna(subset=[label])
X = data[features]
#X = np.log1p(data[features])
#X = pd.DataFrame(X, columns=features)
labels = data[label]

# processing pipeline
preprocessors = make_pipeline(IterativeImputer(estimator=ElasticNetCV(max_iter=int(1e6)), random_state=seed),
                              MinMaxScaler(),
                              )
# split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.9, random_state=seed, shuffle=True, stratify=labels)
#X_train, X_test = preprocessors.fit_transform(X_train), preprocessors.transform(X_test)

# set scoring
scoring='precision'
model = Train_LGBM(features=X_train,
               labels=y_train,
               iterations=20,
               scoring=scoring,
               validation_size=0.1,
               objectives=0, #{0: "valid_score", 1: "train_test_drop"}
               favor_class=1, #{0: "min false negative", 1: "min false positive", 2: "balanced"}
               show_shap=True, # flag to show shap True or False
               )
y_pred=model.predict(X_test) >= 0.5
score = Score(y_true=y_test, y_pred=y_pred, scoring=scoring, favor_class=1)
print(f"Test score: {score}")

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
y_pred_full = model.predict(cat.Pool(data=X, label=labels)) >=0.5
conf_ = confusion_matrix(labels, y_pred_full)
confd_ = ConfusionMatrixDisplay(conf_)
confd_.plot()
plt.show()