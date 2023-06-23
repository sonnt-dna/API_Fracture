from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, precision_score, roc_auc_score
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score

# set seed
seed = 42
score_dict = {
    "balanced_accuracy": balanced_accuracy_score,
    "recall": recall_score,
    "f1_weighted": f1_score,
    "precision": precision_score,
    "roc_auc": roc_auc_score,
}


def Get_ModelTuna():
    # define model list
    models = {}
    models["Logistic"] = LogisticRegression(solver='saga', tol=1e-2, class_weight="balanced")
    #models["Support Vector Machine"] = SVC(tol=1e-2, probability=True, class_weight="balanced", max_iter=-1)
    models["ExtraTree"] = ExtraTreesClassifier(random_state=seed)
    models["RandomForest"] = RandomForestClassifier(class_weight="balanced", random_state=seed)
    models["GradientBoost"] = GradientBoostingClassifier(random_state=seed)
    models["Voting"] = VotingClassifier(estimators=[("ExtraTree", ExtraTreesClassifier(random_state=seed)),
                                                    ("RandomForest", RandomForestClassifier(random_state=seed)),
                                                    ("GradientBoost", GradientBoostingClassifier(random_state=seed))],
                                        voting="soft")
    models["Stacking"] = StackingClassifier(estimators=[("gbt", GradientBoostingClassifier(random_state=seed)),
                                                        ("rf", RandomForestClassifier(random_state=seed))],
                                            final_estimator=ExtraTreesClassifier(random_state=seed))
    models["HistGradientBoost"] = HistGradientBoostingClassifier()
    models["XGBoost"] = xgb.XGBClassifier(objective="binary:logistic", random_state=seed, verbosity=0,
                                          eval_metric='logloss', use_label_encoder=False)
    models["LightGBM"] = lgb.LGBMClassifier(verbose=-1, metric=None, early_stopping_round=None, num_iterations=100)

    #models["blender"] = build_ensemble(scoring=scoring)

    # models["sequential"] = sequential_ensemble()

    models["BalancedRandomForest"] = BalancedRandomForestClassifier(class_weight="balanced",
                                                                    sampling_strategy='not majority', random_state=seed)

    # models["EasyEnsemble"] = EasyEnsembleClassifier(sampling_strategy='not majority', random_state=seed)

    models["BalancedBagging"] = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=10),
                                                          sampling_strategy='not majority', random_state=seed)

    models["CatBoost"] = CatBoostClassifier(random_state=seed, iterations=50, verbose=0)

    return models