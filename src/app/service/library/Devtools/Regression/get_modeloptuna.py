# generalized linear model
from sklearn.linear_model import ElasticNetCV, BayesianRidge, PoissonRegressor
from sklearn.linear_model import GammaRegressor, TweedieRegressor
from sklearn.svm import SVR, LinearSVR
# ensemble
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from .blender import build_ensemble

seed = 42


def Get_ModelTuna():
    # define model list
    models = {}
    models["LinearRegression"] = ElasticNetCV(max_iter=10000, tol=1e-2, random_state=seed)

    models["BayesianRidge"] = BayesianRidge()

    models["Poisson"] = PoissonRegressor(max_iter=10000, tol=0.001)

    models["Gamma"] = GammaRegressor(max_iter=10000, tol=0.001)

    models["Tweedie"] = TweedieRegressor(max_iter=10000, tol=0.001)

    models["Support Vector Machine"] = SVR(max_iter=10000000, tol=0.01)

    models["ExtraTree"] = ExtraTreesRegressor(random_state=seed)

    models["RandomForest"] = RandomForestRegressor(random_state=seed)

    models["GradientBoost"] = GradientBoostingRegressor(random_state=seed)

    models["Voting"] = VotingRegressor(
        estimators=[
            ("ElasticNetCV", ElasticNetCV()),
            ("rf", RandomForestRegressor(random_state=seed)),
            ("GradientBoost", GradientBoostingRegressor(random_state=seed))
        ]
    )

    models["Stacking"] = StackingRegressor(
        estimators=[
            ("svr", LinearSVR(random_state=seed)),
            ("rf", RandomForestRegressor(random_state=seed))
        ],
        final_estimator=HistGradientBoostingRegressor(random_state=seed)
    )

    models["HistGradientBoost"] = HistGradientBoostingRegressor(random_state=seed)

    models["XGBoost"] = xgb.XGBRegressor(
        # objective="reg:squaredlogerror",
        random_state=seed,
        verbosity=0,
    )

    models["LightGBM"] = lgb.LGBMRegressor(verbose=-1, metric=None, early_stopping_round=None)

    models["blender"] = build_ensemble()

    # models["CatBoost"] = cat.CatBoostRegressor(random_state=seed)

    return models