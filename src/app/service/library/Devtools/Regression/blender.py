from mlens.ensemble import BlendEnsemble
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV

# set seed
seed = 42

def build_ensemble(**kwargs):
    """Return an ensemble."""

    estimators = [RandomForestRegressor(random_state=seed),
                  ElasticNetCV()]

    ensemble = BlendEnsemble(**kwargs)
    ensemble.add(estimators)
    ensemble.add_meta(GradientBoostingRegressor(random_state=seed))

    return ensemble
