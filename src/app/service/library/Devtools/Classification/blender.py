from mlens.ensemble import BlendEnsemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from mlens.config import get_backend
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

def build_ensemble(proba=True, scoring='balanced_accuracy', **kwargs):
    """Return an ensemble."""

    estimators = [RandomForestClassifier(random_state=seed),
                  KNeighborsClassifier(n_neighbors=10)]

    ensemble = BlendEnsemble(**kwargs, test_size=0.2, scorer=score_dict[scoring])
    ensemble.add(estimators, proba=proba)   # Specify 'proba' here
    ensemble.add_meta(HistGradientBoostingClassifier(random_state=seed))

    return ensemble
