# import libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer, enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from .score_cal import Score

def Training(
            data:any=None,
            labels:any=None,
            test_size:float=None,
            model:any=None,
            scoring:any=None,
            hypers:any=None,
            ):
    
    """
    scoring: "balanced_accuracy", "recall_score", "f1_weighted", "precision_score", "roc_auc"

    """
    # set seed
    seed=42
    def warn(*args, **kwargs):
        pass

    assert labels.isna().sum() == 0, "Labels contain nan value. Please apply dropna with subset is labels column to remove it before feeding to model"

    # split data
    x_train, x_test, y_train, y_test, = train_test_split(data,
                                                          labels.astype(int),
                                                          test_size=test_size,
                                                          random_state=seed,
                                                          stratify=labels.astype(int),
                                                         )

    # build model
    clf = Pipeline(
        steps=[
            ("imputer", IterativeImputer(estimator=ElasticNetCV(l1_ratio=0.55), random_state=seed, max_iter=10000)),
            ("scaler", MinMaxScaler()),
            ("estimator", model)
        ]
    )

    # define search
    if hypers:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        search = RandomizedSearchCV(estimator=clf,
                                    param_distributions=hypers,
                                    cv=cv, 
                                    n_iter=20,
                                    scoring=scoring,
                                    n_jobs=-1,
                                    random_state=seed
                                    )
            
        # fitting 
        search.fit(x_train, y_train)

        clf = search.best_estimator_

        train_score = Score(clf, x_train, y_train, scoring)
        test_score = Score(clf, x_test, y_test, scoring)

    else:
        clf.fit(x_train, y_train)
        train_score = Score(clf, x_train, y_train, scoring)
        test_score = Score(clf, x_test, y_test, scoring)

    return clf, train_score, test_score

if __name__=='__main__':
    Training()
