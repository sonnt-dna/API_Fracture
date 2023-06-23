# Regression

This folder contains support modules for Regression task.

To train a single algorithm:

```
from Regression.regressoion import Regressor
Model, train_score, validation_score = Regressor(
    features=X,
    target=y,
    validation_size=0.2,
    scoring="MAE",
    algorithm="XGBoost",
)
```
To train a single algorithm with optuna:

```
from Regression.regressoionoptuna import RegressorTuna
Model, train_score, validation_score = RegressorTuna(
    features=X,
    target=y,
    validation_size=0.2,
    scoring="MAE",
    algorithm="XGBoost",
)
```

To find a best algorithm:

```
from Regression.regressionfinder import RegressorFinder
Model, train_score, validation_score, scores = RegressorFinder(
    features=X,
    target=y,
    validation_size=0.2,
    scoring='MAE',
    base_score=100,
    max_train_valid_drop=1.,
)
```
To find a best algorithm using optuna package:

```
from Regression.regressionfinderoptuna import RegressorFinderTuna
Model, train_score, validation_score, scores = RegressorFinderTuna(
    features=X,
    target=y,
    validation_size=0.2,
    scoring='MAE',
    base_score=100,
    max_train_valid_drop=1.,
)
```