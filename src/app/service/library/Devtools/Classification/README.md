# Classification

This folder contains several support modules for classification task.

To train a single algorithm:

```
from Classification.binary_classificaiton import Binary_Classifier
Model, train_score, validation_score = Binary_Classifier(
    features=X,
    labels=labels,
    validation_size=0.2,
    scoring='f1_weighted',
    imbalanced=True,
)
```
To train a single algorithm with optuna:

```
from Classification.binary_optuna import Optuna_Classifier
Model, train_score, validation_score = Optuna_Classifier(
    features=X,
    labels=labels,
    validation_size=0.2,
    scoring='f1_weighted',
    imbalanced=True,
)
```

To find a best algorithm:

```
from Classification.best_classifier import ClassifierFinder
Model, train_score, validation_score, scores = ClassifierFinder(
    features=X,
    labels=labels,
    validation_size=0.2,
    scoring='f1_weighted',
    max_train_valid_drop=0.075,
    imbalanced=True,
)
```
To find a best algorithm with optuna:
```
from Classification.best_optuna import OptunaFinder
Model, train_score, validation_score, scores = OptunaFinder(
    features=X,
    labels=labels,
    validation_size=0.2,
    scoring='f1_weighted',
    max_train_valid_drop=0.075,
    imbalanced=True,
)
```