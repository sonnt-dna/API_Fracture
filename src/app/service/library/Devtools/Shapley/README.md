# SHAP Values

This folder contains support modules for model explaination. 
- Cshapley_values: Explain classification model
- RShapley: Explain regression model

With Classification:

```
from Shapley.Cshapley_values import Shapley_importances
Shapley_importances(model, X, show_plot=True) # if you call from notebook
or
Shapley_importances(model, X, show_plot=False) # if you call from terminal
```

With Regression:

```
from Shapley.RShapley import Shapley_importances
Shapley_importances(model, X, show_plot=True) # if you call from notebook
or
Shapley_importances(model, X, show_plot=False) # if you call from terminal
```