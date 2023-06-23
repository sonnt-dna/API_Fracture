import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# create dir to store images
cwd = os.getcwd()
if 'imgs' not in os.listdir():
    path = cwd + '/imgs'
    os.makedirs(path)
else:
    path = cwd + '/imgs'

# plot feature importance by shap
def shapley_importances(model:any=None, X:any=None, feature_names:list=None,
                        shap_sample_size:float=0.5, show_plot:bool=False):
    """
        This function use results to visualize features shap values importance for each model
        Inputs must be declared:
            - model: a model to compute shap
            - X: input array / dataframe
            - feature_names: default is None. When you fitted model with dataframe, you don't need to supply feature_names
            feature_names is required when you fitted model by numpy array.
            - shap_sample_size: the size of sample to compute shape valuese
            - show_plot: True or False, default is False (run script). If you run this function in Jupyter
            environment, use 'show_plot=True'
        Outputs:
            - Shap value and dependencies plot.

        Example code:
        >> from features_algos_evaluation import scikit_shapley_importances
        >> shapley_importances(model, X, shap_sample_size=0.1, show_plot=False)
        """

    # Check input
    if X.shape[0] < 10000 or shap_sample_size==1:
        x_shap = X.copy()
    else:
        _, x_shap = train_test_split(X, test_size=shap_sample_size, random_state=42)

    # get model parameters
    model_name = type(model).__name__

    if model_name=='Pipeline':
        model_name = type(model[-1]).__name__
        preprocess = model[:-1]
        x_shap = preprocess.transform(x_shap)
        if model_name =="VotingRegressor":
            model = model.estimator[-1]
        elif model_name=="StackingRegressor":
            model = model.final_estimator
        else:
            model = model[-1]
    else:
        model_name = model_name

    if feature_names is None:
        try:
            feature_names = model.feature_names_in_
        except:
            try:
                feature_names = model.feature_name()
            except:
                feature_names = model.feature_names

    # Calculate Shap value for model
    explainer = shap.Explainer(model, x_shap, feature_names=feature_names)
    shap_values = explainer(x_shap)
    plt.clf()
    shap.summary_plot(shap_values, plot_type='dot', plot_size=(12, 6),
                      feature_names=feature_names, show=show_plot,
                      title=f"{model_name}_summary features importance plot")
    if show_plot is False:
        f = plt.gcf()
        plt.title(f"Feature importances of {model_name} model")
        plt.savefig(f"imgs/{model_name}_summary_plot.png", bbox_inches='tight')
        plt.close()
    # # watterfall
    # shap.plots.waterfall(shap_values, show=show_plot)
    # f = plt.gcf()
    # plt.title(f"Water Fall Plot of {model_name} model")
    # plt.savefig(f"imgs/waterfall_{model_name}.png", bbox_inches='tight')
    # plt.close()
    # # Beeswarm
    # shap.plots.beeswarm(shap_values, show=show_plot)
    # f = plt.gcf()
    # plt.title(f"Beeswarm Plot of {model_name} model")
    # plt.savefig(f"imgs/Beeswarm_{model_name}.png", bbox_inches='tight')
    # plt.close()
    # # Heatmap
    # shap.plots.heatmap(shap_values)
    # f = plt.gcf()
    # plt.title(f"Heatmap Plot of {model_name} model")
    # plt.savefig(f"imgs/Heatmap_{model_name}.png", bbox_inches='tight')
    # plt.close()
    for feature in feature_names:
        shap.plots.scatter(shap_values[:, feature], color=shap_values, show=show_plot)
        if show_plot is False:
            f = plt.gcf()
            plt.title(f"{model_name}_{feature}_features dependency plot")
            plt.savefig(f"imgs/{model_name}_{feature}_features dependency plot.png", bbox_inches='tight')
            plt.close()
    # shap.summary_plot(shape)
    # else:
    #     # explain all the predictions in the test set
    #     explainer = fasttreeshap.TreeExplainer(model, algorithm="auto", n_jobs=-1)
    #     shap_values = explainer(x_shap)
    #     plt.clf()
    #     fasttreeshap.summary_plot(shap_values.values, feature_names=feature_names, show=show_plot,
    #                       title=f"{model_name}_summary features importance plot.png")
    #     f = plt.gcf()
    #     plt.title(f"Feature importances of {model_name} model")
    #     plt.savefig(f"imgs/summary_plot_{model_name}.png", bbox_inches='tight')
    #     plt.close()
    #     # #watterfall
    #     # fasttreeshap.plots.waterfall(shap_values, show=show_plot)
    #     # f = plt.gcf()
    #     # plt.title(f"Water Fall Plot of {model_name} model")
    #     # plt.savefig(f"imgs/waterfall_{model_name}.png", bbox_inches='tight')
    #     # plt.close()
    #     # # Beeswarm
    #     # fasttreeshap.plots.beeswarm(shap_values, show=show_plot)
    #     # f = plt.gcf()
    #     # plt.title(f"Beeswarm Plot of {model_name} model")
    #     # plt.savefig(f"imgs/Beeswarm_{model_name}.png", bbox_inches='tight')
    #     # plt.close()
    #     # # Heatmap
    #     # fasttreeshap.plots.heatmap(shap_values, show=show_plot)
    #     # f = plt.gcf()
    #     # plt.title(f"Heatmap Plot of {model_name} model")
    #     # plt.savefig(f"imgs/Heatmap_{model_name}.png", bbox_inches='tight')
    #     # plt.close()
    #     for feature in feature_names:
    #         fasttreeshap.dependence_plot(feature, shap_values.values, features=x_shap, show=show_plot,
    #                              feature_names=feature_names)
    #         f = plt.gcf()
    #         plt.title(f"{model_name}_{feature}_features dependency plot")
    #         plt.savefig(f"imgs/{model_name}_{feature}_features dependency plot.png", bbox_inches='tight')
    #         plt.close()
