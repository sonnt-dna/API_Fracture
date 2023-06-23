import pandas as pd
import numpy as np

#Combine 2 models 1&2 with weight alpha for 1st model and threshold to classify result
def Combine_2Models(model_1=None, model_2=None, features:pd.DataFrame=None,
                       threshold:float=None, alpha:float=None):

    #input validation
    assert len(features) != 0, 'Invalid dataframe'

    # Make predictions
    prediction_model1 = model_1.predict_proba(features)[:, 1]
    prediction_model2 = model_2.predict_proba(features)[:, 1]

    #Combine 2 predictions
    beta = 1 - alpha

    if threshold:
        return np.asarray([ 1 if i*alpha + j* beta >= threshold else 0 for i, j in zip(prediction_model1, prediction_model2)])
            #(pd.DataFrame(np.hstack((prediction_model1.reshape(-1,1), prediction_model2.reshape(-1,1))))
            #    .apply(lambda x: 1 if x[0]*alpha + x[1]* beta>=threshold else 0, axis=1))
    else:
        return (pd.DataFrame(np.hstack((prediction_model1.reshape(-1,1), prediction_model2.reshape(-1,1))))
                .apply(lambda x: x[0]*alpha + x[1]* beta, axis=1))
