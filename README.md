# Keras Examples

This project contains Keras examples. It will be updated as new examples are added

Below find a list of examples currently available with a short description:
* lambda_layer_engineering/lambda_layer_engineering.ipynb:
    
    Often for structured data problems we end up using multiple libraries for preprocessing or feature engineering. We can go as far as having a full ML 
    training pipeline using different libraries for example Pandas for reading data and also feature engineeering, sklearn for encoding features for example
    OneHot encoding and Normalization. The estimator might be an sklearn classifier, xgboost or it can for example be a Keras model. In the latter case, 
    we would end up with artifacts for feature engineering and encoding and also different artifacts for the saved model. The pipeline is also disconnected 
    and an extra step is needed to feed encoded data to the Keras model. For this step the data can be mapped from a dataframe to something like tf.data.Datasets
    type or numpy array before feeding it to a Keras model.
