import pandas as pd
import tensorflow as tf


def data_frame_to_dataset(data_frame_features=None, data_frame_labels=None):
    """Transform from data frame to data_set
    Args:
        data_frame_features (pandas.DataFrame): Features Data.
        data_frame_labels (pandas.DataFrame): Target Data
    """
    if data_frame_labels is not None:
        return tf.data.Dataset.from_tensor_slices((dict(data_frame_features), data_frame_labels.values))
    return tf.data.Dataset.from_tensor_slices(dict(data_frame_features))


def create_inputs(dataset=None, features=None):
    """Create model inputs
    Args:
        dataset (tf.data.Dataset): Features Data to apply encoder on.
        features (list): list of feature names
    Returns:
        (dict): Keras inputs for each feature
    """
    def get_data_type(ds, feature):
        """helper function to get dtype of feature"""
        if len(ds.element_spec) == 2:
            return ds._structure[0][feature].dtype
        return ds._structure[feature].dtype

    return {feature: tf.keras.Input(shape=(1,), name=feature, dtype=get_data_type(dataset, feature))\
        for feature in features}


if __name__ == '__main__':
    dataframe = pd.read_csv('heart.csv')
    labels = dataframe.pop("target")
    dataset = data_frame_to_dataset(dataframe, labels).batch(32)
    # features = dataframe.columns.tolist()
    # feature_layer_inputs = create_inputs(dataset, features)

    # dataset1 = tf.data.experimental.CsvDataset(
    #     'heart.csv',
    #     #batch_size=32,
    #     label_name='target',
    #     #num_epochs=1,
    #     #ignore_errors=True
    # )

    features = [
       'age',
       'sex',
       'cp',
       'trestbps',
       'chol',
       'fbs',
       'restecg',
       'thalach',
       'exang',
       'oldpeak',
       'slope',
       'ca',
       'thal'
    ]
    
    numeric_features =  ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']

    feature_layer_inputs = create_inputs(dataset, features)

    # Define functions for engineered features
    def ratio(x):
        """compute the ratio between two numeric features"""
        return x[0] / x[1]

    def cross_feature(x):
        """compute the crossing of two features"""
        return tf.cast(x[0] * x[1], dtype = tf.float32)

    def is_fixed(x):
        """encode categoric feature if value is equal to fixed"""
        return tf.cast(x == 'fixed', dtype = tf.float32)

    # Doing this way your feature engineering is part of your network architecture and can be 
    # persisted and loaded for inference as a standalone. Note the caveats of Lambda layers when
    # porting saved model to a different environment. Lambda layers persist as python bytecode
    # Apply functions by creating Lambda layers
    trest_chol_ratio = tf.keras.layers.Lambda(ratio, name='trest_chol_ratio')(
       [feature_layer_inputs['trestbps'], feature_layer_inputs['chol']]
    )

    sex_cp_cross = tf.keras.layers.Lambda(cross_feature)(
       [feature_layer_inputs['sex'], feature_layer_inputs['cp']]
    )

    is_fixed = tf.keras.layers.Lambda(is_fixed)(
       feature_layer_inputs['thal']
    )

    # concat all newly created features into one layer
    feature_layer = tf.keras.layers.concatenate([trest_chol_ratio, sex_cp_cross, is_fixed])

    # Add the rest of features
    oldpeak_layer = feature_layer_inputs['oldpeak']
    feature_layer = tf.keras.layers.concatenate([feature_layer, oldpeak_layer])
    dense_layer = tf.keras.layers.Dense(32)(feature_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(feature_layer)
    model = tf.keras.Model(inputs=feature_layer_inputs, outputs=output_layer)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(dataset, epochs=10)

    # save model
    tf.keras.models.save_model(model, "lambda_layered_model")
    # load model for inference
    loaded_model = tf.keras.models.load_model("lambda_layered_model")

    # add a tensorflow serving component?