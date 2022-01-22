import tensorflow as tf


def create_inputs(data_type_mapper):
    """Create model inputs
    Args:
        data_type_mapper (dict): Dictionary with feature as key and dtype as value
    Returns:
        (dict): Keras inputs for each feature
    """
    return {feature: tf.keras.Input(shape=(1,), name=feature, dtype=dtype)\
        for feature, dtype in data_type_mapper.items()}


if __name__ == '__main__':
    binary_features = ['sex', 'fbs', 'exang']
    numeric_features =  ['trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'cp', 'restecg', 'ca']
    categoric_features = ['thal']

    dtype_mapper = {
        'age': tf.float32,
        'sex': tf.float32,
        'cp': tf.float32,
        'trestbps': tf.float32,
        'chol': tf.float32,
        'fbs': tf.float32,
        'restecg': tf.float32,
        'thalach': tf.float32,
        'exang': tf.float32,
        'oldpeak': tf.float32,
        'slope': tf.float32,
        'ca': tf.float32,
        'thal': tf.string,
    }

    heart_dir = tf.keras.utils.get_file("heart.csv", origin="http://storage.googleapis.com/download.tensorflow.org/data/heart.csv")
     
    dataset = tf.data.experimental.make_csv_dataset(
      heart_dir,
      batch_size=64,
      label_name='target',
      num_epochs=10,
    )

    feature_layer_inputs = create_inputs(dtype_mapper)

    # Define functions for engineered features
    def square(x):
        return x ** 2

    def ratio(x):
        """compute the ratio between two numeric features"""
        return x[0] / x[1]

    def cross_feature(x):
        """compute the crossing of two features"""
        return tf.cast(x[0] * x[1], dtype = tf.float32)

    def age_and_gender(x):
        """check if age gt 50 and if gender is male"""
        return tf.cast(
            tf.math.logical_and(x[0] > 50, x[1] == 1), dtype = tf.float32
        )

    def is_fixed(x):
        """encode categoric feature if value is equal to fixed"""
        return tf.cast(x == 'fixed', dtype = tf.float32)

    def is_reversible(x):
        """encode categoric feature if value is equal to fixed"""
        return tf.cast(x == 'reversible', dtype = tf.float32)
    
    def is_normal(x):
        """encode categoric feature if value is equal to fixed"""
        return tf.cast(x == 'normal', dtype = tf.float32)

    # The features based on thal is similar to one-hot encoding. This is just for illustration
    # purposes
    is_fixed = tf.keras.layers.Lambda(is_fixed)(
       feature_layer_inputs['thal']
    )

    is_normal = tf.keras.layers.Lambda(is_normal)(
       feature_layer_inputs['thal']
    )

    is_reversible = tf.keras.layers.Lambda(is_reversible)(
       feature_layer_inputs['thal']
    )

    age_and_gender = tf.keras.layers.Lambda(age_and_gender)(
        (feature_layer_inputs['age'], feature_layer_inputs['sex'])
    )

    age = tf.keras.layers.Lambda(lambda x: tf.cast(x > 50, dtype = tf.float32))(
        feature_layer_inputs['age']
    )

    trest_chol_ratio = tf.keras.layers.Lambda(ratio, name='trest_chol_ratio')(
       (feature_layer_inputs['trestbps'], feature_layer_inputs['chol'])
    )

    trest_cross_thalach = tf.keras.layers.Lambda(cross_feature)(
       (feature_layer_inputs['trestbps'], feature_layer_inputs['thalach'])
    )

    # concat all newly created features into one layer
    lambda_feature_layer = tf.keras.layers.concatenate([is_fixed, is_normal, is_reversible, 
                                                 age, age_and_gender, trest_chol_ratio, trest_cross_thalach
                                                 ]
    )

    numeric_feature_layer = tf.keras.layers.concatenate(
        [feature_layer_inputs[feature] for feature in numeric_features]
    )

    binary_feature_layer = tf.keras.layers.concatenate(
        [feature_layer_inputs[feature] for feature in binary_features]
    )

    # Add the rest of features
    feature_layer = tf.keras.layers.concatenate([lambda_feature_layer, numeric_feature_layer, binary_feature_layer])

    # setup model, this is basically Logistic regression
    x = tf.keras.layers.BatchNormalization()(feature_layer)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=feature_layer_inputs, outputs=output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(dataset, epochs=10)

    # save model
    tf.keras.models.save_model(model, "lambda_layered_model")

    # load model for inference
    loaded_model = tf.keras.models.load_model("lambda_layered_model")

    loaded_model.evaluate(dataset)
