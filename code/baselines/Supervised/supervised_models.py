import tensorflow as tf

def run_supervised_base_model(input_shape, output_shape, dense_layers=2, model_name="supervised_base_model"):
    """
    Create the supervised base model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.

    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
        -> Dense: 128 hidden size, relu
        -> Dropout: 40%
        -> Dense: (output_shape) hidden size, relu
        -> Softmax

    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)

    Returns:
        model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Conv1D(
        #64, 12,
        32,24,
        activation='relu', padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv1D(
        #96, 8,
        64, 16,
        activation='relu', padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv1D(
        #144, 4,
        96, 8,
        activation='relu', padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last',
            name='global_max_pooling1d')(x)

    if dense_layers==2:
        # first dense layer
        x = tf.keras.layers.Dense(128, kernel_initializer=tf.random_normal_initializer(stddev=.01), activation='relu')(x)
        #x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5, name='drop_1')(x)
    # last layer
    # x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    lr = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=1000)
    #lr = 0.03
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model

