from tensorflow.keras import layers, Model, optimizers


def build_model(num_class, length):

    inputdata_1dcnn = layers.Input(shape=(length, 1), name='1dcnn')
    final = layers.Conv1D(64, 3, padding="same", activation='relu', input_shape=(length, 1))(inputdata_1dcnn)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)
    final = layers.MaxPooling1D(pool_size=2, strides=2)(final)
    final = layers.Conv1D(32, 3, padding="same", activation='relu')(final)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)
    final = layers.MaxPooling1D(pool_size=2, strides=2)(final)
    final = layers.Conv1D(16, 3, padding="same", activation='relu')(final)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)
    final = layers.MaxPooling1D(pool_size=2, strides=2)(final)
    final = layers.Conv1D(32, 3, padding="same", activation='relu')(final)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)

    final = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(final)
    final = layers.Bidirectional(layers.LSTM(32))(final)
    final = layers.Flatten()(final)
    final = layers.Dense(num_class)(final)
    final = layers.Softmax()(final)

    model = Model(inputs=inputdata_1dcnn, outputs=final)
    optimizer = optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    modelcnn = build_model(2, 3000)