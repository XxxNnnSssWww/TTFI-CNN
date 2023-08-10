from tensorflow import keras


def build_model(class_nums, high, wide):
    inputdata = keras.Input(shape=(high, wide, 1))

    final = keras.layers.Conv2D(32, (3, 3), padding="same",activation='relu')(inputdata)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(8, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    #final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Flatten()(final)
    #final = keras.layers.Reshape((4096, 1))(final)
    #final = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(final)
    # final = keras.layers.Dropout(0.5)(final)
    #final = keras.layers.Bidirectional(keras.layers.LSTM(32))(final)
    final = keras.layers.Dense(class_nums)(final)
    # final = keras.layers.BatchNormalization()(final)
    final = keras.layers.Softmax()(final)

    model = keras.Model(inputs=inputdata, outputs=final)
    optimizer = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':
    modelcnn = build_model(class_nums=2)