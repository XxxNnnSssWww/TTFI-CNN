from tensorflow.keras import layers, Model, optimizers, activations
import math

def eca_2d_block(input_feature, b=1, gamma=2):
    channel = 64
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)

    x = layers.Reshape((-1, 1))(avg_pool)
    x = layers.Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False, )(x)
    x = activations.sigmoid(x)
    x = layers.Reshape((1, 1, -1))(x)
    x = layers.Multiply()([x, input_feature])

    output = layers.GlobalMaxPooling2D()(x)
    return output

def eca_1d_block(input_feature, b=1, gamma=2):
    channel = 16
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    avg_pool = layers.GlobalAveragePooling1D()(input_feature)

    x = layers.Reshape((-1, 1))(avg_pool)
    x = layers.Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False, )(x)
    x = activations.sigmoid(x)
    x = layers.Reshape((1, -1))(x)
    x = layers.Multiply()([x, input_feature])

    output = layers.GlobalMaxPooling1D()(x)
    return output

def build_model(num_class, length, high, wide):

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
    final = layers.MaxPooling1D(pool_size=2, strides=2)(final)

    final = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(final)
    eca_input = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(final)

    #if data_format='channels_last',input :(batch_size, rows, cols, channels)。
    #if keepdims=True, out:(batch_size, 1, 1, channels)。

    final_1dcnn = eca_1d_block(eca_input)

    inputdata_2dcnn = layers.Input(shape=(high, wide, 1), name='2dcnn')
    final = layers.Conv2D(32, (3, 3), padding="same",activation='relu')(inputdata_2dcnn)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)
    final = layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    final = layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)
    final = layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    final = layers.Conv2D(8, (3, 3), padding="same",activation='relu')(final)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)
    final = layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    final = layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = layers.BatchNormalization()(final)
    final = layers.ReLU()(final)
    eca_input = layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final_2dcnn = eca_2d_block(eca_input)

    final = layers.concatenate([final_1dcnn, final_2dcnn])
    final = layers.Dense(num_class)(final)
    final = layers.Softmax()(final)

    model = Model(inputs=[inputdata_1dcnn, inputdata_2dcnn], outputs=final)

    optimizer = optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())
    return model


if __name__ == '__main__':
    modelcnn = build_model(2, 3000, 149, 39)

