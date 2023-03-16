from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import TimeDistributed, LSTM, UpSampling2D, BatchNormalization, Input
from tensorflow.keras.models import Model
import tensorflow
import numpy as np
from data_generate_4dim import data_generate

def run_task():
    window_size = 16
    # get all data
    (x_train, y_train) = data_generate()
    # shuffle
    x_train = x_train.reshape(x_train.shape[0], window_size, 8, 16, 1)
    per = np.random.permutation(x_train.shape[0])  # 打亂後的行號
    x_train = x_train[per, :, :, :, :]  # 獲取打亂後的訓練資料
    y_train = y_train[per]
    # encoder decoder model
    input_img = Input(shape=(window_size, 8, 16, 1))

    def encoder(input_img):
        conv1 = TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"))(input_img)
        conv1 = TimeDistributed(BatchNormalization())(conv1)
        conv1 = TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"))(conv1)
        conv1 = TimeDistributed(BatchNormalization())(conv1)
        pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
        conv2 = TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same"))(pool1)
        conv2 = TimeDistributed(BatchNormalization())(conv2)
        conv2 = TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same"))(conv2)
        conv2 = TimeDistributed(BatchNormalization())(conv2)
        pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
        conv3 = TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"))(pool2)
        conv3 = TimeDistributed(BatchNormalization())(conv3)
        conv3 = TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"))(conv3)
        conv3 = TimeDistributed(BatchNormalization())(conv3)
        # conv4 = TimeDistributed(Conv2D(256, (3, 3), activation="relu", padding="same"))(conv3)
        # conv4 = TimeDistributed(BatchNormalization())(conv4)
        # conv4 = TimeDistributed(Conv2D(256, (3, 3), activation="relu", padding="same"))(conv4)
        # conv4 = TimeDistributed(BatchNormalization())(conv4)
        return conv3

    def decoder(conv4):
        conv5 = TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"))(conv4)
        conv5 = TimeDistributed(BatchNormalization())(conv5)
        conv5 = TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"))(conv5)
        conv5 = TimeDistributed(BatchNormalization())(conv5)
        conv6 = TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same"))(conv5)
        conv6 = TimeDistributed(BatchNormalization())(conv6)
        conv6 = TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same"))(conv6)
        conv6 = TimeDistributed(BatchNormalization())(conv6)
        up1 = TimeDistributed(UpSampling2D((2, 2)))(conv6)
        conv7 = TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"))(up1)
        conv7 = TimeDistributed(BatchNormalization())(conv7)
        conv7 = TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"))(conv7)
        conv7 = TimeDistributed(BatchNormalization())(conv7)
        up2 = TimeDistributed(UpSampling2D((2, 2)))(conv7)
        decoded = TimeDistributed(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))(up2)
        return decoded

    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss='mean_squared_error', optimizer=tensorflow.keras.optimizers.Adam())
    autoencoder.summary()
    autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=128, epochs=200, verbose=1,
                                        validation_split=0.1)
    autoencoder.save("./encode_decoder_3.h5")
if __name__ == '__main__':
    run_task()