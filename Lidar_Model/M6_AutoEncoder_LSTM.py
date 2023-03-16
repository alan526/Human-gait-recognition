from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import TimeDistributed, LSTM, UpSampling2D, BatchNormalization, Input
from tensorflow.keras.models import Model, load_model
import tensorflow
import numpy as np
from data_generate_4dim import data_generate
from sklearn.model_selection import train_test_split
from print_outcome import outcome
from sklearn import preprocessing
import time

num_class = 5
window_size = 16
autoencoder = load_model(".\encode_decoder_3.h5")
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
    return conv3


# spilt the data
(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)


# lstm model
def lstm(enco):
    flat = TimeDistributed(Flatten())(enco)
    lstm1 = LSTM(64, return_sequences=True)(flat)
    lstm1 = Dropout(0.1)(lstm1)
    lstm2 = LSTM(32, return_sequences=False)(lstm1)
    lstm2 = Dropout(0.1)(lstm2)
    out = Dense(num_class, activation="softmax")(lstm2)
    return (out)


# connect encoder lstm model
encode = encoder(input_img)
full_model = Model(input_img, lstm(encode))
# pass params from encoder decoder model to encoder lstm model
encoder_layer_num = 15
for l1, l2 in zip(full_model.layers[:encoder_layer_num], autoencoder.layers[0:encoder_layer_num]):
    l1.set_weights(l2.get_weights())
# set encoder layers don't train
for layer in full_model.layers[0:encoder_layer_num]:
    layer.trainable = False
opt = tensorflow.keras.optimizers.Adam()
full_model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt,
                   metrics=['accuracy'])
full_model.summary()

result = full_model.fit(x_train, y_train, batch_size=128, epochs=200, verbose=1,
                        validation_split=0.1)

# model predict

t_start = time.time()
predict = full_model.predict(x_test)
t_end = time.time()
eval = full_model.evaluate(x_test, y_test)
ct = outcome(result, 'AutoEncoder_LSTM', predict, y_test, eval)
print('predict time => ', (t_end - t_start))
# full_model.save("./ae_lstm.h5")
