from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import TimeDistributed, LSTM
from tensorflow.keras.models import Sequential
import tensorflow
import numpy as np
from sklearn.model_selection import train_test_split
from print_outcome import outcome
from data_generate_4dim import data_generate
from sklearn import preprocessing
import time

num_class = 5
window_size = 16
# get all data
(x_train, y_train) = data_generate()
# shuffle
x_train = x_train.reshape(x_train.shape[0], window_size, 8, 16, 1)
per = np.random.permutation(x_train.shape[0])  # 打亂後的行號
x_train = x_train[per, :, :, :, :]  # 獲取打亂後的訓練資料
y_train = y_train[per]
# spilt the data
(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)
# cnn lstm model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(window_size, 8, 16, 1)))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 4), strides=(1, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 4), strides=(1, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(num_class, activation='softmax'))
opt = tensorflow.keras.optimizers.Adam()
model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
print(model.summary())

result = model.fit(x_train, y_train, epochs=200, batch_size=128, verbose=1,
                   validation_split=0.1)

# model predict

t_start = time.time()
predict = model.predict(x_test)
t_end = time.time()
eval = model.evaluate(x_test, y_test)
ct = outcome(result, 'CNN_LSTM', predict, y_test, eval)
print('predict time => ', (t_end - t_start))
