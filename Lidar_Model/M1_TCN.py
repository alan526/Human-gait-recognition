from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow
from tcn import TCN
import numpy as np
from sklearn.model_selection import train_test_split
from print_outcome import outcome
from data_generate_3dim import data_generate
import time
import pandas as pd

num_class = 5
# get all data
(x_train, y_train) = data_generate()
# shuffle
per = np.random.permutation(x_train.shape[0])  # 打亂後的行號
x_train = x_train[per, :, :]  # 獲取打亂後的訓練資料
y_train = y_train[per]
# spilt the data
(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
# TCN model
model = Sequential()
model.add(TCN(input_shape=(x_train.shape[1], x_train.shape[2]),
              dilations=[2 ** i for i in range(3)], kernel_size=2, nb_filters=64,
              return_sequences=True))
model.add(Dropout(0.1))
model.add(TCN(input_shape=(x_train.shape[1], x_train.shape[2]),
              dilations=[2 ** i for i in range(3)], kernel_size=2, nb_filters=32,
              return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(num_class, activation='softmax'))

opt = tensorflow.keras.optimizers.Adam()
model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
print(model.summary())
result = model.fit(x_train, y_train, epochs=200, batch_size=128, verbose=1,
                   validation_split=0.1)

t_start = time.time()
predict = model.predict(x_test)
t_end = time.time()
eval = model.evaluate(x_test, y_test)

ct = outcome(result, 'TCN', predict, y_test, eval)

print('predict time => ', (t_end - t_start))
# model.save("./tcn_2.h5")
