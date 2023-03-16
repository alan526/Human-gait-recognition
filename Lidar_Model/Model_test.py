from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow
from tcn import TCN, tcn_full_summary
from data_process_3dim import read_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from print_outcome import outcome

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('TCN ' + train)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc=0)
    plt.show()

def data_connect(fname, start, end, x, y):
    for i in range(start, end):
        # print(i)
        (temp_x, temp_y) = read_data(fname + str(i))
        a = np.array(temp_x)
        b = np.array(temp_y)
        x = np.vstack((x, a))
        y = np.hstack((y, b))
    return (x, y)

def run_task():
    # get all data
    (x_train, y_train) = read_data("./8-18walk/1")
    (x_train, y_train) = data_connect("./8-18wall/", 1, 11, x_train, y_train)
    (x_train, y_train) = data_connect("./8-24walk/", 2, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./8-24fall/", 2, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./8-24stand/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5fall/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5stand/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5walk/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5wall/", 1, 36, x_train, y_train)
    (x_train, y_train) = data_connect("./9-10runjump/", 1, 41, x_train, y_train)
    # shuffle
    per = np.random.permutation(x_train.shape[0])  # 打亂後的行號
    x_train = x_train[per, :, :]  # 獲取打亂後的訓練資料
    y_train = y_train[per]
    # spilt the data
    (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.2)
    print(x_test.shape)
    print(y_test.shape)
    # TCN model
    model = Sequential()
    model.add(TCN(input_shape=(x_train.shape[1], x_train.shape[2]),
                  dilations=[2 ** i for i in range(3)], kernel_size=3, nb_filters=64,
                  return_sequences=True))
    model.add(Dropout(0.1))
    model.add(TCN(input_shape=(x_train.shape[1], x_train.shape[2]),
                  dilations=[2 ** i for i in range(3)], kernel_size=3, nb_filters=16,
                  return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(5, activation='softmax'))

    opt = tensorflow.keras.optimizers.Adam()
    model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    result = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
                       validation_split=0.1)

    predict = model.predict(x_test)
    outcome(result, 'TCN', predict, y_test)


if __name__ == '__main__':
    run_task()