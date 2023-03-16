from data_process_3dim import read_data
import numpy as np
from sklearn import preprocessing

def data_connect(fname, start, end, x, y):
    for i in range(start, end):
        # print(i)
        (temp_x, temp_y) = read_data(fname + str(i))
        a = np.array(temp_x)
        b = np.array(temp_y)
        x = np.vstack((x, a))
        y = np.hstack((y, b))
    return (x, y)

def data_generate():

    (x_train, y_train) = read_data("./8-18walk/1")
    (x_train, y_train) = data_connect("./8-24walk/", 2, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-11walk/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5walk/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./8-24fall/", 2, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-11fall/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5fall/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5stand/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-11stand/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./8-24stand/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-11runjump/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-10runjump/", 1, 41, x_train, y_train)
    (x_train, y_train) = data_connect("./9-11nobody/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-5wall/", 1, 36, x_train, y_train)
    (x_train, y_train) = data_connect("./9-30nobody/", 1, 26, x_train, y_train)
    (x_train, y_train) = data_connect("./9-30stand/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-30fall/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-30walk/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./9-30runjump/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./10-05nobody/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./10-05stand/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./10-05fall/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./10-05walk/", 1, 21, x_train, y_train)
    (x_train, y_train) = data_connect("./10-05runjump/", 1, 21, x_train, y_train)

    # Min_Max_Scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # data_num = x_train.shape[0]
    # x_train = x_train.reshape(data_num * window_size, 128)
    # print(x_train.shape)
    # x_train = Min_Max_Scaler.fit_transform(x_train)
    # x_train = x_train.reshape(data_num, window_size, 128)
    return (x_train, y_train)
