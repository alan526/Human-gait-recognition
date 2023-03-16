from sklearn import preprocessing
import numpy as np
import re

def window_data(data, windowsize, label):
    x = []
    y = []
    i = 0
    window_disp = 4
    while (i + windowsize) <= len(data) - 1:
        x.append(data[i:i + windowsize])
        y.append(label[i + windowsize])
        i += window_disp
    return x, y

def read_data(fname):
    frame_num = 60
    dataset = np.zeros((frame_num, 8, 16))
    label = np.zeros(frame_num)
    # read file
    with open(fname + ".txt", mode="r", encoding="utf-8") as file:
        for k in range(frame_num):
            # print(k)
            clear_data = file.readline()
            label[k] = file.readline()
            for i in range(8):
                data = file.readline()
                dataset[k][i] = ([float(s) for s in re.findall(r'-?\d+\.?\d*', data)])
    # Min_Max_Scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # dataset = dataset.reshape(frame_num * 8, 16)
    # MinMax_Data = Min_Max_Scaler.fit_transform(dataset)
    # dataset = MinMax_Data.reshape(frame_num, 8, 16)
    for i in range(frame_num):
        for j in range(8):
            for k in range(16):
                if dataset[i][j][k] > 3000:
                    dataset[i][j][k] = 3000
    dataset = dataset/3000
    # sliding window
    window_size = 16
    (x_train, y_train) = window_data(dataset, window_size, label)
    return (x_train, y_train)