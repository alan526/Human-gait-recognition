# import
from sklearn import preprocessing
import numpy as np
import re

# sliding window
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
    # variable
    frame_num = 60
    dataset = np.zeros((8*frame_num, 16))
    label = np.zeros(frame_num)
    # read file
    with open(fname+".txt", mode="r", encoding="utf-8") as file:
        for k in range(frame_num):
            # print(k)
            clear_data = file.readline()
            label[k] = file.readline()
            for i in range(8):
                data = file.readline()
                dataset[8 * k + i] = ([float(s) for s in re.findall(r'-?\d+\.?\d*', data)])

    # min max normalize
    # Min_Max_Scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # MinMax_Data = Min_Max_Scaler.fit_transform(dataset)

    x = np.zeros((frame_num, 128))
    for i in range(frame_num):
        for j in range(8):
            for k in range(16):
                if dataset[8 * i + j][k]>3000:
                    dataset[8 * i + j][k] = 3000
                x[i][16 * j + k] = dataset[8 * i + j][k]
    x = x / 3000
    window_size = 16
    (x_train, y_train) = window_data(x, window_size, label)
    return(x_train, y_train)