import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import numpy as np
from tensorflow.keras.utils import to_categorical

def show_train_history(train_history, train, validation, model_name):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(model_name + ':' + train)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc=0)
    plt.show()



def outcome(result, model_name, predict, y_test, eval):
    show_train_history(result, 'accuracy', 'val_accuracy', model_name)
    show_train_history(result, 'loss', 'val_loss', model_name)
    class_name = ['walk', 'nobody', 'fall', 'stand', 'runjump']
    y_test = to_categorical(y_test, 5)
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(predict, axis=1), target_names=class_name))
    ct = (pd.crosstab(np.argmax(y_test, axis=1), np.argmax(predict, axis=1), rownames=['label'],
                      colnames=['prediction'], normalize='index'))
    print(ct)
    matrix = sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predict, axis=1)), annot=True,
                         fmt='.0f',
                         cmap='BuPu', xticklabels=class_name, yticklabels=class_name)
    matrix.set_xlabel('Predicted Labels')
    matrix.set_ylabel('Actual Labels')
    matrix.set_title('Confusion Matrix')
    print("The predict accuracy => ", eval[1])
    matrix.xaxis.set_ticks_position('top')
    matrix.tick_params(axis='y', rotation=0)
    plt.show()
    return ct