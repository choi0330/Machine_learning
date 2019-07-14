from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Activation,Dropout
from keras.utils import np_utils
from sklearn.utils import class_weight
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from sklearn.decomposition import PCA, KernelPCA
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from keras.layers import BatchNormalization
from hyperas.distributions import choice,uniform

import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path
from load_dataset_2 import ImportData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns


def plot_history(history,output_file):
    # print(history.history.keys())

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

    axL.plot(history.history['acc'])
    axL.plot(history.history['val_acc'])
    axL.set_title('model accuracy')
    axL.set_xlabel('epoch')
    axL.set_ylabel('accuracy')
    axL.legend(['acc', 'val_acc'], loc='lower right')

    axR.plot(history.history['loss'])
    axR.plot(history.history['val_loss'])
    axR.set_title('model loss')
    axR.set_xlabel('epoch')
    axR.set_ylabel('loss')
    axR.legend(['loss', 'val_loss'], loc='upper right')

    fig.savefig(os.path.join(output_file,"loss_acc.png"))

# create confusion matrix
def print_cmx(y_pred, y_true, output_file):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels= labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cmx, annot=True, cmap='Blues',fmt='g')
    plt.savefig(os.path.join(output_file,"confusion_matrix.png"))

def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense({{choice([200,300,400, 500, 600])}}, input_shape=(120,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([200,300,400, 500, 600])}}))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))

    if {{choice(['three', 'four', 'five'])}} == 'three':
        pass
    elif {{choice(['three', 'four', 'five'])}} == 'four':
        model.add(Dense({{choice([200,300,400, 500, 600])}}))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))
    elif {{choice(['three', 'four', 'five'])}} == 'five':
        model.add(Dense({{choice([200,300,400, 500, 600])}}))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense({{choice([200,300,400, 500, 600])}}))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(5))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    opti = SGD(lr=0.001, momentum=0.9, decay=0.001/1000, nesterov=False)
    model.compile(optimizer=opti,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history=model.fit(x_train, y_train,
              batch_size={{choice([16, 32, 64, 128, 256])}},
              epochs={{choice([50, 100, 150, 200, 300, 400])}},
              verbose=2,
              validation_data=(x_test, y_test), class_weight=[1.99577279, 0.6409843, 1.06594544, 0.67703339, 1.909585])

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -val_acc, 'status': STATUS_OK, 'model': model}

def prepare_data():

    input_file = "../data/train.h5"
    output_file = "./result/main"
    test_file = "../data/test.h5"

    # define instances from class
    Train_Data = ImportData(input_file)
    Test_Data = ImportData(test_file)
    # get and split data
    Train_Data.read_split_data(0.1, False)
    Test_Data.read_split_data(0.0, True)
    # get train and test data
    x_train,x_test,y_train, y_test = Train_Data.get_train_data()
    z_test = Test_Data.get_test_data()
    x_data, y_data = Train_Data.get_xandy()
    # keras only accepts one-hot label
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return x_train,y_train,x_test,y_test


if __name__ == "__main__":

    best_run, best_model = optim.minimize(model=create_model,
                                            data=prepare_data,
                                            algo=tpe.suggest,
                                            max_evals=100,
                                            trials=Trials())
    print(best_model.summary())
    print(best_run)
    _, _, x_test, y_test = prepare_data()
    val_loss, val_acc = best_model.evaluate(x_test, y_test)
    print("val_loss: ", val_loss)
    print("val_acc: ", val_acc)
    # plot loss and accuracy
    plot_history(history,output_file)
    # create confusion matrix
    print_cmx(pred_test, true_test, output_file)


