from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Activation,Dropout
from keras.utils import np_utils
from sklearn.utils import class_weight
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA, KernelPCA
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn import preprocessing

import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path
from mostcommon import most_common_selection
from load_dataset_2 import ImportData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns

epoch_param = 1000
split_param = 0.2


input_file = "./train.h5"
output_file = "./"
test_file = "./test.h5"

'''
input_file = "../data/train.h5"
output_file = "./result"
model_save_path = "./result/best_model"
test_file = "../data/test.h5"
'''
# model_path = model_save_path + '.h5'

def keras_learning(x_train, x_test, y_train, y_test, input_layer_num, classweight, batch_param, n_nodes1, n_nodes2, n_nodes3, drop_out):
    model = Sequential()

    model.add(Dense(n_nodes1, input_shape=(input_layer_num,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(n_nodes2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(n_nodes3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opti = SGD(lr=0.001, momentum=0.9, decay=0.001/1000, nesterov=False)
    # opti = Adam(lr=0.001, decay=0.001/epoch_param)
    model.compile(optimizer=opti,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train,
                     batch_size=batch_param, epochs=epoch_param,
                     validation_data=(x_test,y_test), verbose=2,
                     class_weight = classweight, callbacks=[early_stopping])
    # callbacks=[early_stopping, cb_checkpoint]
    return model,history


def keras_predict(x_train, x_test, y_train, y_test, z_test ,model):
    #evaluate
    train_score = model.evaluate(x_train, y_train)
    test_score = model.evaluate(x_test, y_test)
    print("[INFO] Train Score:", train_score[0])
    print("[INFO] Train accuracy:", train_score[1])
    print("[INFO] Test Score:", test_score[0])
    print("[INFO] Test accuracy:", test_score[1])

    # predict
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    pred_z = model.predict(z_test)
    pred_train = np.argmax(pred_train,axis=1)
    pred_test = np.array(np.argmax(pred_test, axis=1))
    true_test = np.array(np.argmax(y_test, axis=1))
    pred_z = np.array(np.argmax(pred_z, axis=1))

    return pred_test, true_test,pred_z

def plot_history(history,output_file,index):
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
    name = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"]
    fig.savefig(os.path.join(output_file, name[index]))

# create confusion matrix
def print_cmx(y_pred, y_true, output_file):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels= labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cmx, annot=True, cmap='Blues',fmt='g')
    plt.savefig(os.path.join(output_file,"confusion_matrix.png"))

def create_data(result_data):
    data_mat = []
    for i, data in enumerate(result_data):
        buf = [45324+i, data]
        data_mat.append(buf)
    return data_mat

# Data writing, reading
def write_csv(path, result_data):
    with open(path, 'w') as file:
        header = ['Id','y']
        writer = csv.writer(file)
        writer.writerow(header)
        #writer.writerows(map(lambda x: [x], result_data))
        writer.writerows(create_data(result_data))
        file.close()

# test writeing
def write_text(path, result_data):
    with open(path, 'w')as file:
        file.write(result_data)

def main():

    # define instances from class
    Train_Data = ImportData(input_file)
    Test_Data = ImportData(test_file)
    # get and split data
    Train_Data.read_split_data(split_param, False)
    Test_Data.read_split_data(0.0, True)
    # get train and test data
    x_train, x_test, y_train, y_test = Train_Data.get_train_data()
    z_test = Test_Data.get_test_data()
    x_data, y_data = Train_Data.get_xandy()

    ## Standardization
    x_train_stand = preprocessing.scale(x_train)
    x_test_stand = preprocessing.scale(x_test)
    z_test_stand = preprocessing.scale(z_test)
    '''
    ## Kernel PCA
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    x_train_kpca = kpca.fit_transform(x_train)
    x_test_kpca = kpca.fit_transform(x_test)
    # X_back = kpca.inverse_transform(X_kpca)
    '''
    ## PCA
    # pca = PCA(n_components=0.999000)
    # x_train_pca = pca.fit_transform(x_train_stand)
    # x_test_pca = pca.fit_transform(x_test_stand)
    # z_test_pca = pca.fit_transform(z_test_stand)

    # get dimension of x
    input_layer_num = x_train_stand.shape[1]
    print("number of features: ", input_layer_num)
    # keras only accepts one-hot label
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # calculate class weight
    classweight = class_weight.compute_class_weight('balanced', np.unique(y_data), y_data)

    batch_param = [32, 32, 30]
    n_nodes1 = [600, 1500, 1500]
    n_nodes2 = [500, 1000, 600]
    n_nodes3 = [120, 500, 500]
    drop_out = [0.26, 0.4, 0.4]
    result = ["1.csv", "2.csv", "3.csv"]
    for i in range(3):
        # define and fit model
        model,history = keras_learning(x_train_stand, x_test_stand, y_train, y_test, input_layer_num,
                                       classweight, batch_param[i], n_nodes1[i], n_nodes2[i], n_nodes3[i], drop_out[i])

        # predict the test dataset
        pred_test, true_test, pred_z = keras_predict(x_train_stand, x_test_stand, y_train, y_test,z_test_stand, model)

        # write pred_z to csv file
        write_csv(os.path.join(output_file, result[i]), pred_z)
        print("batch param: ", batch_param[i])
        print("n_nodes1: ", n_nodes1[i])
        print("n_nodes2: ", n_nodes2[i])
        print("n_nodes3: ", n_nodes3[i])
        print("drop_out: ", drop_out[i])

    print('Generate final result based on the most common labels')
    most_common_selection(3)

if __name__ == '__main__':
    main()
