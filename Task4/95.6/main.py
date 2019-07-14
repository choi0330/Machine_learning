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
from load_dataset_2 import ImportData, Setup_whole_training_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns

epoch_param = 1000
split_param = 0.1


input_file1 = "./train_labeled.h5"
input_file2 = "./train_unlabeled.h5"
output_file = "./"
test_file = "./test.h5"

'''
input_file1 = "../data/train_labeled.h5"
input_file2 = "../data/train_unlabeled.h5"
output_file = "./result/"
test_file = "../data/test.h5"
'''


def keras_learning(x_train, x_test, y_train, y_test, input_layer_num, output_layer_num,
                   classweight, batch_param, n_nodes1, n_nodes2, n_nodes3, drop_out, model_path):
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
    model.add(Dense(output_layer_num))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opti = SGD(lr=0.001, momentum=0.9, decay=0.0001/1000, nesterov=False)
    # opti = Adam(lr=0.001, decay=0.001/epoch_param)
    model.compile(optimizer=opti,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train,
                     batch_size=batch_param, epochs=epoch_param,
                     validation_data=(x_test,y_test), verbose=2,
                     class_weight = classweight, callbacks=[early_stopping, cb_checkpoint])
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
        buf = [30000+i, data]
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
    Train_Data = ImportData(input_file1)
    Train_Data2 = ImportData(input_file2)
    Test_Data = ImportData(test_file)

    # get and split data
    Train_Data.read_split_data(split_param, False)
    Train_Data2.read_split_data(0.0, True)
    Test_Data.read_split_data(0.0, True)

    # get train and test data
    x_train, x_test, y_train, y_test = Train_Data.get_train_data()
    x2_test = Train_Data2.get_test_data()
    z_test = Test_Data.get_test_data()
    x_data, y_data = Train_Data.get_xandy()

    ## Standardization
    x_train_stand = preprocessing.scale(x_train)
    x_test_stand = preprocessing.scale(x_test)
    x2_test_stand = preprocessing.scale(x2_test)
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

    # get dimension of x and y
    input_layer_num = x_train_stand.shape[1]
    output_layer_num = 10
    print("number of features: ", input_layer_num)
    # keras only accepts one-hot label
    y_train_categorical = np_utils.to_categorical(y_train)
    y_test_categorical = np_utils.to_categorical(y_test)

    # calculate class weight
    classweight = class_weight.compute_class_weight('balanced', np.unique(y_data), y_data)

    batch_param = [8, 8, 8, 16, 16, 16]
    n_nodes1 = [600, 300, 300, 600, 300, 300]
    n_nodes2 = [600, 300, 300, 600, 300, 300]
    n_nodes3 = [300, 100, 100, 300, 100, 100]
    drop_out = [0.3, 0.2, 0.2, 0.3, 0.2, 0.2]
    model_path = [output_file + "best_model_pseudo1.h5", output_file + "best_model_pseudo2.h5",
                  output_file + "best_model_pseudo3.h5", output_file + "best_model_pseudo4.h5",
                  output_file + "best_model_pseudo5.h5", output_file + "best_model_pseudo6.h5"]
    pseudo_label_path = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv"]

    ## pseudo labeling algorithm
    num_nn = len(batch_param)
    for i in range(num_nn):
        # define and fit model
        model, history = keras_learning(x_train_stand, x_test_stand, y_train_categorical, y_test_categorical, input_layer_num, output_layer_num,
                                       classweight, batch_param[i], n_nodes1[i], n_nodes2[i], n_nodes3[i], drop_out[i], model_path[i])

        # predict the test dataset
        best_model = load_model(model_path[i])
        pred_test, true_test, pred_y2 = keras_predict(x_train_stand, x_test_stand, y_train_categorical,
                                                      y_test_categorical, x2_test_stand, best_model)

        # write pseudo_label to csv file
        write_csv(os.path.join(output_file, pseudo_label_path[i]), pred_y2)
        print("batch param: ", batch_param[i])
        print("n_nodes1: ", n_nodes1[i])
        print("n_nodes2: ", n_nodes2[i])
        print("n_nodes3: ", n_nodes3[i])
        print("drop_out: ", drop_out[i])

    print('Generate final result based on the most common labels')
    pseudo_path = [output_file + "1.csv", output_file + "2.csv", output_file + "3.csv",
                   output_file + "4.csv", output_file + "5.csv", output_file + "6.csv"]
    y2_test = most_common_selection(num_nn, pseudo_path)

    # Make a whole test set
    x_labeled = np.concatenate((x_train_stand, x_test_stand), axis=0)
    y_labeled = np.concatenate((y_train, y_test))
    x_unlabeled = x2_test_stand
    y_unlabeled = y2_test
    y_data2 = np.concatenate((y_labeled, y_unlabeled))
    Whole_Train_Data = Setup_whole_training_data(x_labeled, y_labeled, x_unlabeled, y_unlabeled)
    Whole_Train_Data.split_balance_data(split_param)
    x_whole_train, x_whole_test, y_whole_train, y_whole_test = Whole_Train_Data.get_train_data()
    x_whole_train_stand = preprocessing.scale(x_whole_train)
    x_whole_test_stand = preprocessing.scale(x_whole_test)
    y_whole_train_categorical = np_utils.to_categorical(y_whole_train)
    y_whole_test_categorical = np_utils.to_categorical(y_whole_test)
    classweight2 = class_weight.compute_class_weight('balanced', np.unique(y_data2), y_data2)

    ## Training with the whole data with same NNs
    whole_test_path = ["whole1.csv", "whole2.csv", "whole3.csv", "whole4.csv", "whole5.csv", "whole6.csv"]
    model_path2 = [output_file + "best_model1.h5", output_file + "best_model2.h5", output_file + "best_model3.h5",
                   output_file + "best_model4.h5", output_file + "best_model5.h5", output_file + "best_model6.h5"]
    num_nn2 = len(whole_test_path)
    for i in range(num_nn2):
        # define and fit model
        model, history = keras_learning(x_whole_train_stand, x_whole_test_stand, y_whole_train_categorical, y_whole_test_categorical,
                                        input_layer_num, output_layer_num, classweight2, batch_param[i], n_nodes1[i], n_nodes2[i],
                                        n_nodes3[i], drop_out[i], model_path2[i])

        # predict the test dataset
        best_model2 = load_model(model_path2[i])
        pred_test, true_test, pred_z = keras_predict(x_whole_train_stand, x_whole_test_stand, y_whole_train_categorical,
                                                     y_whole_test_categorical, z_test_stand, best_model2)

        # write pseudo_label to csv file
        write_csv(os.path.join(output_file, whole_test_path[i]), pred_z)
        print("batch param: ", batch_param[i])
        print("n_nodes1: ", n_nodes1[i])
        print("n_nodes2: ", n_nodes2[i])
        print("n_nodes3: ", n_nodes3[i])
        print("drop_out: ", drop_out[i])

    print('Generate final result based on the most common labels')
    whole_path = [output_file + "whole1.csv", output_file + "whole2.csv", output_file + "whole3.csv",
                  output_file + "whole4.csv", output_file + "whole5.csv", output_file + "whole6.csv"]
    prediction = most_common_selection(num_nn2, whole_path)
    write_csv(os.path.join(output_file, "result.csv"), prediction)

if __name__ == '__main__':
    main()
