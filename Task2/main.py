from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Activation,Dropout
from keras.utils import np_utils
from keras.optimizers import Adam

import csv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from load_dataset_2 import ImportData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns

batch_param =64
epoch_param = 100
'''
input_file = "./train.csv"
output_file = "./"
test_file = "./test.csv"
'''

input_file = "./data/train.csv"
output_file = "./result/final"
test_file = "./data/test.csv"


def keras_learning(x_train, x_test, y_train, y_test, input_layer_num):
    model = Sequential()

    model.add(Dense(input_dim=input_layer_num,
                    output_dim=50, bias=True,
                    activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(input_dim=50,
                    output_dim=100, bias=True,
                    activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(input_dim=100,
                    output_dim=200, bias=True,
                    activation='relu'))

    #model.add(Dropout(0.1))

    model.add(Dense(input_dim=200,
                    output_dim=3, bias=True,
                    activation='softmax'))

    #model.add(Dropout(0.1))

    model.compile(optimizer=Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                     batch_size=batch_param, epochs=epoch_param,
                     validation_data=(x_test,y_test))

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

def create_data(result_data):
    data_mat = []
    for i, data in enumerate(result_data):
        buf = [2000+i, data]
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
    Train_Data.read_split_data(0.01, False)
    Test_Data.read_split_data(0.0, True)
    # get train and test data
    x_train, x_test, y_train, y_test = Train_Data.get_train_data()
    z_test = Test_Data.get_test_data()
    # get dimension of x
    input_layer_num = x_train.shape[1]
    # one-hot encoding(ex.[1,2,0] -> [[0,1,0],[0,0,1],[1,0,0]])
    # keras only accepts one-hot label
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    # define and fit model
    model,history = keras_learning(x_train, x_test, y_train, y_test, input_layer_num,)
    # predict the test dataset
    pred_test, true_test,pred_z = keras_predict(x_train, x_test, y_train, y_test,z_test,model)
    # plot loss and accuracy
    plot_history(history,output_file)
    # create confusion matrix
    print_cmx(pred_test, true_test, output_file)
    # write pred_z to csv file
    write_csv(os.path.join(output_file, "result.csv"),pred_z)
    # classification report(accuracy, precision, recall ,p-value)
    write_text(os.path.join(output_file, "report.txt"),classification_report(true_test, pred_test))

if __name__ == '__main__':
    main()
