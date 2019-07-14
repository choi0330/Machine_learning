import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import h5py
import csv
import tables

class ImportData:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_split_data(self,splitsize, test_data=True):
        self.df = pd.read_hdf(self.filepath, index_col=0)
        print("[INFO] Read hdf data")
        if(test_data == False):
            # shuffle data
            self.df_s = self.df.sample(frac=1).reset_index(drop=True)
            print("[INFO] Shuffled hdf data")
            # drop column(Id & y) = get data from all columns except Id and y
            self.x_data = self.df_s.drop('y',axis='columns')
            # get data from y columns
            self.y_data = self.df_s['y']
            print("[INFO] Split data: done!")
            # split the data into train and test
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x_data, self.y_data, test_size=splitsize)
            print("[INFO] shape of x_train:{}".format(self.x_train.shape))
            print("[INFO] shape of x_test:{}".format(self.x_test.shape))
            print("[INFO] shape of y_train:{}".format(self.y_train.shape))
            print("[INFO] shape of y_test:{}".format(self.y_test.shape))
        else:
            self.z_test = self.df
            print("[INFO] shape of test:{}".format(self.z_test.shape))

    def get_train_data(self):
        return np.array(self.x_train),np.array(self.x_test), np.array(self.y_train), np.array(self.y_test)
    def get_test_data(self):
        return np.array(self.df)
    def get_xandy(self):
        return np.array(self.x_data), np.array(self.y_data)

# Data writing, reading
def write_csv(path, result_data):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], result_data))
        file.close()
