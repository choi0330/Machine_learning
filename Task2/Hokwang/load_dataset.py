import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv

class ImportData:
    def __init__(self, filepath, splitsize):
        self.filepath = filepath
        self.splitsize = splitsize

    def read_csv(self):
        self.df = pd.read_csv(self.filepath, index_col=0)
        print("[INFO] Read csv data")

    def shuf_data(self):
        ### shuffle the data and reset the index
        self.df_s = self.df.sample(frac=1).reset_index(drop=True)
        print("[INFO] Shuffled csv data")

    def split_data(self):
        # drop column(Id & y) = get data from all columns except Id and y
        self.x_data = self.df_s.drop('y',axis='columns')
        # get data from y columns
        self.y_data = self.df_s['y']
        print("[INFO] Split data: done!")
        print("[INFO] shape of x_data:{}".format(self.x_data.shape))
        print("[INFO] shape of y_data:{}".format(self.y_data.shape))
        return self.x_data, self.y_data

    def split_traintest(self):
        # split the data into train and test
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x_data, self.y_data, test_size=self.splitsize)
        print("[INFO] Made train&test data!")
        print("[INFO] shape of x_train:{}".format(self.x_train.shape))
        print("[INFO] shape of x_test:{}".format(self.x_test.shape))
        print("[INFO] shape of y_train:{}".format(self.y_train.shape))
        print("[INFO] shape of y_test:{}".format(self.y_test.shape))
        return self.x_train,self.x_test, self.y_train, self.y_test

# Data writing, reading
def write_csv(path, result_data):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], result_data))
        file.close()

def data_set(num_shuffle):
    i_data = ImportData("./train.csv", 0.1)
    i_data.read_csv()
    for i in range(num_shuffle):
        i_data.shuf_data()
    train = np.array(i_data.df_s)
    return train

def main():

    if(len(sys.argv)!=2):
        print("[ERROR] !!Missing some arguments!!")
        print("[ERROR] Usage: $ python ThisFile.py ${arg[1]} ${arg[2]} <ENTER>")
        print("[ERROR] arg[1]: path to the dataset")
    else:
        argv = sys.argv
        file_path = argv[1]
        split_size = float(0.1)
        #split_size = float(argv[2])

    i_data = ImportData(file_path, split_size)
    i_data.read_csv()
    i_data.shuf_data()
    i_data.split_data()


if(__name__=='__main__'):
    main()
