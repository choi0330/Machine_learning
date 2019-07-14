from collections import Counter
import numpy as np
import pandas as pd
import csv
import os.path

def read_file(path):
    return np.array(pd.read_csv(path))

def create_data(result_data):
    data_mat = []
    for i, data in enumerate(result_data):
        buf = [45324+i, data]
        data_mat.append(buf)
    return data_mat

def most_common_selection(num_data, paths):
    path = paths[0]
    Full = read_file(path)
    Full = Full[:,1]
    for i in range(num_data-1):
        path = paths[i+1]
        result_nn = read_file(path)
        result_nn = result_nn[:,1]
        Full = np.column_stack((Full,result_nn))

    cnt = Counter(np.array(Full[0, :]))
    most_common_label = cnt.most_common(1)
    result = most_common_label[0][0]
    for i in range(len(Full[:,0])-1):
        result_nn = np.array(Full[i+1, :])
        cnt = Counter(result_nn)
        most_common_label = cnt.most_common(1)
        result = np.row_stack((result,most_common_label[0][0]))

    print(result)
    result = result.flatten()
    return result
