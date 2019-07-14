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

def write_csv(path, result_data):
    with open(path, 'w') as file:
        header = ['Id','y']
        writer = csv.writer(file)
        writer.writerow(header)
        #writer.writerows(map(lambda x: [x], result_data))
        writer.writerows(create_data(result_data))
        file.close()

def most_common_selection(num_data):
    name = '1.csv'
    # path = './result/' + name
    path = './' + name

    a = read_file(path)
    a = a[:,1]
    for i in range(num_data-1):
        name = ['2.csv','3.csv','4.csv','5.csv','6.csv']
        # path = './result/' + name[i]
        path = './' + name[i]
        b = read_file(path)
        b = b[:,1]
        # print(b.shape)
        a = np.column_stack((a,b))

    d = 4
    for i in range(len(a[:,0])-1):
        # print(a[i+1,:])
        b = np.array(a[i+1, :])
        cnt = Counter(b)
        c = cnt.most_common(1)
        # print(c)
        d = np.row_stack((d,c[0][0]))
    # print(d)
    d = d.flatten()
    output_file = './'
    write_csv(os.path.join(output_file, "result.csv"), d)
