import numpy as np
from load_dataset_2 import ImportData

input_file = "../data/train.h5"
output_file = "../result/1"

# define instances from class 
Train_Data = ImportData(input_file)
Train_Data.read_split_data(0.1, False)
x_data, y_data = Train_Data.get_xandy()


print(np.sum(y_data==1))
print(np.sum(y_data==2))
print(np.sum(y_data==3))
print(np.sum(y_data==4))
print(y_data.shape)
