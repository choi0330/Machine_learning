import numpy as np
from load_dataset_2 import ImportData

input_file = "../data/train.csv"
output_file = "../result/4"

# define instances from class 
Train_Data = ImportData(input_file)
Train_Data.read_split_data(0.2, False)
x_data, y_data = Train_Data.get_xandy()


print(np.sum(y_data==0))
print(np.sum(y_data==1))
print(np.sum(y_data==2))
print(y_data.shape)
