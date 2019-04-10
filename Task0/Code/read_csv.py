import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train = pd.read_csv('../../dataset/train.csv')
# test = pd.read_csv('../../dataset/test.csv')

def read_file(path):
    return np.array(pd.read_csv(path))
