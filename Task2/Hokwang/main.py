import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC

epoch = 1

test = np.array(pd.read_csv("./test.csv"))
X_test = test[:,1:]
train = np.array(pd.read_csv("./train.csv"))
X, y = train[:,2:], train[:,1]

y_result1 = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovo').fit(X, y).predict(X_test)
y_result2 = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr').fit(X, y).predict(X_test)
# y_result3 = GaussianProcessClassifier(kernel=None, random_state=0).fit(X, y).predict(X_test)

print("Result1: ",y_result1)
print("Result2: ",y_result2)

sample1 = np.array(pd.read_csv("./sample.csv"))
sample2 = np.array(pd.read_csv("./sample.csv"))
sample1[:,1] = y_result1
sample2[:,1] = y_result2
pd.DataFrame(sample1, columns=['Id', 'y']).to_csv("./OVR_1_rbf.csv",index = None)
pd.DataFrame(sample2, columns=['Id', 'y']).to_csv("./OVO_1_rbf.csv",index = None)
