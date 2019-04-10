import load_dataset
from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
import csv
import pandas as pd
import numpy as np

<<<<<<< 3cb49f1d116f486d428fc0d590542d471846c454
# Read data
read = np.array(pd.read_csv("../../Task1a/data/train.csv"))
D = read[:,1:]
samples, parameters = D.shape

X = D[: , 1:]
y = D[: , 0]

lambdas = [0.1, 1, 10, 100, 1000]
ridges = [Ridge(alpha = lam) for lam in lambdas]
results = []

kf = KFold(n_splits = 10, shuffle = True)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    w = []
    for ridge in ridges:
        temp = ridge.fit(X_train, y_train)
        w.append( (2e-3 * np.sum( (y_test - temp.predict(X_test)) ** 2 )) ** 0.5 )
    results.append(w)

=======
# Data writing, reading
>>>>>>> Jayce_changes
def write_csv(path, result_data):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], result_data))
        file.close()

def data_set(num_shuffle):
    i_data = load_dataset.ImportData("../../Task1a/data/train.csv", 0.1)
    i_data.read_csv()
    for i in range(num_shuffle):
        i_data.shuf_data()
    train = np.array(i_data.df_s)
    return train

def rr(X,y,lamb):
    ridge_model = Ridge(alpha=lamb, fit_intercept = False)
    ridge_model.fit(X, y)
    w = ridge_model.coef_
    return w, ridge_model


def rrCV_RMSE(X,y,lamb,k):
    ridge_model = RidgeCV(alphas=lamb, cv = k, fit_intercept = False)
    ridge_model.fit(X, y)
    w = ridge_model.coef_
    lamb = ridge_model.alpha_
    return ridge_model

def k_fold_with_lr(train, k):
    fold_size = k
    kf = KFold(n_splits = fold_size)
    kf.get_n_splits(train)
    counter = 1
    w_aver = np.zeros(21)

    for train_index, test_index in kf.split(train):
        print(counter,"th fold")
        counter += 1
        X_train, X_test = train[train_index, :-1], train[test_index, :-1]
        y_train, y_test = train[train_index, -1], train[test_index, -1]
        w_star = lr(X_train,y_train)
        w_aver = w_aver + w_star
    w_aver = w_aver*(k**(-1))
    return w_aver


def k_fold_with_rr_RMSE(train, Params, k):
    fold_size = k
    kf = KFold(n_splits = fold_size)
    kf.get_n_splits(train)
    min = 100000000000000000000000

    result = []
    for i, compo in enumerate(Params):
        print(i+1,"th component")
        sum = 0
        counter = 1
        RMSE = 0
        RMSE_mean = 0
        for train_index, test_index in kf.split(train):
            print(counter,"th fold")
            counter += 1
            X_train, X_test = train[train_index, 1:], train[test_index, 1:]
            y_train, y_test = train[train_index, 0], train[test_index, 0]
            w_star, regr = rr(X_train,y_train, compo)
            y_hat = regr.predict(X_test)
            sum = np.sum(np.square(y_test-y_hat))
            RMSE = np.sqrt(sum*(np.size(y_test))**(-1))
            RMSE_mean = RMSE_mean+RMSE
        RMSE_mean = RMSE_mean*(k)**(-1)
        print(RMSE_mean)
        result.append(RMSE_mean)
    return result


#read data
train = data_set(1)
print(train)
y, X= train[:,0],train[:,1:]
k = 10
lambdas = [0.1, 1, 10, 100, 1000]

answer = k_fold_with_rr_RMSE(train, lambdas, k)
print(answer)

print(answer)
write_csv("../../Task1a/Hokwang/result.csv", answer)
