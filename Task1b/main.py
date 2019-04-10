import csv
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn import datasets, linear_model
from numpy.linalg import inv
from regressors import LinearRegressor                  # From demos by Andreas Krause
from regularizers import Regularizer, L2Regularizer     # From demos by Andreas Krause
from util import gradient_descent                       # From demos by Andreas Krause
import load_dataset
from sklearn.datasets import load_iris
from sklearn import preprocessing

# Data writing, reading
def write_csv(path, result_data):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], result_data))
        file.close()

def data_set(num_shuffle):
    i_data = load_dataset.ImportData("./train.csv", 0.1)
    i_data.read_csv()
    for i in range(num_shuffle):
        i_data.shuf_data()
    train = np.array(i_data.df_s)
    y = train[:,0]
    y = y.reshape(1000,1)
    x = train[:,1:]
    X = np.concatenate((x,sqr(x)),axis = 1)
    X = np.concatenate((X,expo(x)),axis = 1)
    X = np.concatenate((X,cos_mat(x)),axis = 1)
    const = np.ones((np.size(y),1))
    X = np.append(X,const,axis = 1)
    train2 = np.append(X, y, axis =1)
    return train2

# element-wise functions
def sqr(x):
    return x**2
def exp(a,x):
    return a**x
def expo(x):
    return [exp(np.e,a) for a in x]
def cos_scalar(x):
    return math.cos(x)
def cos_array(x):
    return [cos_scalar(a) for a in x]
def cos_mat(x):
    return [cos_array(a) for a in x]

# Different methods
def closed_form_sol(X,y):
    Mat = np.matmul(inv(np.matmul(np.transpose(X),X)),np.transpose(X))
    w = Mat.dot(y)
    return w


def lr(X,y):
    regr = LinearRegression(fit_intercept = False)
    regr.fit(X, y)
    w = regr.coef_
    return w


def rr(X,y,lamb):
    ridge_model = Ridge(alpha=lamb, fit_intercept = False)
    ridge_model.fit(X, y)
    w = ridge_model.coef_
    return w, ridge_model


def rrCV(X,y,lamb,k):
    ridge_model = RidgeCV(alphas=lamb, cv = k, fit_intercept = False)
    ridge_model.fit(X, y)
    w = ridge_model.coef_
    lamb = ridge_model.alpha_
    return w, lamb


def lasoCV(X,y,lamb,k):
    ridge_model = LassoCV(alphas=lamb, cv = k, fit_intercept = False)
    ridge_model.fit(X, y)
    w = ridge_model.coef_
    lamb = ridge_model.alpha_
    return w, lamb


def gd(X,y,compo):
    ## Initial guess with closed form solution
    # w0 = closed_form_sol(X,y)
    # w0 = w0.reshape(21)
    w0 = np.zeros(21)

    regularizer = L2Regularizer(compo)    # regularize component
    regressor = LinearRegressor(X,y)
    opts = {'eta0': 0.001,              # learning rate component
            'n_iter': 1000,            # number of iteration component
            'n_samples': X.shape[0],
            'algorithm': 'GD',
            'learning_rate_scheduling': 'Bold driver' # choose
            }
    trajectory, indexes = gradient_descent(w0, regressor, regularizer, opts)
    print(trajectory)
    w_ = trajectory[-1,:]
    return w_


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


def k_fold_with_rr(train, Params, k):
    fold_size = k
    kf = KFold(n_splits = fold_size)
    kf.get_n_splits(train)
    min = 100000000000000000000000

    for i, compo in enumerate(Params):
        print(i+1,"th component")
        sum = 0
        counter = 1
        for train_index, test_index in kf.split(train):
            print(counter,"th fold")
            counter += 1
            X_train, X_test = train[train_index, :-1], train[test_index, :-1]
            y_train, y_test = train[train_index, -1], train[test_index, -1]
            w_star, regr = rr(X_train,y_train, compo)
            result = regr.predict(X_test)
            sum = sum + np.sum(np.square(y_test-result))
        if (min > sum):
            min = sum
            print(min)
            opt_value = compo
            w_fin = w_star
            print("updated: ",opt_value)
    return opt_value, w_fin


def k_fold_with_gd(train, params, k):
    fold_size = k
    kf = KFold(n_splits = fold_size)
    kf.get_n_splits(train)
    min = 100000000000000000000000

    for i, compo in enumerate(params):
        print(i+1,"th component")
        sum = 0
        counter = 1
        for train_index, test_index in kf.split(train):
            print(counter,"th fold")
            counter += 1
            X_train, X_test = train[train_index, :-1], train[test_index, :-1]
            y_train, y_test = train[train_index, -1], train[test_index, -1]
            w_star = gd(X_train,y_train,compo)
            sum = sum + np.sum(np.square(y_test-X_test.dot(w_star)))
        if (min > sum):
            min = sum
            print(min)
            opt_value = compo
            print("updated: ",opt_value)
    return opt_value


# K_fold to find optimal parameters
reg_params = [1e-1, 1, 1e1, 1e2, 1e3, 1e4]
# reg_params = [9e2, 9.2e2, 9.4e2, 9.6e2, 9.8e2, 1e3, 1.02e3, 1.04e3, 1.06e3, 1.1e3, 1.2e3, 1.3e3]
# reg_params = range(980, 1000, 1)
num_iter_params = [10, 100, 200, 500]
learning_rate = [1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-10, 1e-15, 1e-17, 1e-20]
k = 100


# Parameters optimization
# train = np.array(pd.read_csv("../../Task1b/data/train_nonlinear.csv"))
# X, y = train[:,2:], train[:,1]
# y = y.reshape(1000,1)
# train2 = np.append(X, y, axis =1)
# opt_reg1 = k_fold_with_gd(train2, reg_params, k)
# opt_num_iter = k_fold_with_gd(train, num_iter_params, k)
# opt_learn_rate = k_fold_with_gd(train, learning_rate, k)
# print("Optimal value with K_fold with gd: ",opt_reg1)

# opt_reg2, w_fin = k_fold_with_rr(train2, reg_params, k)
# print("Optimal value with K_fold with rr: ",opt_reg2)
## 256 for 500 folds, 274 for 500 folds

# epoch > 1
epoch = 5
w_average = np.zeros(21)

for i in range(epoch):
    train = data_set(i+1)
    X, y = train[:,:-1], train[:,-1]
    w_star, lamb = rrCV(X,y,reg_params,k)
    w_average = w_average + w_star
    print(i+1, "th epoch")
    print("w_star = ",w_star)
    print("lambda_star = ",lamb)
w_average = w_average*(epoch**(-1))

# # epoch = 1
# train = data_set(1)
# X, y = train[:,:-1], train[:,-1]
# w_star, lamb = rrCV(X,y,reg_params,k)
# print("w_star: ",w_star)
# print("Optimal lambda: ",lamb)

write_csv("./final_result.csv", w_star)
