from sklearn import datasets, linear_model
import pandas as pd
import numpy as np

from read_csv import read_file

train = read_file("train.csv")
print(train)



# Create linear regression object
regr = linear_model.LinearRegression()

y = train[:,1]
X = train[:,2:]

print("y: ", y)
print(X)

# Train the model using the training sets
regr.fit(X, y)
b = regr.coef_
print("b:")
print(b)


test = read_file("test.csv")

result = regr.predict(test[:,1:])
print ("Result:")
print (result)

indexed_result = []
for i, value in enumerate(result):
    indexed_result.append([i+10000, value])


df = pd.DataFrame(indexed_result, columns = ["Id", "y"])
df.to_csv("result.csv", index = False)
