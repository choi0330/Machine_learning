# Task0

This task is a trivial form of regression: Your goal is to predict a value **y** based on a vector **x**. While the exact relationship is usually not known, in this task, **y** is the mean of **x**. You may verify this on the provided training set. Your task is to make predictions for **y** on the provided test set.

## Dataset

* [train.csv](./dataset/train.csv)

  the training set

  Each line in train.csv is one data instance indexed by an Id. It consists of one double for y and 10 doubles for the vector x1-x10.

* [test.csv](./dataset/test.csv)

  the test set (make predictions based on this file)

  The test set file (test.csv) has the same structure except that the column for y is omitted

* [sample.csv](./dataset/sample.csv)

  a sample submission file in the correct format.

  **For every data instance in the test set**, submission files should contain two columns: *Id* and *y* where *y* should be a double with your prediction.

  The file should contain a header.

## Evaluation

The evaluation metric for this task is the **Root Mean Squared Error** which is the square root of the mean/average of the square of all of the error.

<img src="https://latex.codecogs.com/gif.latex?ERMS&space;=&space;\sqrt{\frac{1}{n}\sum_{i=1}^n&space;(y_i-\hat{y_i})^{2}}">

* How to compute it in Python

  ```python
  from sklearn.metrics import mean_squared_error
  RMSE = mean_squared_error(y, y_pred)**0.5
  ```



## Grading

We provide you with **one test set** for which you have to compute predictions. We have partitioned this test set into two parts and use it to compute a *public* and a *private* score for each submission. You only receive feedback about your performance on the public part in the form of the public score, while the private leaderboard remains secret. The purpose of this division is to prevent overfitting to the public score. Your model should generalize well to the private part of the test set.

When handing in the task, you need to select which of your submissions will get graded and provide a short description of your approach. This has to be done **individually by each member** of the team. We will then compare your selected submission to three baselines (easy, medium and hard). Your final grade depends on the public score and the private score (weighted equally), on your submitted code and on a properly-written description of your approach. The following **non-binding** guidance provides you with an idea on what is expected to obtain a certain grade: If you hand in a properly-written description, your source code is runnable and reproduces your predictions, and your submission performs better than the easy baseline, you may expect a grade exceeding a 4. If it further beats the medium baseline, you may expect that the grade will exceed a 5. If in addition your submission performs equal to or better than the hard baseline, you may expect a 6. If you do not hand in a properly-written description of your approach, you may obtain zero points regardless of how well your submission performs.



## Submission

You can assume the data will be available under the path that you specify in your code. For example, you could read in the dataset as:

```python
import pandas as pd
df_train = pd.read_csv('train.csv')
```
