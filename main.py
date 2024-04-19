import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart_attack_prediction_dataset.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
X_train = X_train.T
X_test = X_test.T
# print(X_train.shape)
first_row = X_train[:, 1]
string_columns = []
print(first_row.shape)
for i in range(first_row.shape[0]):
    if type(first_row[i]) is str:
        string_columns.append(i)
string_columns = string_columns[1:]
print(string_columns)
for i in string_columns:
    unique_values = np.unique(X_train[i])
    dictionary = {}
    for j in unique_values:
        dictionary[j] = len(dictionary)
    print(dictionary)