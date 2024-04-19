from typing import final
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from param_names import data_headers

data = pd.read_csv('heart_attack_prediction_dataset.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
X_train = X_train.T
X_test = X_test.T

first_column = X_train[:, 1]
string_rows = []

for rowIndex in range(first_column.shape[0]):
    if type(first_column[rowIndex]) is str and '/' not in first_column[rowIndex]:
        string_rows.append(rowIndex)


for i in string_rows:
    unique_values = np.unique(X_train[i])
    dictionary = {}
    for j in unique_values:
        dictionary[j] = len(dictionary)
    X_train[i] = np.array([dictionary[x] for x in X_train[i]])
    X_test[i] = np.array([dictionary[x] for x in X_test[i]])


train_dataset = []
test_dataset = []
row_names = []

it = 0

for rowIndex in range(X_train.shape[0]):

    row = X_train[rowIndex, :]
    testRow = X_test[rowIndex, :] 
    first_entry = row[0] 

    if type(first_entry) is not str:
        train_dataset.append(row)
        test_dataset.append(testRow)
        row_names.append(data_headers[it])
    else:
        ### train
        bp_list = row.tolist()
        split_bp_list = [bp.split('/') for bp in bp_list]
        systolic_bp_list = np.array([int(bp[0]) for bp in split_bp_list])
        diastolic_bp_list = np.array([int(bp[1]) for bp in split_bp_list])
        train_dataset.append(systolic_bp_list)
        train_dataset.append(diastolic_bp_list)
        #### test
        bp_list = testRow.tolist()
        split_bp_list = [bp.split('/') for bp in bp_list]
        systolic_bp_list = np.array([int(bp[0]) for bp in split_bp_list])
        diastolic_bp_list = np.array([int(bp[1]) for bp in split_bp_list])
        test_dataset.append(systolic_bp_list)
        test_dataset.append(diastolic_bp_list)
        ### idk
        row_names.append("Systolic Blood Pressure")
        row_names.append("Diastolic Blood Pressure")

    it += 1

train_dataset = np.array(train_dataset)
test_dataset = np.array(test_dataset)

if __name__ == '__main__':
    # testing here
    print(X_train.shape)
    print(train_dataset.shape)
    print(train_dataset[3])
    print(train_dataset[4])
    print(len(row_names))

    for i in range(25):
        print(row_names[i], train_dataset[i, 0])

    print(test_dataset)
    print(test_dataset.shape)
