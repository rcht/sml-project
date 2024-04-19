import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart_attack_prediction_dataset.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = X.T

print(X.shape)
print(y.shape)
