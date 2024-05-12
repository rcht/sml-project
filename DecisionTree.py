import process_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier()
dt.fit(process_data.train_dataset.T, process_data.y_train)
dt_predictions = dt.predict(process_data.test_dataset.T)
dt_accuracy = accuracy_score(process_data.y_test, dt_predictions)

dt1 = DecisionTreeClassifier()
dt1.fit(process_data.X_train1.T, process_data.y_train1)
dt_predictions = dt1.predict(process_data.X_test1.T)
dt_accuracy1 = accuracy_score(process_data.y_test1, dt_predictions)

dt = dt_accuracy
dt1 = dt_accuracy1

if __name__ == '__main__':
    print(dt)
    print(dt1)
