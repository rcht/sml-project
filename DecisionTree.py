import process_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
decisiontree = []
decisiontree1 = []
for i in range(0, 10):
    dt = DecisionTreeClassifier(max_depth=i+1)
    dt.fit(process_data.train_dataset.T, process_data.y_train)
    dt_predictions = dt.predict(process_data.test_dataset.T)
    dt_accuracy = accuracy_score(process_data.y_test, dt_predictions)
    decisiontree.append(dt_accuracy)

    dt1 = DecisionTreeClassifier(max_depth=i+1)
    dt1.fit(process_data.X_train1.T, process_data.y_train1)
    dt_predictions = dt1.predict(process_data.X_test1.T)
    dt_accuracy1 = accuracy_score(process_data.y_test1, dt_predictions)
    decisiontree1.append(dt_accuracy1)

dt = max(decisiontree)
dt1 = max(decisiontree1)

if __name__ == '__main__':
    print(decisiontree)
    print(decisiontree1)
