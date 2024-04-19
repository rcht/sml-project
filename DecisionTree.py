import process_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

for i in range(1):
    dt = DecisionTreeClassifier()
    dt.fit(process_data.train_dataset.T, process_data.y_train)
    dt_predictions = dt.predict(process_data.test_dataset.T)
    dt_accuracy = accuracy_score(process_data.y_test, dt_predictions)
    print("Decision Tree Accuracy:", dt_accuracy)
