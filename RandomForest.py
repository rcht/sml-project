from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import process_data

rf = RandomForestClassifier(n_estimators=100, random_state=14)
rf.fit(process_data.train_dataset.T, process_data.y_train)
rf_predictions = rf.predict(process_data.test_dataset.T)
accuracy = accuracy_score(process_data.y_test, rf_predictions)

rf1 = RandomForestClassifier(n_estimators=100, random_state=14)
rf1.fit(process_data.X_train1.T, process_data.y_train1)
rf_predictions1 = rf1.predict(process_data.X_test1.T)
accuracy1 = accuracy_score(process_data.y_test1, rf_predictions1)

if __name__ == '__main__':
    print(accuracy)
    print(accuracy1)