from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import process_data

rf = RandomForestClassifier(n_estimators=100, random_state=14)
rf.fit(process_data.train_dataset.T, process_data.y_train)
rf_predictions = rf.predict(process_data.test_dataset.T)
accuracy = accuracy_score(process_data.y_test, rf_predictions)

