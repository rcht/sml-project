from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import process_data

bagging = BaggingClassifier(n_estimators=100, random_state=14)
bagging.fit(process_data.train_dataset.T, process_data.y_train)
bagging_predictions = bagging.predict(process_data.test_dataset.T)
bagging_accuracy = accuracy_score(process_data.y_test, bagging_predictions)