from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import process_data

adaboost = AdaBoostClassifier(n_estimators=100, random_state=14)
adaboost.fit(process_data.train_dataset.T, process_data.y_train)
adaboost_predictions = adaboost.predict(process_data.test_dataset.T)
adaboost_accuracy = accuracy_score(process_data.y_test, adaboost_predictions)