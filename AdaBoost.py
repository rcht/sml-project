from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import process_data

adaboost_accuracies = []
for i in range(0, 10):
    ada = AdaBoostClassifier(n_estimators=i+1, random_state=i+1, algorithm='SAMME')
    ada.fit(process_data.train_dataset.T, process_data.y_train)
    ada_predictions = ada.predict(process_data.test_dataset.T)
    ada_accuracy = accuracy_score(process_data.y_test, ada_predictions)
    adaboost_accuracies.append(ada_accuracy)
