from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import process_data

qda = QuadraticDiscriminantAnalysis()
qda.fit(process_data.train_dataset.T, process_data.y_train)
qda_predictions = qda.predict(process_data.test_dataset.T)
qda_accuracy = accuracy_score(process_data.y_test, qda_predictions)
print("QDA Accuracy:", qda_accuracy)
