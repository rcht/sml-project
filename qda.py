from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import process_data

qda = QuadraticDiscriminantAnalysis()
qda.fit(process_data.train_dataset.T, process_data.y_train)
qda_predictions = qda.predict(process_data.test_dataset.T)
qda_accuracy = accuracy_score(process_data.y_test, qda_predictions)

qda1 = QuadraticDiscriminantAnalysis()
qda1.fit(process_data.X_train1.T, process_data.y_train1)
qda_predictions1 = qda1.predict(process_data.X_test1.T)
qda_accuracy1 = accuracy_score(process_data.y_test1, qda_predictions1)
