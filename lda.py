from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import process_data

lda = LinearDiscriminantAnalysis()
lda.fit(process_data.train_dataset.T, process_data.y_train)
lda_predictions = lda.predict(process_data.test_dataset.T)
accuracy = accuracy_score(process_data.y_test, lda_predictions)

lda1 = LinearDiscriminantAnalysis()
lda1.fit(process_data.X_train1.T, process_data.y_train1)
lda_predictions1 = lda1.predict(process_data.X_test1.T)
accuracy1 = accuracy_score(process_data.y_test1, lda_predictions1)
