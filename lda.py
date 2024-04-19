from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import process_data

lda = LinearDiscriminantAnalysis()
lda.fit(process_data.train_dataset.T, process_data.y_train)
lda_predictions = lda.predict(process_data.test_dataset.T)
accuracy = accuracy_score(process_data.y_test, lda_predictions)
print("LDA Accuracy:", accuracy)

