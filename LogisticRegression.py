from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import process_data

scaler = StandardScaler()
train_scaled = scaler.fit_transform(process_data.train_dataset.T)
test_scaled = scaler.transform(process_data.test_dataset.T)
lr = LogisticRegression(max_iter=10, random_state=67)
lr.fit(train_scaled, process_data.y_train)
lr_predictions = lr.predict(test_scaled)
accuracy = accuracy_score(process_data.y_test, lr_predictions)

