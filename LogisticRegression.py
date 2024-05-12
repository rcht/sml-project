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

scaler1 = StandardScaler()
train_scaled1 = scaler1.fit_transform(process_data.X_train1.T)
test_scaled1 = scaler1.transform(process_data.X_test1.T)
lr1 = LogisticRegression(max_iter=10, random_state=67)
lr1.fit(train_scaled1, process_data.y_train1)
lr_predictions1 = lr1.predict(test_scaled1)
accuracy1 = accuracy_score(process_data.y_test1, lr_predictions1)
