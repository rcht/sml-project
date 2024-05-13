import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("heart_disease_health_indicators.csv")

no_risk_data = data[data['HeartDiseaseorAttack'] == 0]
risk_data = data[data['HeartDiseaseorAttack'] == 1]

no_risk_data_bmi = no_risk_data['BMI']
risk_data_bmi = risk_data['BMI']
no_risk_data_age = no_risk_data['Age']
risk_data_age = risk_data['Age']

plt.figure(figsize=(10, 6))
plt.scatter(risk_data_bmi, risk_data_age, color='red', label='Risk', marker='.', linewidths=0.5)
plt.xlabel('BMI')
plt.ylabel('Age Ranges')
plt.title('Scatter Plot of BMI vs Age for patients with risk')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(no_risk_data_bmi, no_risk_data_age, color='blue', label='No Risk', marker='.', linewidths=0.5)
plt.xlabel('BMI')
plt.ylabel('Age Ranges')
plt.title('Scatter Plot of BMI vs Age for patients with no risk')
plt.legend()
plt.show()

no_risk_data_diabetes = no_risk_data['Diabetes']
risk_data_diabetes = risk_data['Diabetes']
plt.figure(figsize=(10, 6))
sns.histplot(data['Diabetes'], bins=3)
plt.xticks([0, 1, 2])
plt.ylabel('Number of Patients')
plt.title('Histogram of Diabetes')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(no_risk_data_diabetes, bins=3, color='green')
plt.xticks([0, 1, 2])
plt.ylabel('Number of Patients')
plt.title('Histogram of Diabetes for patients with no risk')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(risk_data_diabetes, bins=3, color='red')
plt.xticks([0, 1, 2])
plt.ylabel('Number of Patients')
plt.title('Histogram of Diabetes for patients with risk')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(no_risk_data['HvyAlcoholConsump'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Heavy Alcohol Consumption with no risk')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(risk_data['HvyAlcoholConsump'], bins=2, color='red')
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Heavy Alcohol Consumption with risk')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(no_risk_data['Smoker'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Smoking with no risk')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(risk_data['Smoker'], bins=2, color='red')
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Smoking with risk')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(no_risk_data['HighChol'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of High Cholesterol with no risk')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(risk_data['HighChol'], bins=2, color='red')
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of High Cholesterol with risk')
plt.show()

plt.figure(figsize=(14, 14))
data1 = data.drop(['HeartDiseaseorAttack'], axis=1)
sns.heatmap(data1.corr(), annot=True, cmap='coolwarm',annot_kws={"size": 5})
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=10)
plt.ylabel('Number of Patients')
plt.title('Histogram of Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['BMI'], bins=10)
plt.ylabel('Number of Patients')
plt.title('Histogram of BMI')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['GenHlth'], bins=10)
plt.ylabel('Number of Patients')
plt.title('Histogram of General Health')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['MentHlth'], bins=10)
plt.ylabel('Number of Patients')
plt.title('Histogram of Mental Health')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['PhysHlth'], bins=10)
plt.ylabel('Number of Patients')
plt.title('Histogram of Physical Health')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['DiffWalk'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Difficulty Walking')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Sex'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Sex')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Education'], bins=10)
plt.ylabel('Number of Patients')
plt.title('Histogram of Education')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Income'], bins=10)
plt.ylabel('Number of Patients')
plt.title('Histogram of Income')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['HighBP'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of High Blood Pressure')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Veggies'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Veggies')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Fruits'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Fruits')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['PhysActivity'], bins=2)
plt.xticks([0, 1])
plt.ylabel('Number of Patients')
plt.title('Histogram of Physical Activity')
plt.show()
