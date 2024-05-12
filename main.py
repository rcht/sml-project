import RandomForest
import DecisionTree
import lda
import qda
import LogisticRegression
import AdaBoost
import Bagging
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np

dictionary = {0: "Random Forest", 1: "Decision Tree", 2: "QDA", 3: "Logistic Regression", 4: "LDA", 5: "AdaBoost",
              6: "Bagging"}
print("Test Accuracy Results for Dataset 1:")
print("Random Forest Accuracy:", RandomForest.accuracy)
print("Decision Tree Accuracy:", DecisionTree.dt_acc)
print("LDA Accuracy:", lda.accuracy)
print("QDA Accuracy:", qda.qda_accuracy)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy)
print("AdaBoost Accuracy:", AdaBoost.ada)
print("Bagging Accuracy:", Bagging.bagging_accuracy)
print()
Accuracy = [RandomForest.accuracy, DecisionTree.dt_acc, qda.qda_accuracy, LogisticRegression.accuracy,lda.accuracy,
            AdaBoost.ada, Bagging.bagging_accuracy]
Model = dictionary[Accuracy.index(max(Accuracy))]
print("Best Model:", Model)
print("Best Accuracy:", max(Accuracy))

print()
print("Test Accuracy Results for Dataset 2:")
print("Random Forest Accuracy:", RandomForest.accuracy1)
print("Decision Tree Accuracy:", DecisionTree.dt1_acc)
print("AdaBoost Accuracy:", AdaBoost.ada1)
print("Bagging Accuracy:", Bagging.bagging_accuracy1)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy1)
print("LDA Accuracy:", lda.accuracy1)
print("QDA Accuracy:", qda.qda_accuracy1)
print()
Accuracy1 = [RandomForest.accuracy1, DecisionTree.dt1_acc, qda.qda_accuracy1, LogisticRegression.accuracy1, lda.accuracy1,
             AdaBoost.ada1, Bagging.bagging_accuracy1]
Model1 = dictionary[Accuracy1.index(max(Accuracy1))]
print("Best Model:", Model1)
print("Best Accuracy:", max(Accuracy1))

while True:
    cont = input("Do you want to classify a sample? (default: no) ").strip()
    if not cont:
        break
    BP = input("Do you have high BP? (if not, leave empty): ").strip()
    BP = 1 if BP else 0
    Chol = input("Do you have high cholestrol? (if not, leave empty): ").strip()
    Chol = 1 if Chol else 0
    BMI1 = float(input("Enter BMI: "))
    Smoking = input("Are you a smoker? (if not, leave empty): ").strip()
    Smoking = 1 if Smoking else 0
    Diabetes = input("Do you have diabetes? (if not, leave empty): ").strip()
    Diabetes = 1 if Diabetes else 0
    Alcohol = input("Do you consume alcohol regularly? (if not, leave empty): ").strip()
    Alcohol = 1 if Alcohol else 0
    Walk = input("Do you have difficulty walking? (if not, leave empty): ").strip()
    Walk = 1 if Walk else 0
    # Walk = input("Enter 1 if difficulty walking 0 if not: ")
    Sex1 = input("Enter 0 if female 1 if male: ").strip()
    Sex1 = int(Sex1)
    print("Age Ranges: 1: 10-15 2: 16-20 3: 21-30 4: 31-40 5: 41-50 6: 51-60 7: 61-70 8: 71-80 9: 81-90 10: 91-100")
    Age1 = float(input("Enter Age Range: "))
    Input = {"HighBp": BP, "HighChol": Chol, "CholCheck": 1, "BMI": BMI1, "Smoking": Smoking, "Stroke": 0,
             "Diabetes": Diabetes, "Physical Activity": (1 if BMI1 < 25 else 0), "Fruits": 1, "Veggies": (1 if BMI1 < 25 else 0), "HvyAlcoholConsump": Alcohol,
             "AnyHealthCare": 0, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": (0 if Alcohol else 18) , "PhysHlth": (20 if BMI1 < 25 and Walk == 0 else 0), "DiffWalk": Walk,
             "Sex": Sex1, "Age": Age1, "Education": 4, "Income": 4}
    InputArr = np.array([i for i in Input.values()])
    prediction = DecisionTree.dt1.predict(InputArr.reshape(1, -1))
    print(prediction)
plt.figure(figsize=(10, 6))
plot_tree(DecisionTree.dt1, filled=True)
plt.show()

plt.figure(figsize=(10, 6))
plot_tree(DecisionTree.dt, filled=True)
plt.show()