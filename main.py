import RandomForest
import DecisionTree
import lda
import qda
import LogisticRegression
import AdaBoost
import Bagging

dictionary = {0: "Random Forest", 1: "Decision Tree", 2: "QDA", 3: "Logistic Regression", 4: "LDA", 5: "AdaBoost",
              6: "Bagging"}
print("Test Accuracy Results for Dataset 1:")
print("Random Forest Accuracy:", RandomForest.accuracy)
print("Decision Tree Accuracy:", DecisionTree.dt)
print("LDA Accuracy:", lda.accuracy)
print("QDA Accuracy:", qda.qda_accuracy)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy)
print("AdaBoost Accuracy:", AdaBoost.ada)
print("Bagging Accuracy:", Bagging.bagging_accuracy)
print()
Accuracy = [RandomForest.accuracy, DecisionTree.dt, qda.qda_accuracy, LogisticRegression.accuracy,lda.accuracy,
            AdaBoost.ada, Bagging.bagging_accuracy]
Model = dictionary[Accuracy.index(max(Accuracy))]
print("Best Model:", Model)
print("Best Accuracy:", max(Accuracy))

print()
print("Test Accuracy Results for Dataset 2:")
print("Random Forest Accuracy:", RandomForest.accuracy1)
print("Decision Tree Accuracy:", DecisionTree.dt1)
print("AdaBoost Accuracy:", AdaBoost.ada1)
print("Bagging Accuracy:", Bagging.bagging_accuracy1)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy1)
print("LDA Accuracy:", lda.accuracy1)
print("QDA Accuracy:", qda.qda_accuracy1)
print()
Accuracy1 = [RandomForest.accuracy1, DecisionTree.dt1, qda.qda_accuracy1, LogisticRegression.accuracy1, lda.accuracy1,
             AdaBoost.ada1, Bagging.bagging_accuracy1]
Model1 = dictionary[Accuracy1.index(max(Accuracy1))]
print("Best Model:", Model1)
print("Best Accuracy:", max(Accuracy1))
BP = input("Enter 1 if High Blood Pressure 0 if not: ")
Chol = input("Enter 0 if low cholestrol and 1 if high: ")
BMI1 = input("Enter BMI: ")
Smoking = input("Enter 1 if smoker 0 if not: ")
Diabetes = input("Enter 1 if diabetic 0 if not: ")
Alcohol = input("Enter 1 if heavy alcohol consumer 0 if not: ")
Sex1 = input("Enter 0 if female 1 if male: ")
Walk = input("Enter 1 if difficulty walking 0 if not: ")
Age1 = input("Enter Age: ")
Input = {"HighBp": BP, "HighChol": Chol, "CholCheck": 1, "BMI": BMI1, "Smoking": Smoking, "Stroke": 0,
         "Diabetes": Diabetes, "Physical Activity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": Alcohol,
         "AnyHealthCare": 0, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 18, "PhysHlth": 20, "DiffWalk": Walk,
         "Sex": Sex1, "Age": Age1, "Education": 4, "Income": 4}
