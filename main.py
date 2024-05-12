import RandomForest
import DecisionTree
import lda
import qda
import LogisticRegression
import AdaBoost
import Bagging


print("Test Accuracy Results for Dataset 1:")
ada = AdaBoost.adaboost_accuracies
print("Random Forest Accuracy:", RandomForest.accuracy)
print("Decision Tree Accuracy:", DecisionTree.dt)
print("LDA Accuracy:", lda.accuracy)
print("QDA Accuracy:", qda.qda_accuracy)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy)
print("AdaBoost Accuracy:", AdaBoost.ada)
print("Bagging Accuracy:", Bagging.bagging_accuracy)
print()
print("Best Accuracy:", max(RandomForest.accuracy, DecisionTree.dt, lda.accuracy, qda.qda_accuracy,
                            LogisticRegression.accuracy, AdaBoost.ada, Bagging.bagging_accuracy))

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
print("Best Accuracy:", max(RandomForest.accuracy1, DecisionTree.dt1, lda.accuracy1, qda.qda_accuracy1,
                            LogisticRegression.accuracy1, AdaBoost.ada1, Bagging.bagging_accuracy1))

