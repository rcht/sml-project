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
print("Decision Tree Accuracy:", DecisionTree.dt_accuracy)
print("LDA Accuracy:", lda.accuracy)
print("QDA Accuracy:", qda.qda_accuracy)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy)
print("AdaBoost Accuracy:", ada[9])
print("Bagging Accuracy:", Bagging.bagging_accuracy)
print()
print("Best Accuracy:", max(RandomForest.accuracy, DecisionTree.dt_accuracy, lda.accuracy, qda.qda_accuracy,
                            LogisticRegression.accuracy, ada[9], Bagging.bagging_accuracy))
print()
print("Test Accuracy Results for Dataset 2:")
ada1 = AdaBoost.adaboost_accuracies1
print("Random Forest Accuracy:", RandomForest.accuracy1)
print("Decision Tree Accuracy:", DecisionTree.dt_accuracy1)
print("AdaBoost Accuracy:", ada1[9])
print("Bagging Accuracy:", Bagging.bagging_accuracy1)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy1)
print("LDA Accuracy:", lda.accuracy1)
print("QDA Accuracy:", qda.qda_accuracy1)
print()
print("Best Accuracy:", max(RandomForest.accuracy1, DecisionTree.dt_accuracy1, lda.accuracy1, qda.qda_accuracy1,
                            LogisticRegression.accuracy1, ada1[9], Bagging.bagging_accuracy1))