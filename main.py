import RandomForest
import DecisionTree
import lda
import qda
import LogisticRegression
import AdaBoost
import Bagging

ada = AdaBoost.adaboost_accuracies
print("Random Forest Accuracy:", RandomForest.accuracy)
print("Decision Tree Accuracy:", DecisionTree.dt_accuracy)
print("LDA Accuracy:", lda.accuracy)
print("QDA Accuracy:", qda.qda_accuracy)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy)
print("AdaBoost Accuracy:", ada[9])
print("Bagging Accuracy:", Bagging.bagging_accuracy)

print("Best Accuracy:", max(RandomForest.accuracy, DecisionTree.dt_accuracy, lda.accuracy, qda.qda_accuracy,
                            LogisticRegression.accuracy, ada[9], Bagging.bagging_accuracy))
