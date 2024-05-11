import RandomForest
import DecisionTree
import lda
import qda
import LogisticRegression
import AdaBoost

print("Random Forest Accuracy:", RandomForest.accuracy)
print("Decision Tree Accuracy:", DecisionTree.dt_accuracy)
print("LDA Accuracy:", lda.accuracy)
print("QDA Accuracy:", qda.qda_accuracy)
print("Logistic Regression Accuracy:", LogisticRegression.accuracy)
print("AdaBoost Accuracy:", AdaBoost.adaboost_accuracy)

print("Best Accuracy:", max(RandomForest.accuracy, DecisionTree.dt_accuracy, lda.accuracy, qda.qda_accuracy,
                            LogisticRegression.accuracy, AdaBoost.adaboost_accuracy))
