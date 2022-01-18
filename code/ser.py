from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



# evaluate bagging algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

from utils import load_data

import os
import pickle

# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# print('For Y Train: ',y_train)
# print('For Y Test',y_test)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])
# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 40,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 10000,
}
# initialize Multi Layer Perceptron classifier
# initialize KNeighbors Classifier
# initialize Decision Tree Classifier
# initialize Random Forest Classifier

# with best parameters ( so far )
modelmlp = MLPClassifier(**model_params)
modelknn = KNeighborsClassifier(n_neighbors=1)
modeltree = DecisionTreeClassifier(criterion = "gini")
modelrf = RandomForestClassifier(n_estimators=110, random_state=1)
modelvoting = VotingClassifier(estimators=[('MLP', modelmlp), ('KNN', modelknn), ('DT', modeltree) ,('RF', modelrf)], voting='hard')

# train the model
print("[*] Training the model...")
modelmlp.fit(X_train, y_train)
modelknn.fit(X_train, y_train)
modeltree.fit(X_train, y_train)
modelrf.fit(X_train, y_train)
modelvoting.fit(X_train, y_train)

# predict 25% of data to measure how good we are (For Test data)
y_pred_mlp = modelmlp.predict(X_test)
y_pred_knn = modelknn.predict(X_test)
y_pred_tree = modeltree.predict(X_test)
y_pred_rf = modelrf.predict(X_test)
y_preds_vote = modelvoting.predict(X_test)

# For Train Data
y_pred_mlp_train = modelmlp.predict(X_train)
y_pred_knn_train = modelknn.predict(X_train)
y_pred_tree_train = modeltree.predict(X_train)
y_pred_rf_train = modelrf.predict(X_train)
y_preds_vote_train = modelvoting.predict(X_train)


# Testing Rough
# print('MLP: ',y_pred_mlp)
# print('KNN: ',y_pred_knn)

# calculate the accuracy
accuracy_mlp = modelmlp.score(X_test, y_test)
accuracy_knn = modelknn.score(X_test, y_test)
accuracy_tree = modeltree.score(X_test, y_test)
accuracy_rf = modelrf.score(X_test, y_test)
accurecy_vote = modelvoting.score(X_test,y_test)

# print("Voting Score: ",accurecy_vote)

print("Accuracy of MLP: {:.2f}%".format(accuracy_mlp*100))
print("Accuracy of KNN: {:.2f}%".format(accuracy_knn*100))
print("Accuracy of DT: {:.2f}%".format(accuracy_tree*100))
print("Accuracy of RF: {:.2f}%".format(accuracy_rf*100))
averaged_accurecy = (accuracy_mlp + accuracy_knn + accuracy_tree + accuracy_rf)/4
print("Average accurecy: {:.2f}%".format(averaged_accurecy*100))
print("Accuracy of Voting: {:.2f}%".format(accurecy_vote*100))


# Testing Rough
# accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_mlp)
# print("Accuracy: {:.2f}%".format(accuracy*100))

# Confution matrix 
# print(confusion_matrix(y_test, y_pred_mlp))
# print(confusion_matrix(y_test, y_pred_knn))
# print(confusion_matrix(y_test, y_pred_tree))
# print(confusion_matrix(y_test, y_pred_rf))

# For Testing Test emotion samples........
# print("MLP Confution Matrix:")
# print(classification_report(y_test, y_pred_mlp))
# print("KNN Confution Matrix:")
# print(classification_report(y_test, y_pred_knn))
# print("DT Confution Matrix:")
# print(classification_report(y_test, y_pred_tree))
# print("RF Confution Matrix:")
# print(classification_report(y_test, y_pred_rf))





# Testing Rough
# print("Compare List:", y_test, y_pred_mlp)
# print("Y-Trai Data:", y_train)

# For testing Train emontion samples............
# print("MLP Confution Matrix:")
# print(classification_report(y_train, y_pred_mlp_train))
# print("KNN Confution Matrix:")
# print(classification_report(y_train, y_pred_knn_train))
# print("DT Confution Matrix:")
# print(classification_report(y_train, y_pred_tree_train))
# print("RF Confution Matrix:")
# print(classification_report(y_train, y_pred_rf_train))


# # # now we save the model
# # # make result directory if doesn't exist yet
# if not os.path.isdir("result"):
#     os.mkdir("result")

# pickle.dump(modelmlp, open("result/mlp.model", "wb"))
# pickle.dump(modelknn, open("result/knn.model", "wb"))
# pickle.dump(modeltree, open("result/dt.model", "wb"))
# pickle.dump(modelrf, open("result/rf.model", "wb"))
# pickle.dump(modelvoting, open("result/voting.model", "wb"))

# # Predict test emotion


# # Original test emotion
# # voting_clf2 = VotingClassifier(estimators=[('MLP', modelmlp), ('KNN', modelknn), ('DT', modeltree) ,('RF', modelrf)], voting='hard')
# # voting_clf2.fit(X_test,y_test)
# # preds2 = voting_clf2.predict(X_test)
# # print(preds2)
# # acc2 = accuracy_score(y_test, preds2)
# # print("Voting Section")
# # print("Accuracy is: " + str(acc2))

# # ---Begging---
# # define dataset মডেলের নির্ভুলতার গড় এবং মান বিচ্যুতি প্রতিবেদন করব।
# '''
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# # define the model
# model = BaggingClassifier()
# # evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# print('Begging Model Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# '''
# # ---Boosting---
# # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
# # print('Boosting Model Accuracy:',clf.score(X_test, y_test))


# ----Bagging Fiting---
# X_train, X_test, y_train, y_test
print('\n')
print('DT Bagging: ')
bgDT = BaggingClassifier(DecisionTreeClassifier(criterion = "gini"), max_samples= 1.0, max_features = 180, n_estimators = 30)
bgDT.fit(X_train, y_train)
baggingDT_test = bgDT.score(X_test,y_test)*100
baggingDT_train = bgDT.score(X_train,y_train)*100
print(f'Bagging Test Model Accuracy: {baggingDT_test}  Bagging Train Model Accuracy: {baggingDT_train}')

print('RF Bagging: ')
bgRF = BaggingClassifier(RandomForestClassifier(n_estimators=50, random_state=1), max_samples= 1.0, max_features = 180, n_estimators = 30)
bgRF.fit(X_train, y_train)
baggingRF_test = bgRF.score(X_test,y_test)*100
baggingRF_train = bgRF.score(X_train,y_train)*100
print(f'Bagging Test Model Accuracy: {baggingRF_test}  Bagging Train Model Accuracy: {baggingRF_train}')

print('KNN Bagging: ')
bgKNN = BaggingClassifier(KNeighborsClassifier(n_neighbors=1), max_samples= 1.0, max_features = 180, n_estimators = 30)
bgKNN.fit(X_train, y_train)
baggingKNN_test = bgKNN.score(X_test,y_test)*100
baggingKNN_train = bgKNN.score(X_train,y_train)*100
print(f'Bagging Test Model Accuracy: {baggingKNN_test}  Bagging Train Model Accuracy: {baggingKNN_train}')

print('MLP Bagging: ')
bgMLP = BaggingClassifier(MLPClassifier(**model_params), max_samples= 1.0, max_features = 180, n_estimators = 30)
bgMLP.fit(X_train, y_train)
baggingMLP_test = bgMLP.score(X_test,y_test)*100
baggingMLP_train = bgMLP.score(X_train,y_train)*100
print(f'Bagging Test Model Accuracy: {baggingMLP_test}  Bagging Train Model Accuracy: {baggingMLP_train}')

print('Voting Bagging: ')
baggingModelVoting = VotingClassifier(estimators=[('MLP', bgMLP), ('KNN', bgKNN), ('DT', bgDT) ,('RF', bgRF)], voting='hard')
baggingModelVoting.fit(X_train,y_train)
baggingModelVoting_test = baggingModelVoting.score(X_test,y_test)*100
print(f'Bagging Test Model Accuracy: {baggingModelVoting_test}')


if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(bgDT, open("result/BaggingDT.model", "wb"))
pickle.dump(bgRF, open("result/BaggingRF.model", "wb"))
pickle.dump(bgKNN, open("result/BaggingKNN.model", "wb"))
pickle.dump(bgMLP, open("result/BaggingMLP.model", "wb"))
pickle.dump(baggingModelVoting, open("result/BaggingVoting.model", "wb"))



# # modelvoting = VotingClassifier(estimators=[('MLP', modelmlp), ('KNN', modelknn), ('DT', modeltree) ,('RF', modelrf)], voting='hard')
# # modelvoting.fit(X_train, y_train)


# # ----Boosting Fiting---
# # print('\n\n')
# # print('DT Boosting: ')
# # adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
# # adb.fit(X_train,y_train)
# # boostingDT_test = adb.score(X_test,y_test)
# # boostingDT_train = adb.score(X_train,y_train)
# # print(f'Boosting Test Model Accuracy: {boostingDT_test}  Boosting Train Model Accuracy: {boostingDT_train}')

# # print('\n')
# # print('RF Boosting: ')
# # adb = AdaBoostClassifier(RandomForestClassifier(),n_estimators = 5, learning_rate = 1)
# # adb.fit(X_train,y_train)
# # boostingRF_test = adb.score(X_test,y_test)
# # boostingRF_train = adb.score(X_train,y_train)
# # print(f'Boosting Test Model Accuracy: {boostingRF_test}  Boosting Train Model Accuracy: {boostingRF_train}')

# # print('\n')
# # print('LR Boosting: ')
# # adb = AdaBoostClassifier(LogisticRegression(),n_estimators = 5, learning_rate = 1)
# # adb.fit(X_train,y_train)
# # boostingLR_test = adb.score(X_test,y_test)
# # boostingLR_train = adb.score(X_train,y_train)
# # print(f'Boosting Test Model Accuracy: {boostingLR_test}  Boosting Train Model Accuracy: {boostingLR_train}')

# # print('\n')
# # print('SVM Boosting: ')
# # adb = AdaBoostClassifier(SVC(kernel = 'poly', degree = 2 ), algorithm='SAMME' ,n_estimators = 5, learning_rate = 1)
# # adb.fit(X_train,y_train)
# # boostingSVM_test = adb.score(X_test,y_test)
# # boostingSVM_train = adb.score(X_train,y_train)
# # print(f'Boosting Test Model Accuracy: {boostingSVM_test}  Boosting Train Model Accuracy: {boostingSVM_train}')
