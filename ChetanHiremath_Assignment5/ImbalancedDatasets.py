#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 5's ImbalancedDatasets.py
#Description of the Program: This program uses the Neural Network ML Model that is applied to different methods of imbalanced data sets on the Iris data-set.
#Inputs of the Program: This program has various functions, libraries, and parameters that are used in the program.
#Outputs of the Program: This program prints the accuracy value, the confusion matrix, the class balanced accuracy value, the balanced accuracy value, and Skikit-learn's balanced accuracy value of the method of imbalanaced data set for Part 1. Also, this program prints the overall accuracy values and the confusion matrices of their respective methods of imbalanced data sets for Parts 2 and 3.
#Creation Date of the Program: October 16, 2023
#Collaborator/Collaborators of the Program: Ayaan Lakhani
#Source/Sources of the Program: Supervised Learning.pdf, Neural Networks.pdf, and Imbalanced Datasets.pdf (The code is used from these sources.). The background information of the links that are below this multi-line comment is applied for this program.
"""
Pandas.read_csv's source link- https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
Sklearn.metrics.precision_score's source link- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
Sklearn.metrics.recall_score's source link- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
Sklearn.metrics.balanced_accuracy_score's source link- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
Sklearn.neural_network.MLPClassifier's source link- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
Sklearn.cluster.MiniBatchKMeans' source link- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
Specificity_score's source link- https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.specificity_score.html
RandomOverSampler's source link- https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
SMOTE's source link- https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
ADASYN's source link- https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html
RandomUnderSampler's source link- https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
ClusterCentroids' source link- https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html
TomekLinks' source link- https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html
"""

# Load Libraries
"""
These 17 lines that are below this multi-line comment import the modules from Numpy, Pandas, Scikit-learn, and Imbalanced-learn since these modules are mandatory for the calculations of this program.
"""
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import MiniBatchKMeans
from imblearn.metrics import specificity_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks

# Load Dataset
url = "imbalanced iris.csv" #This line sets the url variable to the imbalanced iris.csv file.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #This line contains the array/list of features of the dataset.
dataset = read_csv(url, skiprows = 1, names = names) #This line uses the read_csv() function to access data from the imbalanced iris.csv file and store that data in this variable. The skiprows' parameter is added in this function for the first time since it skips the imbalanced iris.csv file's first row that contains non-numerical strings only and considers the remaining rows that contain the numerical strings. Then, the numerical strings of those rows are converted into float numbers for the data that will be used for the calculations of this program. 

# Create Arrays for Features and Classes
array = dataset.values #This line contains the imbalanced iris.csv file's values that are stored in the array variable. 
X = array[:,0:4] #This line contains the flower features/inputs of the Iris Varieties.
y = array[:,4] #This line contains the flower names/outputs of the Iris Varieties.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

#PART 1: IMBALANCED DATA SET CODE
print('IMBALANCED DATA SET:') #This line prints the "IMBALANCED DATA SET:" message.
model1 = MLPClassifier(max_iter = 1000) #This line sets the model1 variable to the MLPClassifier() function of the sklearn.neural_network library.
model1.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model1.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model1.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model1.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Imbalanced Data Set.
print('NEURAL NETWORK - Imbalanced Data Set') #This line prints the "NEURAL NETWORK - Imbalanced Data Set" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Imbalanced Data Set.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted)) #This line prints the confusion matrix of the Imbalanced Data Set.
precision = precision_score(actual, predicted, average = None) #This line calculates the precision value of the Imbalanced Data Set.
recall = recall_score(actual, predicted, average = None) #This line calculates the recall value of the Imbalanced Data Set.
min_pr = ([None] * len(precision)) #This line defines the min_pr value that creates an array with 3 empty values to calculate the minimum values of precision and recall values.
#The for-loop with 2 lines finds and calculates the minimum values of precision and recall values for the Iris Varieties. 
for i in range(len(precision)):
    min_pr[i] = min(precision[i], recall[i])
cba_average = (sum(min_pr)/len(min_pr)) #This line defines the cba_average variable that calculates the average of the minimum values.
print('Class Balanced Accuracy:', cba_average) #This line prints the class balanced accuracy value of the Imbalanced Data Set.
specificity = specificity_score(actual, predicted, average = None) #This line calculates the specificity value of the Imbalanced Data Set.
average_rs = ([None] * len(recall)) #This line defines the average_rs value that creates an array with 3 empty values to calculate the average values of recall and specificity values.
#The for-loop with 2 lines finds and calculates the average values of recall and specificity values for the Iris Varieties.
for i in range(len(recall)):
    average_rs[i] = ((recall[i] + specificity[i])/2)
ba_average = (sum(average_rs)/len(average_rs)) #This line defines the ba_average variable that calculates the average of the average values.
print('Balanced Accuracy:', ba_average) #This line prints the balanced accuracy value of the Imbalanced Data Set.
print('Scikit-learn Balanced Accuracy:', balanced_accuracy_score(actual, predicted), '\n') #This line uses the balanced_accuracy_score() function from the sklearn.metrics library, calculates the macro-average of the recall values, prints the balanced accuracy value of the Imbalanced Data Set, and makes a new line.

#PART 2: OVERSAMPLING CODE
print('OVERSAMPLING:') #This line prints the "OVERSAMPLING:" message.
ros = RandomOverSampler(random_state = 0) #This line sets the ros variable to the RandomOverSampler() function of the imblearn.over_sampling library.
X_res, y_res = ros.fit_resample(X, y) #This line sets the X_res variable and the y_res variable to the fit_resample() function that resamples the sets of X and y respectively.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual2 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

model2 = MLPClassifier(max_iter = 1000) #This line sets the model2 variable to the MLPClassifier() function of the sklearn.neural_network library.
model2.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model2.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model2.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model2.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted2 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Random Oversampling.
print('NEURAL NETWORK - Random Oversampling') #This line prints the "NEURAL NETWORK - Random Oversampling" message.
print('Accuracy:', accuracy_score(actual2, predicted2)) #This line prints the accuracy value of the Random Oversampling.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual2, predicted2), '\n') #This line prints the confusion matrix of the Random Oversampling and makes a new line.

#SMOTE CODE
sm = SMOTE() #This line sets the sm variable to the SMOTE() function of the imblearn.over_sampling library.
X_res, y_res = sm.fit_resample(X, y) #This line sets the X_res variable and the y_res variable to the fit_resample() function that resamples the sets of X and y respectively.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual3 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

model3 = MLPClassifier(max_iter = 1000) #This line sets the model3 variable to the MLPClassifier() function of the sklearn.neural_network library.
model3.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model3.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model3.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model3.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted3 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the SMOTE Oversampling.
print('NEURAL NETWORK - SMOTE Oversampling') #This line prints the "NEURAL NETWORK - SMOTE Oversampling" message.
print('Accuracy:', accuracy_score(actual3, predicted3)) #This line prints the accuracy value of the SMOTE Oversampling.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual3, predicted3), '\n') #This line prints the confusion matrix of the SMOTE Oversampling and makes a new line.

#ADASYN CODE
ada = ADASYN(random_state = 0, sampling_strategy = 'minority') #This line sets the ada variable to the ADASYN() function of the imblearn.over_sampling library.
X_res, y_res = ada.fit_resample(X, y) #This line sets the X_res variable and the y_res variable to the fit_resample() function that resamples the sets of X and y respectively.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual4 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

model4 = MLPClassifier(max_iter = 1000) #This line sets the model4 variable to the MLPClassifier() function of the sklearn.neural_network library.
model4.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model4.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model4.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model4.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted4 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the ADASYN Oversampling.
print('NEURAL NETWORK - ADASYN Oversampling')  #This line prints the "NEURAL NETWORK - ADASYN Oversampling" message.
print('Accuracy:', accuracy_score(actual4, predicted4)) #This line prints the accuracy value of the ADASYN Oversampling.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual4, predicted4), '\n') #This line prints the confusion matrix of the ADASYN Oversampling and makes a new line.

#PART 3: UNDERSAMPLING CODE
print('UNDERSAMPLING:') #This line prints the "UNDERSAMPLING:" message.
rus = RandomUnderSampler(random_state = 1) #This line sets the rus variable to the RandomUnderSampler() function of the imblearn.under_sampling library.
X_res, y_res = rus.fit_resample(X, y) #This line sets the X_res variable and the y_res variable to the fit_resample() function that resamples the sets of X and y respectively.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual5 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

model5 = MLPClassifier(max_iter = 1000) #This line sets the model5 variable to the MLPClassifier() function of the sklearn.neural_network library.
model5.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model5.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model5.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model5.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted5 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Random Undersampling.
print('NEURAL NETWORK - Random Undersampling') #This line prints the "NEURAL NETWORK - Random Undersampling" message.
print('Accuracy:', accuracy_score(actual5, predicted5)) #This line prints the accuracy value of the Random Undersampling.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual5, predicted5), '\n') #This line prints the confusion matrix of the Random Undersampling and makes a new line.

#CLUSTER CODE
cc = ClusterCentroids(estimator = MiniBatchKMeans(n_init = 1, random_state = 0)) #This line sets the cc variable to the ClusterCentroids() function of the imblearn.under_sampling library, and the ClusterCentroids() function contains the MiniBatchKMeans() function of the sklearn.cluster library.
X_res, y_res = cc.fit_resample(X, y) #This line sets the X_res variable and the y_res variable to the fit_resample() function that resamples the sets of X and y respectively.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual6 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

model6= MLPClassifier(max_iter = 1000) #This line sets the model6 variable to the MLPClassifier() function of the sklearn.neural_network library.
model6.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model6.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model6.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model6.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted6 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Cluster Undersampling.
print('NEURAL NETWORK - Cluster Undersampling') #This line prints the "NEURAL NETWORK - Cluster Undersampling" message.
print('Accuracy:', accuracy_score(actual6, predicted6)) #This line prints the accuracy value of the Cluster Undersampling.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual6, predicted6), '\n') #This line prints the confusion matrix of the Cluster Undersampling and makes a new line.

#TOMEK LINKS CODE
tl = TomekLinks() #This line sets the tl variable to the TomekLinks() function of the imblearn.under_sampling library.
X_res, y_res = tl.fit_resample(X, y) #This line sets the X_res variable and the y_res variable to the fit_resample() function that resamples the sets of X and y respectively.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual7 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

model7 = MLPClassifier(max_iter = 1000) #This line sets the model7 variable to the MLPClassifier() function of the sklearn.neural_network library.
model7.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model7.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model7.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model7.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted7 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Tomek Links Undersampling.
print('NEURAL NETWORK - Tomek Links Undersampling') #This line prints the "NEURAL NETWORK - Tomek Links Undersampling" message.
print('Accuracy:', accuracy_score(actual7, predicted7)) #This line prints the accuracy value of the Tomek Links Undersampling.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual7, predicted7)) #This line prints the confusion matrix of the Tomek Links Undersampling.
