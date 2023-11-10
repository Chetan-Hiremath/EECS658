#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 2's CompareMLModels.py
#Description of the Program: This program performs the the 2-Fold Cross Variation for 7 different ML Models and prints their respective accuracy values and confusion matrices.
#Inputs of the Program: This program has various functions, libraries, and parameters that are used in the program.
#Outputs of the Program: This program prints the overall accuracy values and the confusion matrices of their respective ML Models.
#Creation Date of the Program: September 5, 2023
#Collaborator/Collaborators of the Program: Ayaan Lakhani
#Source/Sources of the Program: Supervised Learning.pdf and Regression Classifiers.pdf (The code is used from these sources.), and the background information of the links that are below this multi-line comment is applied for this program.
"""
Sklearn.preprocessing.LabelEncoder's source link- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
Sklearn.preprocessing.PolynomialFeatures's source link- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
Sklearn.linear_model.LinearRegression's source link- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
Cross-validation: evaluating estimator performance's source link- https://scikit-learn.org/stable/modules/cross_validation.html
Sklearn.neighbors.KNeighborsClassifier's source link- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
Sklearn.discriminant_analysis.LinearDiscriminantAnalysis's source link- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
Sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis's source link- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
"""

# Load Libraries
"""
These 12 lines that are below this multi-line comment import the modules from Numpy, Pandas, and Scikit-learn since these modules are mandatory for the calculations of this program.
"""
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Load Dataset
url = "iris.csv" #This line sets the url variable to the iris.csv file.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #This line contains the array/list of features of the dataset.
dataset = read_csv(url, names = names) #This line uses the read_csv() function to access data from the iris.csv file and store that data in this variable.

# Create Arrays for Features and Classes
array = dataset.values #This line contains the iris.csv file's values that are stored in the array variable.
X = array[:,0:4] #This line contains the flower features/inputs of the Iris Varieties. 
y = array[:,4] #This line contains the flower names/outputs of the Iris Varieties.

#Encode for Each Class
#Use Encoded Training and Validation Values For Prediction on Linear Regression
encoder = LabelEncoder() #This line sets the encoder variable to the LabelEncoder() function of the sklearn.preprocessing library.
encoder.fit(y) #This line fits the Label Encoder for the encoder variable. 

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, encoder.transform(y), test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties and uses these classes for various ML Models.

#LINEAR REGRESSION CODE
model1 = LinearRegression() #This line sets the model1 variable to the LinearRegression() function of the sklearn.linear_model library.
model1.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model1.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
pred1 = pred1.round() #This line will round the values of the pred1 variable to the nearest integers.
model1.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model1.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
pred2 = pred2.round() #This line will round the values of the pred2 variable to the nearest integers.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Linear Regression Model.
print('LINEAR REGRESSION:') #This line prints the "LINEAR REGRESSION:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Linear Regression Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted), '\n') #This line prints the confusion matrix of the Linear Regression Model and makes a new line.

#DEGREE 2 POLYNOMIAL REGRESSION CODE
poly_reg = PolynomialFeatures(degree = 2) #This line sets the poly_reg variable to the PolynomialFeatures() function of the sklearn.preprocessing library.
X_Poly1 = poly_reg.fit_transform(X_Fold1) #This line fits the data for the X_Poly1 variable and transforms it into an array of values.
X_Poly2 = poly_reg.fit_transform(X_Fold2) #This line fits the data for the X_Poly2 variable and transforms it into an array of values.
y_Poly1 = poly_reg.fit_transform(X_Fold2) #This line fits the data for the y_Poly1 variable and transforms it into an array of values.
y_Poly2 = poly_reg.fit_transform(X_Fold1) #This line fits the data for the y_Poly2 variable and transforms it into an array of values.
model2 = LinearRegression() #This line sets the model2 variable to the LinearRegression() function of the sklearn.linear_model library.
model2.fit(X_Poly1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model2.predict(y_Poly1) #This line performs the first fold testing of the 2-Fold Cross Variation.
pred1 = pred1.round() #This line will round the values of the pred1 variable to the nearest integers.
model2.fit(X_Poly2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model2.predict(y_Poly2) #This line performs the second fold testing of the 2-Fold Cross Variation.
pred2 = pred2.round() #This line will round the values of the pred2 variable to the nearest integers.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Degree 2 Polynomial Regression Model.
"""
These 5 lines that are below this multi-line comment show a for-loop of a range from 0 to 150 since there are 150 samples in the iris.csv file, and the for-loop checks if the values are in the range.
"""
for i in range(0, 150):
    if (predicted[i] > 2):
        predicted[i] = 2
    elif (predicted[i] < 0):
        predicted[i] = 0
predicted = np.around(predicted) #This line rounds the values of the predicted classes' array for the predicted variable that will be used for the calculations.
print('DEGREE 2 POLYNOMIAL REGRESSION:') #This line prints the "DEGREE 2 POLYNOMIAL REGRESSION:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Degree 2 Polynomial Regression Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted), '\n') #This line prints the confusion matrix of the Degree 2 Polynomial Regression Model and makes a new line.

#DEGREE 3 POLYNOMIAL REGRESSION CODE
poly_reg = PolynomialFeatures(degree = 3) #This line sets the poly_reg variable to the PolynomialFeatures() function of the sklearn.preprocessing library.
X_Poly1 = poly_reg.fit_transform(X_Fold1) #This line fits the data for the X_Poly1 variable and transforms it into an array of values.
X_Poly2 = poly_reg.fit_transform(X_Fold2) #This line fits the data for the X_Poly2 variable and transforms it into an array of values.
y_Poly1 = poly_reg.fit_transform(X_Fold2) #This line fits the data for the y_Poly1 variable and transforms it into an array of values.
y_Poly2 = poly_reg.fit_transform(X_Fold1) #This line fits the data for the y_Poly2 variable and transforms it into an array of values.
model3 = LinearRegression() #This line sets the model3 variable to the LinearRegression() function of the sklearn.linear_model library.
model3.fit(X_Poly1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model3.predict(y_Poly1) #This line performs the first fold testing of the 2-Fold Cross Variation.
pred1 = pred1.round() #This line will round the values of the pred1 variable to the nearest integers.
model3.fit(X_Poly2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model3.predict(y_Poly2) #This line performs the second fold testing of the 2-Fold Cross Variation.
pred2 = pred2.round() #This line will round the values of the pred2 variable to the nearest integers.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Degree 3 Polynomial Regression Model.
"""
These 5 lines that are below this multi-line comment show a for-loop of a range from 0 to 150 since there are 150 samples in the iris.csv file, and the for-loop checks if the values are in the range.
"""
for i in range(0, 150):
    if (predicted[i] > 2):
        predicted[i] = 2
    elif (predicted[i] < 0):
        predicted[i] = 0
predicted = np.around(predicted) #This line rounds the values of the predicted classes' array for the predicted variable that will be used for the calculations.
print('DEGREE 3 POLYNOMIAL REGRESSION:') #This line prints the "DEGREE 3 POLYNOMIAL REGRESSION:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Degree 3 Polynomial Regression Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted), '\n') #This line prints the confusion matrix of the Degree 3 Polynomial Regression Model and makes a new line.

#NAIVE-BAYESIAN CODE
model4 = GaussianNB() #This line sets the model4 variable to the GaussianNB() function of the sklearn.naive_bayes library.
model4.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model4.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model4.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model4.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Naive-Bayesian Model.
print('NAIVE-BAYESIAN:') #This line prints the "NAIVE-BAYESIAN:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Naive-Bayesian Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted), '\n') #This line prints the confusion matrix of the Naive-Bayesian Model and makes a new line.

#K-NEAREST NEIGHBORS CODE
model5 = KNeighborsClassifier() #This line sets the model5 variable to the KNeighborsClassifier() function of the sklearn.neighbors library.
model5.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model5.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model5.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model5.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the K-Nearest Neighbors Model.
print('K-NEAREST NEIGHBORS:') #This line prints the "K-NEAREST NEIGHBORS:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the K-Nearest Neighbors Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted), '\n') #This line prints the confusion matrix of the K-Nearest Neighbors Model and makes a new line.

#LINEAR DISCRIMINANT ANALYSIS CODE
model6 = LinearDiscriminantAnalysis() #This line sets the model6 variable to the LinearDiscriminantAnalysis() function of the sklearn.discriminant_analysis library.
model6.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model6.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model6.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model6.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Linear Discriminant Analysis Model.
print('LINEAR DISCRIMINANT ANALYSIS:') #This line prints the "LINEAR DISCRIMINANT ANALYSIS:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Linear Discriminant Analysis Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted), '\n') #This line prints the confusion matrix of the Linear Discriminant Analysis Model and makes a new line.

#QUADRATIC DISCRIMINANT ANALYSIS CODE
model7 = QuadraticDiscriminantAnalysis() #This line sets the model7 variable to the QuadraticDiscriminantAnalysis() function of the sklearn.discriminant_analysis library.
model7.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model7.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model7.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model7.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Quadratic Discriminant Analysis Model.
print('QUADRATIC DISCRIMINANT ANALYSIS:') #This line prints the "QUADRATIC DISCRIMINANT ANALYSIS:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Quadratic Discriminant Analysis Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted)) #This line prints the confusion matrix of the Quadratic Discriminant Analysis Model.
