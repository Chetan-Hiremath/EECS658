#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 1's NBClassifier.py
#Description of the Program: This program performs the 2-Fold Cross Variation and prints the confusion matrix, the accuracy value, the precision values, the recall values, F1 values, and other values of the Iris Varieties by the iris.csv file and the Python libraries.
#Inputs of the Program: This program has functions and their parameters that are used in the program.
#Outputs of the Program: This program prints the overall accuracy value, the confusion matrix, the precision values, the recall values, the F1 values, and other values of the Iris Varieties. 
#Creation Date of the Program: August 24, 2023
#Collaborator/Collaborators of the Program: Ayaan Lakhani
#Source/Sources of the Program: Supervised Learning.pdf (The code is mostly used from this source.) and Cross-validation: evaluating estimator performance's source link- https://scikit-learn.org/stable/modules/cross_validation.html (Its background information is applied for this program.)

# Load Libraries
"""
These 7 lines that are below this multi-line comment import the modules from Numpy, Pandas, and Scikit-learn since these modules are mandatory for the calculations of this program.
"""
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

#Load Dataset
url = "iris.csv" #This line sets the url variable to the iris.csv file.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #This line contains the array/list of features of the dataset.
dataset = read_csv(url, names = names) #This line uses the read_csv() function to access data from the iris.csv file and store that data in this variable.

#Create Arrays for Features and Classes
array = dataset.values #This line contains the iris.csv file's values that are stored in the array variable.
X = array[:,0:4] #This line contains the flower features/inputs of the Iris Varieties.
y = array[:,4] #This line contains the flower names/outputs of the Iris Varieties.
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.

model = GaussianNB() #This line sets the model variable to the GaussianNB() function of the sklearn.naive_bayes library.
model.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.

actual = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the Iris Varieties.
print("Accuracy:", accuracy_score(actual, predicted)) #This line prints the accuracy value of the Iris Varieties.
print("Confusion Matrix:") #This line prints the "Confusion Matrix:" message. 
print(confusion_matrix(actual, predicted)) #This line prints the confusion matrix of the Iris Varieties.
print("Final Calculated Results:\n", classification_report(actual, predicted)) #This line uses the confusion matrix and prints the table that has the Precision, the Recall, the F1, and other values of the Iris Varieties.
