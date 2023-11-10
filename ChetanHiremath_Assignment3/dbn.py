#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 3's dbn.py
#Description of the Program: This program uses the Deep-Belief Network (DBN) to calculate the accuracy value and the epoch values.
#Inputs of the Program: This program has various functions, libraries, and parameters that are used in the program.
#Outputs of the Program: This program prints the accuracy value and the epoch values. 
#Creation Date of the Program: September 18, 2023
#Collaborator/Collaborators of the Program: N/A
#Source/Sources of the Program: Deep Learning Classifiers.pdf (Its background information is applied for this program.) and Deep-Belief-Network's source link- https://github.com/albertbup/deep-belief-network (The code is mostly used from this source.).

import numpy as np #This line imports the modules from Numpy that is known as np. 

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits #This line imports the load_digits function from the sklearn.datasets library.
from sklearn.model_selection import train_test_split #This line imports the train_test_split function from the sklearn.model_selection library.
from sklearn.metrics import accuracy_score #This line imports the accuracy_score function from the sklearn.metrics library.

from dbn import SupervisedDBNClassification #This line imports the SupervisedDBNClassification function from the dbn library.


# from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

# Loading dataset
digits = load_digits() #This line sets the digits variable to the load_digits() function of the sklearn.datasets library.
X, Y = digits.data, digits.target #This line contains the features and the names for the Deep-Belief Network Model.

# Data scaling
X = (X / 16).astype(np.float32) #This line converts the X variable's data type into the float type.

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) #This line uses the train_test_split() function for the Deep-Belief Network Model.

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2) #This line sets the classifier variable to the SupervisedDBNClassification() function of the dbn library.
classifier.fit(X_train, Y_train) #This line performs the first fold training for the Deep-Belief Network Model. 

# Save the model
classifier.save('model.pkl') #This line saves the model to a binary file. 

# Restore
classifier = SupervisedDBNClassification.load('model.pkl') #This line loads and restores the binary file.

# Test
Y_pred = classifier.predict(X_test) #This line calculates the predicted classes of the Deep-Belief Network Model.
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred)) #This line sequentially prints the "Done." message, creates a new line, and prints the "Accuracy:" message and the accuracy value of the Deep-Belief Network Model.
