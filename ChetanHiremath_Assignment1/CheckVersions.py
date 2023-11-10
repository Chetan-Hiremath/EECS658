#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 1's CheckVersions.py
#Description of the Program: This program checks if Python and its libraries are downloaded properly and prints their current versions and the "Hello World!" message.
#Inputs of the Program: This program uses the format function for the library version. 
#Outputs of the Program: This program prints the current versions of the Python libraries and the "Hello World!" message. 
#Creation Date of the Program: August 22, 2023
#Collaborator/Collaborators of the Program: N/A
#Source/Sources of the Program: Assignment 1 Insturctions.pdf (The code is mostly used from this source.)

#Part 1
# Python version
import sys #This line imports the modules from Python.
print('Python: {}'.format(sys.version)) #This line prints the current version of Python.
# scipy
import scipy #This line imports the modules from Scipy.
print('scipy: {}'.format(scipy.__version__)) #This line prints the current version of Scipy.
# numpy
import numpy #This line imports the modules from Numpy.
print('numpy: {}'.format(numpy.__version__)) #This line prints the current version of Numpy.
# pandas
import pandas #This line imports the modules from Pandas.
print('pandas: {}'.format(pandas.__version__)) #This line prints the current version of Pandas.
# scikit-learn
import sklearn #This line imports the modules from Scikit-learn.
print('sklearn: {}'.format(sklearn.__version__)) #This line prints the current version of Scikit-learn.

#Part 2
print('Hello World!') #This line prints the "Hello World!" message.
