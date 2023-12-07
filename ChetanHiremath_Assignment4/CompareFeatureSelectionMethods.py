#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 4's CompareFeatureSelectionMethods.py
#Description of the Program: This program uses the Decision Tree ML Model that is applied to different dimensionality reduction methods on the Iris data-set.
#Inputs of the Program: This program has various functions, libraries, and parameters that are used in the program.
#Outputs of the Program: This program prints the respective and overall accuracy values, the respective confusion matrices, the respective features' lists, and other respective results of Decision Tree ML Model, PCA Feature Transformation, Simulated Annealing Feature Selection, and Genetic Algorithm Feature Selection.
#Creation Date of the Program: October 1, 2023
#Collaborator/Collaborators of the Program: Ayaan Lakhani
#Source/Sources of the Program: Supervised Learning.pdf, SVM and DT Classifiers.pdf, PCA Feature Transformation.pdf, Simulated Annealing Feature Selection.pdf, and Genetic Algorithm Feature Selection.pdf (The code is used from these sources.). The background information of the links that are below this multi-line comment is applied for this program.
"""
Python Random Module's source link- https://www.geeksforgeeks.org/python-random-module/
Python Math's source link- https://www.w3schools.com/python/python_math.asp
Numpy.cov's source link- https://numpy.org/doc/stable/reference/generated/numpy.cov.html
Numpy.where's source link- https://numpy.org/doc/stable/reference/generated/numpy.where.html 
Numpy.atleast_1d's source link- https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html
Linear algebra (numpy.linalg)'s source link- https://numpy.org/doc/stable/reference/routines.linalg.html
Sklearn.tree.DecisionTreeClassifier's source link- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

# Load Libraries
"""
These 10 lines that are below this multi-line comment import the modules from Numpy, Pandas, Scikit-learn, and other libraries since these modules are mandatory for the calculations of this program.
"""
import numpy as np
import random
import math
from numpy import array, mean, cov
from numpy.linalg import eig
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
"""
These 7 lines that are below this multi-line comment define the feature_names() function that prints the feature names of the set.
"""
def feature_names(index_arr):
    ret_arr = np.array([])
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'z1', 'z2', 'z3', 'z4']
    index_arr = np.sort(index_arr)
    for index in index_arr:
        ret_arr = np.append(ret_arr, names[int(index)])
    return ret_arr
"""
These 7 lines that are below this multi-line comment define the get_removed() function that removes a set.
"""
def get_removed(set):
    all = np.array([0,1,2,3,4,5,6,7])
    removed = np.array([])
    for feature in all:
        if feature not in set:
            removed = np.append(removed, feature)
    return removed
"""
These 3 lines that are below this multi-line comment define the get_new() function that creates a new set.
"""
def get_new(set):
    removed = get_removed(set)
    return random.choice(removed)
"""
These 6 lines that are below this multi-line comment define the find_min_index() function that finds the min index of a set.
"""
def find_min_index(set):
    min = 0
    for i in range(len(set)):
        if (set[i][0] < set[min][0]):
            min = i
    return min
"""
These 6 lines that are below this multi-line comment define the find_max_index() function that finds the max index of a set.
"""
def find_max_index(set):
    max = 0
    for i in range(len(set)):
        if (set[i][0] > set[max][0]):
            max = i
    return max
"""
These 7 lines that are below this multi-line comment define the duplicate_exists() function that checks if a duplicate set exists.
"""
def duplicate_exists(sample, set):
    sample = np.sort(sample)
    for i in range(len(set)):
        set[i][1] = np.sort(set[i][1])
        if np.array_equal(set[i][1], sample):
            return True
    return False
    
# Load Dataset
url = "iris.csv" #This line sets the url variable to the iris.csv file.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class', 'class-num'] #This line contains the array/list of features of the dataset.
dataset = read_csv(url, names = names)  #This line uses the read_csv() function to access data from the iris.csv file and store that data in this variable.

# Create Arrays for Features and Classes
array = dataset.values #This line contains the iris.csv file's values that are stored in the array variable.
X = array[:,0:4] #This line contains the flower features/inputs of the Iris Varieties. 
y = array[:,4] #This line contains the flower names/outputs of the Iris Varieties.

# Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

#PART 1: DECISION TREE CODE
model1 = DecisionTreeClassifier() #This line sets the model1 variable to the DecisionTreeClassifier() function of the sklearn.tree library.
model1.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model1.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model1.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model1.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the the Decision Tree Model.
print('DECISION TREE:') #This line prints the "DECISION TREE:" message.
print('Accuracy:', accuracy_score(actual, predicted)) #This line prints the accuracy value of the Decision Tree Model.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual, predicted)) #This line prints the confusion matrix of the the Decision Tree Model. 
print('Features: [\'sepal-length\' \'sepal-width\' \'petal-length\' \'petal-width\']\n') #This line prints the features of the Decision Tree Model and makes a new line.

#PART 2: PCA FEATURE TRANSFORMATION CODE
print('PCA FEATURE TRANSFORMATION:') #This line prints the "PCA FEATURE TRANSFORMATION:" message. 
M = mean(X.T, axis = 1) #This line uses the mean() function to calculate the mean values of the columns.
C = X - M #This line will align the columns to the center portion.
V = cov(C.T.astype(float)) #This line uses the cov() function to calculate the covariance matrix of the centered matrix.
values, vectors = eig(V) #This line sets the values variable and the vectors variable to the eig() function that calculates the eigenvalues and the eigenvectors respectively.
print('Eigenvectors:\n' + str(vectors)) #This line makes a new line and prints the eigenvectors of the matrix
print('Eigenvalues:', values) #This line prints the eigenvalues of the matrix.
print('PoV:', (values[0]/np.sum(values))) #This line prints the PoV value of the matrix.
P = vectors.T.dot(C.T) #This line sets the P variable to the dot() function.
Z = [[i[0]] for i in (P.T)] #This line selects a subset of the features that are transformed from the data to Z if the PoV > 0.9 or 90%.

Z_Fold1, Z_Fold2, y_Fold1, y_Fold2 = train_test_split(Z, y, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
actual2 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

#DECISION TREE CODE OF PCA FEATURE TRANSFORMATION 
model2 = DecisionTreeClassifier() #This line sets the model2 variable to the DecisionTreeClassifier() function of the sklearn.tree library.
model2.fit(Z_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
pred1 = model2.predict(Z_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
model2.fit(Z_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
pred2 = model2.predict(Z_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
predicted2 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the the PCA Feature Transformation.
print('Final Results:') #This line prints the "Final Results:" message.
print('Accuracy:', accuracy_score(actual2, predicted2)) #This line prints the accuracy value of the PCA Feature Transformation.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(actual2, predicted2)) #This line prints the confusion matrix of the PCA Feature Transformation. 
print('Features: [\'z1\' \'z2\' \'z3\' \'z4\']\n') #This line prints the features of the PCA Feature Transformation and makes a new line.
PCA_accuracy = accuracy_score(actual2, predicted2) #This line sets the PCA_accuracy variable to the accuracy_score() function that contains this method's accuracy value. 

#PART 3: SIMULATED ANNEALING CODE
print('SIMULATED ANNEALING:\n') #This line sequentially prints the "SIMULATED ANNEALING:" message and makes a new line.
X_all = np.concatenate((X,(P.T)), axis = 1) #This line sets the X_all variable to the np.concatenate() function that concatenates the original and the transformed features.
X_all_copy = X_all #This line sets the X_all_copy variable to the X_all variable.
X_all_length = len(X) #This line sets the X_all_length variable to the len() function.
removed = np.array([]) #This line sets the removed variable to the np.array() function.
current_set = np.array([0,1,2,3,4,5,6,7]) #This line sets the current_set variable to the np.array() function.
feature_count = len(current_set) #This line sets the feature_count variable to the len() variable.
accepted = np.array([]) #This line sets the accepted variable to the np.array() function.
best_accuracy = PCA_accuracy #This line sets the best_accuracy variable to the PCA_accuracy variable that contains the accuracy value of PCA.
best_set = current_set #This line sets the best_set variable to the current_set variable.
restart_counter = 0 #This line defines and sets the restart_counter variable to 0 since it tracks the set and the accuracy value from the old set.
previous_accuracy = best_accuracy #This line sets the previous_accuracy variable to the best_accuracy variable that already contains the accuracy value of PCA.
"""
These 103 lines that are below this multi-line comment show a very big for-loop of a range from 0 to 99 since there are 100 iterations in this part.
It uses the Simulated Annealing Algorithm from the lecture to calculate and print the features, the accuracy values, the acceptance probabilities, the random uniform values, and the statuses of the respective iterations.
The algorithm's steps and the Decision Tree ML Model are utilized in this algorithm since they are required for crucial calculations.
"""
for i in range(100):
    r1 = random.random()
    num_modified = round((r1 % 2) + 1)
    if (len(removed) == 1):
        num_modified = 1
    r2 = random.random()
    #These 28 lines randomly remove or add features due to the if-else statement.
    if (len(current_set) == feature_count):
        to_remove = random.sample(list(current_set), num_modified)
        removed = np.concatenate([removed, to_remove])
        for element in to_remove:
            if element in current_set:
                index = np.argwhere(current_set == element)
                current_set = np.delete(current_set, index)
    elif (len(current_set) <= num_modified):
        to_add = random.sample(list(removed), num_modified)
        current_set = np.concatenate([current_set, to_add])
        for element in to_add:
            if element in removed:
                index = np.argwhere(removed == element)
                removed = np.delete(removed, index)
    elif (random.choice([0, 1])): 
        to_add = random.sample(list(removed), num_modified)
        current_set = np.concatenate([current_set, to_add])
        for element in to_add:
            if element in removed:
                index = np.argwhere(removed == element)
                removed = np.delete(removed, index)
    else:
        to_remove = random.sample(list(current_set), num_modified)
        removed = np.concatenate([removed, to_remove])
        for element in to_remove:
            if element in current_set:
                index = np.argwhere(current_set == element)
                current_set = np.delete(current_set, index)

    X_test = np.empty([len(X_all),len(current_set)]) #This line defines the X_test variable that is used as a modified feature set.
    removed = np.sort(removed)[::-1] #This line sets the removed variable to the np.sort() function that calculates a sorted copy of the array.
    
    #The for-loop with 5 lines sets the X_test[j] variable to the temp that returns a new array without the sub-arrays.
    for j in range(len(X_all)):
        temp = X_all[j]
        for elem in removed:
            temp = np.delete(temp, int(elem))
        X_test[j] = temp
    
    # Split Data into 2 Folds for Training and Test of the modified feature set
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_test, y, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
    actual3 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.

    #DECISION TREE CODE OF SIMULATED ANNEALING
    model3 = DecisionTreeClassifier() #This line sets the model3 variable to the DecisionTreeClassifier() function of the sklearn.tree library.
    model3.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
    pred1 = model3.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
    model3.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
    pred2 = model3.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
    predicted3 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the the Decision Tree Model.
    current_accuracy = accuracy_score(actual3, predicted3) #This line calculates the current accuracy value of the Decision Tree Model, and this value is used for future calculations.
    current_confusion = confusion_matrix(actual3, predicted3) #This line calculates the current confusion matrix of the the Decision Tree Model, and this matrix is used for future calculations.

    print('Iteration ' + str(i) + ':') #This line keeps on printing the "Iteration" messages and their corresponding iteration values due to the for-loop.
    c = 1 #This line sets the c variable to 1 since c is constant value or 1 of the acceptance probability formula.
    prob_accept = pow(math.e, -((i/c)*((best_accuracy - current_accuracy)/best_accuracy))) #This line defines the prob_accept variable that calculates the acceptance probability.
    rand_uniform = random.uniform(0,1) #This line sets the rand_uniform variable to the random.uniform() function since it calculates the random uniform value. 

    #These 13 lines consider the status as "Improved" if the previous accuracy value <= current accuracy value. The random uniform value and acceptance probability value are empty since the iteration's status is recognized.
    if (previous_accuracy <= current_accuracy): 
        status = 'Improved'
        rand_uniform = '-'
        prob_accept = '-'
        if (best_accuracy <= current_accuracy):
            best_accuracy = current_accuracy
            best_set = current_set
            best_confusion = current_confusion
            restart_counter = 0
        else:
            restart_counter += 1
        previous_accuracy = current_accuracy
        previous_set = current_set
    #These 3 lines consider the status as "Discarded" if the random uniform value > acceptance probability value.
    elif (rand_uniform > prob_accept):
        restart_counter += 1
        status = 'Discarded'
    #These 4 lines consider the status as "Accepted" if the random uniform value <= acceptance probability value.
    else:
        restart_counter += 1
        status = 'Accepted'
        previous_accuracy = current_accuracy
        previous_set = current_set
    #These 8 lines consider the status as "Restart" if the restart counter value is 10.
    if (restart_counter == 10):
        current_accuracy = best_accuracy
        current_set = best_set
        previous_accuracy = current_accuracy
        previous_set = current_accuracy
        removed = get_removed(current_set)
        restart_counter = 0
        status = 'Restart'
    print('Features:      ', feature_names(current_set)) #This line prints the "Features:" message and the subset of features for the corresponding iteration.
    print('Accuracy:      ', current_accuracy) #This line prints the "Accuracy:" message and the accuracy value for the corresponding iteration.
    print('Pr[accept]:    ', prob_accept) #This line prints the "Pr[accept]:" message and the acceptance probability for the corresponding iteration.
    print('Random Uniform:', rand_uniform) #This line prints the "Random Uniform:" message and the random uniform value for the corresponding iteration.
    print('Status:        ', status, '\n') #This line sequentially prints the "Status:" message and the status for the corresponding iteration and makes a new line.
print('Final Results:') #This line line prints the "Final Results:" message.
print('Accuracy:', best_accuracy) #This line prints the accuracy value of the Simulated Annealing.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(best_confusion) #This line prints the confusion matrix of the Simulated Annealing. 
print('Features:', feature_names(best_set)) #This line prints the features of the Simulated Annealing. 

#PART 4: GENETIC ALGORITHM CODE
print('\nGENETIC ALGORITHM:') #This line sequentially makes a new line and prints the "GENETIC ALGORITHM:" message.
n = 5 #This line defines the number of features for an initial population. The features and their respective indices are located after this line.
# Feature       index
# sepal-length    0
# sepal-width     1
# petal-length    2
# petal-width     3
# z1              4
# z2              5
# z3              6
# z4              7
population =[[4,0,1,2,3],[4,5,1,2,3],[4,5,6,1,2],[4,5,6,7,1],[4,5,6,7,0]] #This line defines the population variable that has the 5 sets of features/individuals/candidate solutions for an initial population. 
initial_size = len(population) #This line uses the length function to calculate the number of sets of the population variable.
"""
These 74 lines that are below this multi-line comment show a very big for-loop of a range from 0 to 49 since there are 50 generations in this part.
It uses the Genetic Algorithm from the lecture to calculate and print the features and the accuracy values of the respective generations. 
The algorithm's steps and the Decision Tree ML Model are utilized in this algorithm since they are required for crucial calculations.
"""
for generation in range(50):
    best = [[0,[], None] for _ in range(n)]
    
    # Crossover Step: These 9 lines use the steps of the Crossover Operation of the Genetic Algorithm.
    cross_population = population
    for j in range(initial_size):
        for k in range(j + 1, initial_size):
            union = np.union1d(population[j], population[k])
            intersection = np.intersect1d(population[j], population[k])
            if (len(union) != 0):
                cross_population.append(union)
            if (len(intersection) != 0):
                cross_population.append(intersection)
                
    # Mutation Step: These 15 lines use the steps of the Mutation Operation of the Genetic Algorithm.
    mutuation_population = cross_population.copy()
    for j in range(len(mutuation_population)):
        mutation_choice = random.randrange(3)
        if ((mutation_choice == 0 and len(mutuation_population[j]) != 8) or len(mutuation_population[j]) == 0):
            new = get_new(mutuation_population[j])
            mutuation_population[j] = np.append(mutuation_population[j], new)
        elif ((mutation_choice == 1 and len(mutuation_population[j]) > 1) or len(mutuation_population[j]) == 8):
            remove = np.atleast_1d(mutuation_population[j] == random.choice(mutuation_population[j])).nonzero()
            mutuation_population[j] = np.delete(mutuation_population[j], remove)
        else:
            removed = get_removed(mutuation_population[j])
            mutuation_population[j] = np.delete(mutuation_population[j], np.atleast_1d(mutuation_population[j] == random.choice(mutuation_population[j])).nonzero())
            mutuation_population[j] = np.append(mutuation_population[j], random.choice(removed))
    X_test = ([0] * len(X_all))
    population = mutuation_population + cross_population

    # Evaluation Step: These 8 lines use the steps of the Evaluation Operation of the Genetic Algorithm.
    for i in range(len(population)):
        removed = get_removed(population[i])
        removed = np.sort(removed)[::-1]
        for j in range(len(X_all)):
            temp = X_all[j]
            for elem in removed:
                temp = np.delete(temp, int(elem))
            X_test[j] = temp
            
        # Split Data into 2 Folds for Training and Test of the modified feature set
        X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_test, y, test_size = 0.50, random_state = 1) #This line uses the train_test_split() function for the 2-Fold Cross Variation.
        actual4 = np.concatenate([y_Fold2, y_Fold1]) #This line calculates the actual classes of the Iris Varieties.
        
        #DECISION TREE CODE OF GENETIC ALGORITHM
        model4 = DecisionTreeClassifier() #This line sets the model4 variable to the DecisionTreeClassifier() function of the sklearn.tree library.
        model4.fit(X_Fold1, y_Fold1) #This line performs the first fold training of the 2-Fold Cross Variation.
        pred1 = model4.predict(X_Fold2) #This line performs the first fold testing of the 2-Fold Cross Variation.
        model4.fit(X_Fold2, y_Fold2) #This line performs the second fold training of the 2-Fold Cross Variation.
        pred2 = model4.predict(X_Fold1) #This line performs the second fold testing of the 2-Fold Cross Variation.
        predicted4 = np.concatenate([pred1, pred2]) #This line calculates the predicted classes of the the Decision Tree Model. 
        current_accuracy = accuracy_score(actual4, predicted4) #This line calculates the current accuracy value of the Decision Tree Model, and this value is used for future calculations.
      
        min_index = find_min_index(best) #This line defines the min_index variable that uses the find_min_index function to find the min index. 
        #The if-statement with 4 lines shows that it will calculate the current accuracy value, the population, and the confusion matrix by 2 conditions: 
        #Condition 1: The min index is less than the current accuracy value, and Condition 2: There is no duplicate set for the population.
        if ((best[min_index][0] < current_accuracy) and (not duplicate_exists(population[i], best))):
              best[min_index][0] = current_accuracy
              best[min_index][1] = population[i]
              best[min_index][2] = confusion_matrix(actual4, predicted4)
    print('\nGeneration ' + str(generation) + ':') #This line sequentially keeps on making new lines and printing the "Generation" messages and the corresponding generation indices due to the for-loop.
    sorted_best = sorted(best, key = lambda x: x[0], reverse = True) #This line defines the sorted_best variable that sorts the best set of features.
    #The for-loop with 4 lines prints the features and the accuracy values for the 5 best sets of the features.
    for i in range(len(sorted_best)):
        print(str(i + 1) + '.') #This line keeps on printing the values of the best sets of features due to the for-loop.
        print('Features: ', feature_names(sorted_best[i][1])) #This line keeps on printing the "Features:" messages and the corresponding features of the best sets due to the for-loop.
        print('Accuracy: ', sorted_best[i][0]) #This line keeps on printing the "Accuracy:" messages and the corresponding accuracy values of the best sets due to the for-loop.

    # Selection Step: These 4 lines use the steps of the Selection Operation of the Genetic Algorithm.
    new_population = [[[]] for _ in range(n)]
    for j in range(len(best)):
        new_population[j] = best[j][1]
    population = new_population
max_index = find_max_index(best) #This line defines the max_index variable that uses the find_max_index function to find the max index. 
print('\nFinal Results:') #This line sequentially makes a new line and prints the "Final Results:" message.
print('Accuracy:', (best[max_index][0])) #This line prints the accuracy value of the Genetic Algorithm.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(best[max_index][2]) #This line prints the confusion matrix of the Genetic Algorithm. 
print('Features:', feature_names(best[max_index][1])) #This line prints the features of the Genetic Algorithm. 
