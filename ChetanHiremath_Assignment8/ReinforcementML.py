#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 8's ReinforcementML.py
#Description of the Program: This program uses the Reinforcement Learning's Monte Carlo First Visit, Monte Carlo Every Visit, Q-Learning, SARSA, and Decaying Epsilon-Greedy Algorithms to generate the respective iterations' policy arrays/matrices/grids and graphs.
#Inputs of the Program: This program has various functions, libraries, and parameters that are used in the program.
#Outputs of the Program: This program prints the Monte Carlo First Visit, Monte Carlo Every Visit, Q-Learning, SARSA, and Decaying Epsilon-Greedy Algorithm's policy arrays/matrices/grids and plots the graphs. Also, it plots the graph of the Cumulative Average Rewards of Q-Learning, SARSA, and Decaying Epsilon-Greedy Algorithms for their comparison.
#Creation Date of the Program: November 26, 2023
#Collaborator/Collaborators of the Program: Ayaan Lakhani
#Source/Sources of the Program: RL Monte Carlo.pdf, RL Q-Learning.pdf, RL SARSA & Epsilon-Greedy Algorithms.pdf, and PlottingCode.py (The code is used from these sources.). The background information of the links that are below this multi-line comment is applied for this program.
"""
Numpy.array's source link- https://numpy.org/doc/stable/reference/generated/numpy.array.html
Numpy.add's source link- https://numpy.org/doc/stable/reference/generated/numpy.add.html
Numpy.zeros' source link- https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
Numpy.copy's source link- https://numpy.org/doc/stable/reference/generated/numpy.copy.html
Numpy.vstack's source link- https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
Numpy.array_equal's source link- https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
Numpy.around's source link- https://numpy.org/doc/stable/reference/generated/numpy.around.html
Python all()'s source link- https://www.programiz.com/python-programming/methods/built-in/all
Numpy.all's source link- https://numpy.org/doc/stable/reference/generated/numpy.all.html
Numpy.full's source link- https://numpy.org/doc/stable/reference/generated/numpy.full.html
Numpy.append's source link- https://numpy.org/doc/stable/reference/generated/numpy.append.html
Numpy.delete's source link- https://numpy.org/doc/stable/reference/generated/numpy.delete.html
Python math library|exp() method's source link- https://www.geeksforgeeks.org/python-math-library-exp-method/
Uniform() method in Python Random module's source link- https://www.geeksforgeeks.org/python-number-uniform-method/
Numpy.matrix.flatten's source link- https://numpy.org/doc/stable/reference/generated/numpy.matrix.flatten.html
Python end parameter in print()'s source link- https://www.geeksforgeeks.org/gfact-50-python-end-parameter-in-print/
Numpy.divide's source link- https://numpy.org/doc/stable/reference/generated/numpy.divide.html
Numpy.cumsum's source link- https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
Numpy.arange's source link- https://numpy.org/doc/stable/reference/generated/numpy.arange.html
"""

# Load Libraries
import numpy as np #This line imports the module from Numpy since this module is mandatory for the calculations of this program.
import random #This line imports the module from random since this module is mandatory for the calculations of this program.
import math #This line imports the module from math since this module is mandatory for the calculations of this program.
from matplotlib import pyplot as plt #This line imports the module from Matplotlib since this module is used to plot the graph of data points.

"""
These 10 lines that are below this multi-line comment define the actionReward() function that returns the final position and the reward of Monte Carlo First Visit and Every Visit.
"""
def actionReward(initialPosition, action):
    reward = -1 #This line sets the reward variable to -1 that defines the value of the reward.
    finalPosition = np.add(initialPosition, action) #This line defines the finalPosition variable that calculates the sum of the initial position array and the action array.
    #The if-statement with 2 lines returns the initial position to exit the policy grid, so it will be the final position.
    if -1 in finalPosition or 5 in finalPosition: 
        finalPosition = initialPosition #This line sets the finalPosition variable to the initialPosition variable since the final position is considered as the initial position.
    #The if-statement with 2 lines sets the reward value to 0 if the state is one of the termination states.
    if any(np.array_equal(x, finalPosition) for x in terminationStates):
        reward = 0 #This line sets the reward variable to 0 that defines the value of the reward.
    return (finalPosition, reward) #This line returns the final position and its reward value.

"""
These 33 lines that are below this multi-line comment define the plot_graph() function that plots and shows a graph of an algorithm. This function is borrowed from PlottingCode.py file.
"""    
def plot_graph(arr, name):
  if (name == "Error Value FV"):
    plt.plot(range(1, 26), arr, marker='o')
    plt.title(name + ' vs. t')
    plt.xlabel('t')
    plt.xticks(np.arange(1, 26, 1))
    plt.ylabel('V(s) Values')
    plt.show()
  elif (name == "Error Value EV"):
    plt.plot(range(1, 26), arr, marker='o')
    plt.title(name + ' vs. t')
    plt.xlabel('t')
    plt.xticks(np.arange(1, 26, 1))
    plt.ylabel('V(s) Values')
    plt.show()
  elif (name == "Error Value QL"):
    plt.plot(range(1, 626), arr, marker='o')
    plt.title(name + ' vs. t')
    plt.xlabel('t')
    plt.ylabel('Q-Matrix Values')
    plt.show() 
  elif (name == "Error Value SR"):
    plt.plot(range(1, 626), arr, marker='o')
    plt.title(name + ' vs. t')
    plt.xlabel('t')
    plt.ylabel('Q-Matrix Values')
    plt.show()
  elif (name == "Error Value EG"):
    plt.plot(range(1, 626), arr, marker='o')
    plt.title(name + ' vs. t')
    plt.xlabel('t')
    plt.ylabel('Q-Matrix Values')
    plt.show()

"""
These 9 lines that are below this multi-line comment define the plot_cum_avg_reward() function that plots, shows a graph of the cumulative average rewards of Q-Learning, SARSA, and Decaying Epsilon-Greedy Algorithms respectively. This function is partially borrowed from PlottingCode.py file.
"""
def plot_cum_avg_reward(arr1, arr2, arr3, name):
    plt.plot(range(1, 626), arr1, marker='o', label = 'Q-Learning') 
    plt.plot(range(1, 626), arr2, marker='o', label = 'SARSA')
    plt.plot(range(1, 626), arr3, marker='o', label = 'Decaying Epsilon-Greedy')
    plt.legend()
    plt.title(name)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Average Reward')
    plt.show()

"""
These 13 lines that are below this multi-line comment define the print_matrix() function that creates and prints a matrix of Q-Learning, SARSA, and Decaying Epsilon-Greedy.
"""
def print_matrix(array):
    print("S\A", end="\t") #This line prints the "S/A" message and combines this message and a new tab with the end parameter.
    #The for-loop with 2 lines prints the indices of the matrix.
    for i in range(len(array)):
        print(i, end="\t") #This line keeps on printing the indices and combining new tabs for the action row due to the for-loop.
    print("") #This line prints "" message or a new line.
    #The for-loop with 6 lines prints the values of a matrix.
    for i in range(len(array)):
        print(i, end="\t") #This line keeps on printing the indices and combining new tabs for the state column due to the for-loop.
        #The for-loop with 2 lines prints the values.
        for j in range(len(array[0])):
            print(array[i][j], end="\t") #This line keeps on printing the values and combining new tabs due to the for-loop.
        print("") #This line prints "" message or a new line.

"""
These 9 lines that are below this multi-line comment define the check_ConvergenceQL() function that checks the convergence of Q-Learning (Off-Policy).
"""
def checkConvergenceQL(copyQQL, QQL):
    #The for-loop with 6 lines checks if 1 state has a value of the Q-Matrix.
    for i in range(1,24):
        #The for-loop with 4 lines checks if the convergence is successful.
        for j in range(0,25):
            #The if-statement with 2 lines checks if the state that has have recorded value is the next state.
            if ((RQL[i][j] != -1) and (QQL[i][j] == 0)):
                return False #This line returns False if the actions are evaluated.
    return (QQL == copyQQL).all() #This line checks the iterable elements of the current Q-Matrix and the previous Q-Matrix and returns True if the iterable elements are true.

"""
These 12 lines that are below this multi-line comment define the check_ConvergenceSR() function that checks the convergence of SARSA (On-Policy).
"""
def checkConvergenceSR(copyQSR, QSR):
    #The for-loop with 9 lines checks if 1 state has a value of the Q-Matrix.
    for i in range(1, 24):
        #The for-loop with 7 lines checks if the convergence is successful.
        for j in range(0, 25):
            #The if-statement with 2 lines checks if the state that has have recorded value is the next state.
            if ((RSR[i][j] != -1) and (QSR[i][j] != 0)): 
                break #This line uses the break statement to terminate the for-loop if the actions are evaluated.
            #The if-statement with 2 lines checks if the convergence is not successful.
            if (j == 24): 
                return False #This line returns False if the state is 24.
    return (QSR == copyQSR).all() #This line checks the iterable elements of the current Q-Matrix and the previous Q-Matrix and returns True if the iterable elements are true.

"""
These 12 lines that are below this multi-line comment define the check_ConvergenceEG() function that checks the convergence of Decaying Epsilon-Greedy.
"""
def checkConvergenceEG(copyQEG, QEG):
    #The for-loop with 9 lines checks if 1 state has a value of the Q-Matrix.
    for i in range(1, 24):
        #The for-loop with 7 lines checks if the convergence is successful.
        for j in range(0, 25):
            #The if-statement with 2 lines checks if the state that has have recorded value is the next state.
            if ((REG[i][j] != -1) and (QEG[i][j] != 0)): 
                break #This line uses the break statement to terminate the for-loop if the actions are evaluated.
            #The if-statement with 2 lines checks if the convergence is not successful.
            if (j == 24): 
                return False #This line returns False if the state is 24.
    return (QEG == copyQEG).all() #This line checks the iterable elements of the current Q-Matrix and the previous Q-Matrix and returns True if the iterable elements are true.

#PART 1: RL MONTE CARLO FIRST VISIT ALGORITHM CODE
print('RL MONTE CARLO FIRST VISIT ALGORITHM:') #This line prints the "RL MONTE CARLO FIRST VISIT ALGORITHM:" message.
gamma = 0.9 #This line sets the gamma variable to 0.9 that defines the value of the discount rate.
terminationStates = np.array([[0, 0], [4, 4]]) #This line defines the terminationStates variable that represents the termination states of the policy array.
actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]) #This line defines the actions variable that states the 4 actions of the policy array when [-1, 0], [1, 0], [0, 1], and [0, -1] represent Up, Down, Right, and Left Actions respectively.

states = [[i, j] for i in range(5) for j in range(5)] #This line defines the states variable that creates and shows the 25 states of the policy array.  
N_sFV = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros for N(s). The policy array's shape is (5,5) since the grid/array size of N(s) is 5.
S_sFV = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros for S(s). The policy array's shape is (5,5) since the grid/array size of S(s) is 5.
V_sFV = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros for V(s). The policy array's shape is (5,5) since the grid/array size of V(s) is 5.
print('Epoch 0 (Initial Values - First Visit Method)') #This line prints the "Epoch 0 (Initial Values - First Visit Method)" message.
print('N(s)') #This line prints the "N(s)" message.
print(N_sFV) #This line prints the N(s) matrix of First Visit Algorithm.
print('S(s)') #This line prints the "S(s)" message.
print(S_sFV) #This line prints the S(s) matrix of First Visit Algorithm.
print('V(s)') #This line prints the "V(s)" message.
print(V_sFV) #This line prints the V(s) matrix of First Visit Algorithm.

#The for-loop with 61 lines uses the Monte Carlo First Visit Algorithm to calculate the values of the states and prints the policy arrays of their corresponding epochs/iterations.
for epoch in range(1, 100000):
    copyV_sFV = np.copy(V_sFV) #This line copies the previous values and makes a copy array of V(s) to update the current iteration and check for convergence.
    position = states[random.randint(1,len(states)-2)] #This line defines the position variable that randomly selects an initial state if it is not a termination state.
    G_s_table = np.zeros((0, 5)) #This line creates a new array of the given shape with zeros for the G(s) table that will calculate the values of k, s, r, γ, and G(s). The policy array's shape is (5,5) since the grid/array size of G(s) is 5.
    visitedGrid = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros to track the evaluated states. The policy array's shape is (5,5) since the grid/array size of G(s) is 5.
    k = 1 #This line sets the k variable to 1 since it represents the number of state-reward pairs and is used for the rows of the G(s) table.
    row = [k, ((position[0] * 5) + position[1]), 0 if any(np.array_equal(x, position) for x in terminationStates) else -1, gamma, 0] #This line adds the initial row to the G(s) table.
    G_s_table = np.vstack([G_s_table,row]) #This line stacks the arrays values of k, s, r, γ, and G(s) vertically for the G(s) table.
    #The while loop with 6 lines calculates the values of the G(s) table if the states are not termination state.
    while not any(np.array_equal(x, position) for x in terminationStates): 
        k += 1 #This line increments the count of the k variable if the state is not a termination state.
        action = random.choice(actions) #This line randomly selects actions like Up, Down, Right, and Left Actions.
        position, reward = actionReward(position, action) #This line calculates the positions and the rewards of the correspoding actions by the actionReward() function.
        row = [k, ((position[0] * 5) + position[1]), reward, gamma, 0] #This line records the k, s, r, γ values of one action in a row.
        G_s_table = np.vstack([G_s_table,row]) #This line stacks the arrays values of k, s, r, γ, and G(s) vertically for the G(s) table.
    num_actions = G_s_table.shape[0] #This line defines the rows of the G(s) table since they are needed for the table.
    #The for-loop with 13 lines calculates G(s) for one row and N(s) and S(s) matrices for one position.
    for i in range(num_actions - 1):
        k = int(G_s_table[i][0]) #This line converts the array values that are included the k variable since it represents the number of state-reward pairs.
        s = int(G_s_table[i][1]) #This line converts the array values that are included the s variable since it represents the number of states.
        state = [s//5, s%5] #This line calculates the states with floor division and modulus.
        #The for-loop with 3 lines calculates the values of k, s, r, γ, and G(s) table for one row.
        for j in range(int(num_actions - k) + 1):
            current = G_s_table[int(k + j - 1)] #This line defines the current variable or the current state of the G(s) table.
            G_s_table[i][4] += pow(current[3], j)*(current[2]) #This line uses the formula of G(s) = r(t+1)+ γ*r(t+2) + γ^(2)*r(t+3) + … + γ^(k-1)*r(k) to calculate the values of the current state.
        #The if-statement with 3 lines updates N(s) if the state is not a termination state.
        if not any(np.array_equal(x, state) for x in terminationStates) and not visitedGrid[state[0]][state[1]]:
            N_sFV[state[0]][state[1]] += 1 #This line calculates the values of N(s) matrix that represnts the count of the visits.
            S_sFV[state[0]][state[1]] += G_s_table[i][4] #This line calculates the values of S(s) matrix that represents of sum of the visits.
        visitedGrid[state[0]][state[1]] = 1  #This line tracks the visited states after their respective values are evaluated.     
    #The for-loop with 5 lines calculates V(s) = S(s)/N(s) for one state.
    for s in range(25):
        state = [s//5, s%5] #This line calculates the states with floor division and modulus.
        #The if-statement with 2 lines uses the formula of V(s) if the value of N(s) is greater than 0.
        if int(N_sFV[state[0]][state[1]]) > 0:
            V_sFV[state[0]][state[1]] = S_sFV[state[0]][state[1]]/int(N_sFV[state[0]][state[1]]) #This line calculates the values of V(s) matrix that uses its formula.  
    V_sFV = np.around(V_sFV, decimals = 2) #This line rounds the values of V(s) to tenths' place.
    S_sFV = np.around(S_sFV, decimals = 2) #This line rounds the values of S(s) to tenths' place.
    #The if-statement with 11 lines uses the all() function to see if the iterable elements of the current policy array and the previous policy array are true. Then, it prints the policy array of the final epoch/iteration after convergence.
    if (V_sFV == copyV_sFV).all():
        print('\nEpoch', epoch, '(Final Epoch - First Visit Method)')  #This line makes a new line and prints the "Epoch" message, its corresponding epoch value, and the "(Final Epoch - First Visit Method)" message.
        print('N(s)') #This line prints the "N(s)" message.
        print(N_sFV) #This line prints the N(s) matrix of First Visit Algorithm.
        print('S(s)') #This line prints the "S(s)" message.
        print(S_sFV) #This line prints the S(s) matrix of First Visit Algorithm.
        print('V(s)') #This line prints the "V(s)" message.
        print(V_sFV) #This line prints the V(s) matrix of First Visit Algorithm.
        print('k, s, r, γ, and G(s):') #This line prints the "k, s, r, γ, and G(s):" message.
        print(G_s_table) #This line prints the table of the values of k, s, r, γ, and G(s).
        break #This line uses the break statement to terminate the epoch/iteration of the for loop since the policy array of the final epoch/iteration is printed.
    #The if-statement with 10 lines prints the matrices of Epoch 1 and Epoch 10.
    if epoch in [1,10]:
        print('\nEpoch', epoch, '- First Visit Method') #This line keeps on making new lines and printing the "Epoch" messages, their corresponding epoch values, and the "- First Visit Method" message due to the for-loop.
        print('N(s)') #This line prints the "N(s)" message.
        print(N_sFV) #This line prints the N(s) matrix of First Visit Algorithm.
        print('S(s)') #This line prints the "S(s)" message.
        print(S_sFV) #This line prints the S(s) matrix of First Visit Algorithm.
        print('V(s)') #This line prints the "V(s)" message.
        print(V_sFV) #This line prints the V(s) matrix of First Visit Algorithm.
        print('k, s, r, γ, and G(s):') #This line prints the "k, s, r, γ, and G(s):" message.
        print(G_s_table) #This line prints the table of the values of k, s, r, γ, and G(s).
plot_graph(V_sFV.flatten(), 'Error Value FV') #This line converts the matrix into an array of data points by the flatten() function and plots the graph of the error value. This function is borrowed from PlottingCode.py file.

#PART 2: RL MONTE CARLO EVERY VISIT ALGORITHM CODE
print('\nRL MONTE CARLO EVERY VISIT ALGORITHM:') #This line makes a new line and prints the "RL MONTE CARLO EVERY VISIT ALGORITHM:" message.
N_sEV = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros for N(s). The policy array's shape is (5,5) since the grid/array size of N(s) is 5.
S_sEV = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros for S(s). The policy array's shape is (5,5) since the grid/array size of S(s) is 5.
V_sEV = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros for V(s). The policy array's shape is (5,5) since the grid/array size of V(s) is 5.
print('Epoch 0 (Initial Values - Every Visit Method)') #This line prints the "Epoch 0 (Initial Values - Every Visit Method)" message.
print('N(s)') #This line prints the "N(s)" message.
print(N_sEV) #This line prints the N(s) matrix of Every Visit Algorithm.
print('S(s)') #This line prints the "S(s)" message.
print(S_sEV) #This line prints the S(s) matrix of Every Visit Algorithm.
print('V(s)') #This line prints the "V(s)" message.
print(V_sEV) #This line prints the V(s) matrix of Every Visit Algorithm.

#The for-loop with 60 lines uses the Monte Carlo Every Visit Algorithm to calculate the values of the states and prints the policy arrays of their corresponding epochs/iterations.
for epoch in range(1, 100000):
    copyV_sEV = np.copy(V_sEV) #This line copies the previous values and makes a copy array of V(s) to update the current iteration and check for convergence.
    position = states[random.randint(1,len(states)-2)] #This line defines the position variable that randomly selects an initial state if it is not a termination state.
    G_s_array = np.zeros((0, 5)) #This line creates a new array of the given shape with zeros for the G(s) table that will calculate the values of k, s, r, γ, and G(s). The policy array's shape is (5,5) since the grid/array size of G(s) is 5.
    k = 1 #This line sets the k variable to 1 since it represents the number of state-reward pairs and is used for the rows of the G(s) table.
    row = [k, ((position[0] * 5) + position[1]), 0 if any(np.array_equal(x, position) for x in terminationStates) else -1, gamma, 0] #This line adds the initial row to the G(s) table.
    G_s_array = np.vstack([G_s_array,row]) #This line stacks the arrays values of k, s, r, γ, and G(s) vertically for the G(s) table.
    #The while loop with 6 lines calculates the values of the G(s) table if the states are not termination state.
    while not any(np.array_equal(x, position) for x in terminationStates): 
        k += 1 #This line increments the count of the k variable if the state is not a termination state.
        action = random.choice(actions) #This line randomly selects actions like Up, Down, Right, and Left Actions.
        position, reward = actionReward(position, action) #This line calculates the positions and the rewards of the correspoding actions by the actionReward() function.
        row = [k, ((position[0] * 5) + position[1]), reward, gamma, 0] #This line records the k, s, r, γ values of one action in a row.
        G_s_array = np.vstack([G_s_array, row]) #This line stacks the arrays values of k, s, r, γ, and G(s) vertically for the G(s) table
    num_actions = G_s_array.shape[0] #This line defines the rows of the G(s) table since they are needed for the table.
    #The for-loop with 13 lines calculates G(s) for one row and N(s) and S(s) matrices for one position.
    for i in range(num_actions):
        k = int(G_s_array[i][0]) #This line converts the array values that are included the k variable since it represents the number of state-reward pairs.
        s = int(G_s_array[i][1]) #This line converts the array values that are included the s variable since it represents the number of states.
        state = [s//5, s%5] #This line calculates the states with floor division and modulus.
        #The for-loop with  lines calculates the values of k, s, r, γ, and G(s) table for one row.
        for j in range(int(num_actions - k) + 1):
            # G(s) = r(t+1)+ γ*r(t+2) + γ^(2)*r(t+3) + … + γ^(k-1)*r(k)
            current = G_s_array[int(k + j - 1)] #This line defines the current variable or the current state of the G(s) table.
            G_s_array[i][4] += pow(current[3], j)*(current[2]) #This line uses the formula of G(s) = r(t+1)+ γ*r(t+2) + γ^(2)*r(t+3) + … + γ^(k-1)*r(k) to calculate the values of the current state.
        #The if-statement with 2 lines updates N(s) if the state is not a termination state.
        if not any(np.array_equal(x, state) for x in terminationStates):
            N_sEV[state[0]][state[1]] += 1 #This line calculates the values of N(s) matrix that represnts the count of the visits.
        S_sEV[state[0]][state[1]] += G_s_array[i][4] #This line calculates the values of S(s) matrix that represents of sum of the visits.
    #The for-loop with 5 lines calculates V(s) = S(s)/N(s) for one state.
    for s in range(25):
        state = [s//5, s%5] #This line calculates the states with floor division and modulus.
        #The if-statement with 2 lines uses the formula of V(s) if the value of N(s) is greater than 0.
        if int(N_sEV[state[0]][state[1]]) > 0:
            V_sEV[state[0]][state[1]] = S_sEV[state[0]][state[1]]/int(N_sEV[state[0]][state[1]]) #This line calculates the values of V(s) matrix that uses its formula.
    V_sEV = np.around(V_sEV, decimals = 2) #This line rounds the values of V(s) to tenths' place.
    S_sEV = np.around(S_sEV, decimals = 2) #This line rounds the values of S(s) to tenths' place.
    #The if-statement with 11 lines uses the all() function to see if the iterable elements of the current policy array and the previous policy array are true. Then, it prints the policy array of the final epoch/iteration after convergence.
    if (V_sEV == copyV_sEV).all():
        print('\nEpoch', epoch, '(Final Epoch - Every Visit Method)')  #This line makes a new line and prints the "Epoch" message, its corresponding epoch value, and the "(Final Epoch - Every Visit Method)" message.
        print('N(s)') #This line prints the "N(s)" message.
        print(N_sEV) #This line prints the N(s) matrix of Every Visit Algorithm.
        print('S(s)') #This line prints the "S(s)" message.
        print(S_sEV) #This line prints the S(s) matrix of Every Visit Algorithm.
        print('V(s)') #This line prints the "V(s)" message.
        print(V_sEV) #This line prints the V(s) matrix of Every Visit Algorithm.
        print('k, s, r, γ, and G(s):') #This line prints the "k, s, r, γ, and G(s):" message.
        print(G_s_array) #This line prints the array of k, s, r, γ, and G(s).
        break #This line uses the break statement to terminate the epoch/iteration of the for loop since the policy array of the final epoch/iteration is printed.
    #The if-statement with 10 lines prints the matrices of Epoch 1 and Epoch 10.
    if epoch in [1,10]:
        print('\nEpoch', epoch, '- Every Visit Method') #This line keeps on making new lines and printing the "Epoch" messages, their corresponding epoch values, and the "- Every Visit Method" message due to the for-loop.
        print('N(s)') #This line prints the "N(s)" message.
        print(N_sEV) #This line prints the N(s) matrix of Every Visit Algorithm.
        print('S(s)') #This line prints the "S(s)" message.
        print(S_sEV) #This line prints the S(s) matrix of Every Visit Algorithm.
        print('V(s)') #This line prints the "V(s)" message.
        print(V_sEV) #This line prints the V(s) matrix of Every Visit Algorithm.
        print('k, s, r, γ, and G(s):') #This line prints the "k, s, r, γ, and G(s):" message.
        print(G_s_array) #This line prints the array of k, s, r, γ, and G(s).
plot_graph(V_sEV.flatten(), 'Error Value EV') #This line converts the matrix into an array of data points by the flatten() function and plots the graph of the error value. This function is borrowed from PlottingCode.py file.

#PART 3: RL Q-LEARNING ALGORITHM CODE
print('\nRL Q-LEARNING ALGORITHM:') #This line makes a new line and prints the "RL Q-LEARNING ALGORITHM:" message.
gamma = 0.9 #This line sets the gamma variable to 0.9 that defines the value of the discount rate.
terminationStates = np.array([[0, 0], [4, 4]]) #This line defines the terminationStates variable that represents the termination states of the array.
actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]) #This line defines the actions variable that states the 4 actions of the array when [-1, 0], [1, 0], [0, 1], and [0, -1] represent Up, Down, Right, and Left Actions respectively.

QQL = np.zeros((25, 25)) #This line creates a new array of the given shape with zeros for the Q-Matrix/Value Matrix. The array's shape is (25,25) since the grid/array size is 25.
RQL = np.full((25, 25), -1) #This line creates a new array of the given shape with zeros for the R-Matrix/Reward Matrix and fills this array with -1's. The array's shape is (25,25) since the grid/array size is 25.

#The for-loop with 10 lines initializes the values in the R-Matrix.
for state in range(25):
    #The if-statement with 2 lines sets the state to 100 as the reward value.
    if any(np.array_equal(x, [state//5, state%5]) for x in terminationStates):
        RQL[int(state)][int(state)] = 100 #This line sets the reward value to 100 if the termination states and coordinate positions are equal.
    #The for-loop with 5 lines initializes the values before the values are printed.
    for action in actions:
        neighbor = np.add([state//5, state%5], action) #This line defines the neighbor variable that adds the coordinate position array and the action array.
        #The if-statement with 2 lines sets the reward value to 100 or 0 for certain states.
        if not(-1 in neighbor or 5 in neighbor): 
            RQL[int(state)][int(neighbor[0]*5 + neighbor[1])] = 100 if any(np.array_equal(x, neighbor) for x in terminationStates) else 0 #This line sets the reward value of 100 if the neighbor state and the termination state are equal. If not, then the reward value is 0.

print("Q-Learning Rewards Matrix (R)") #This line prints the "Q-Learning Rewards Matrix (R)" message.
print_matrix(RQL) #This line prints the R-Matrix of Q-Learning Algorithm.
print("\nEpisode 0 (Initial Values - Q-Learning Value Matrix (Q))") #This line makes a new line and prints the "Episode 0 (Initial Values - Q-Learning Value Matrix (Q))" message.
print_matrix(QQL) #This line prints the Q-Matrix of Q-Learning Algorithm.
#The for-loop with 38 lines uses the Q-Learning Algorithm (Off-Policy) to calculate the values of the states and prints the matrices of their corresponding episodes/iterations.
for episode in range(1, 100000):
    copyQQL = np.copy(QQL) #This line copies the previous values and makes a copy array of Q-Matrix to update the current iteration and check for convergence.
    state = random.randint(0, 24) #This line defines the state variable that randomly selects an initial state if it is not a termination state.
    #The while loop with 22 lines calcultaes the values of the Q-Matrix if the states are not termination state.
    while not any(np.array_equal(x, [state//5, state%5]) for x in terminationStates):
        actions = np.array([]) #This line defines the actions variable that currently has an empty array and will find the next states.
        #The for-loop with 4 lines appends the values in the Q-Matrix. 
        for i in range(25):
            #The if-statement with 2 lines appends the actions if the reward value is not -1.
            if (RQL[state][i] != -1):
                actions = np.append(actions, i) #This line appends the actions. 
        action = int(random.choice(actions)) #This line randomly selects the next state.
        next_actions = np.array([]) #This line defines the next_actions variable that currently has an empty array and will find the next states.
        #The for-loop with 4 lines appends the values in the Q-Matrix.       
        for i in range(25):
            #The if-statement with 2 lines appends the next actions if the reward value is not -1.
            if (RQL[action][i] != -1):
                next_actions = np.append(next_actions, i) #This line appends the next actions.
        max = 0 #This lines defines the max variable that will find the max value of the values of the next states of the Q-Matrix.
        #The for-loop with 4 lines calculates the max value for the formula of Q-Learning Algorithm.
        for next in next_actions:
            #The if-statement with 2 lines calculates the max value if the Q-Matrix's next state's value is greater than the max value.
            if (QQL[action][int(next)] > max):
                max = QQL[action][int(next)] #This line sets the max variable to the Q-Matrix's next state's value.
        QQL[state][action] = RQL[state][action] + gamma*max #This line uses the formula of Q(state, action) = R(state, action) + gamma(max[Q(next state, actions]) to calculate the value of the state.
        state = action #This line updates the state after the state's value is added in the Q-Matrix.
    QQL = np.around(QQL, decimals = 3) #This line rounds the values of the Q-Matrix to hundredths' place.
    #The if-statement with 4 lines uses the checkConvergenceQL() function to see if the iterable elements of the current Q-Matrix and the previous Q-Matrix are true. Then, it prints the Q-Matrix of the final episode/iteration after convergence.
    if checkConvergenceQL(copyQQL, QQL):
        print('\nEpisode', episode, '(Final Episode - Q-Learning Value Matrix (Q))')  #This line makes a new line and prints the "Episode" messages, its corresponding epoch value, and the "(Final Episode - Q-Learning Value Matrix (Q))" message.
        print_matrix(QQL) #This line prints the Q-Matrix of Q-Learning Algorithm.
        break #This line uses the break statement to terminate the episode/iteration of the for loop since the Q-Matrix of the final epoch/iteration is printed.
    cum_rewardsQL = np.zeros((25,25)) #This line defines the cum_rewardsQL variable that creates a new array of the given shape with zeros to calculate the cumulative average reward values. The array's shape is (25,25) since the grid/array size is 25.
    cum_rewardsQL += QQL #This adds the values of the Q-Matrix of the episodes/iterations and stores them in the array.
    #The if-statement with 3 lines prints the matrices of Episode 1 and Episode 10.
    if episode in [1,10]:
        print("\nEpisode", episode, "- Q-Learning Value Matrix (Q)") #This line keeps on making new lines and printing the "Episode" messages, their corresponding episode values, and the "- Q-Learning Value Matrix (Q)" message due to the for-loop.
        print_matrix(QQL) #This line prints the Q-Matrix of Q-Learning Algorithm.
plot_graph(QQL.flatten(), 'Error Value QL') #This line converts the matrix into an array of data points by the flatten() function and plots the graph of the error value. This function is borrowed from PlottingCode.py file.

#PART 4: RL SARSA ALGORITHM CODE
print('\nRL SARSA ALGORITHM:') #This line makes a new line and prints the "RL SARSA ALGORITHM:" message.
actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]) #This line defines the actions variable that states the 4 actions of the policy array when [-1, 0], [1, 0], [0, 1], and [0, -1] represent Up, Down, Right, and Left Actions respectively.

QSR = np.zeros((25, 25)) #This line creates a new array of the given shape with zeros for the Q-Matrix/Value Matrix. The array's shape is (25,25) since the grid/array size is 25.
RSR = np.full((25, 25), -1) #This line creates a new array of the given shape with zeros for the R-Matrix/Reward Matrix and fills this array with -1's. The array's shape is (25,25) since the grid/array size is 25.

#The for-loop with 10 lines initializes the values in the R-Matrix.
for state in range(25):
    #The if-statement with 2 lines sets the state to 100 as the reward value.
    if any(np.array_equal(x, [state//5, state%5]) for x in terminationStates):
        RSR[int(state)][int(state)] = 100 #This line sets the reward value to 100 if the termination states and coordinate positions are equal.
    #The for-loop with 5 lines initializes the values before the values are printed.
    for action in actions:
        neighbor = np.add([state//5, state%5], action) #This line defines the neighbor variable that adds the coordinate position array and the action array.
        #The if-statement with 2 lines sets the reward value to 100 or 0 for certain states.
        if not(-1 in neighbor or 5 in neighbor): 
            RSR[int(state)][int(neighbor[0]*5 + neighbor[1])] = 100 if any(np.array_equal(x, neighbor) for x in terminationStates) else 0 #This line sets the reward value of 100 if the neighbor state and the termination state are equal. If not, then the reward value is 0.    

print("SARSA Rewards Matrix (R)") #This line prints the "SARSA Rewards Matrix (R)" message.
print_matrix(RSR) #This line prints the R-Matrix of SARSA Algorithm.
print("\nEpisode 0 (Initial Values - SARSA Value Matrix (Q))") #This line makes a new line and prints the "Episode 0 (Initial Values - SARSA Value Matrix (Q))" message.
print_matrix(QSR) #This line prints the Q-Matrix of SARSA Algorithm.
#The for-loop with 47 lines uses the SARSA Algorithm (On-Policy) to calculate the values of the states and prints the matrices of their corresponding episodes/iterations.
for episode in range(1, 100000):
    copyQSR = np.copy(QSR) #This line copies the previous values and makes a copy array of Q-Matrix to update the current iteration and check for convergence.
    state = random.randint(0, 24) #This line defines the state variable that randomly selects an initial state if it is not a termination state.
    #The while loop with 31 lines calcultaes the values of the Q-Matrix if the states are not termination state.
    while not any(np.array_equal(x, [state//5, state%5]) for x in terminationStates): 
        actions = np.array([]) #This line defines the actions variable that currently has an empty array and will find the next states.
        #The for-loop with 4 lines appends the values in the Q-Matrix. 
        for i in range(25):
            #The if-statement with 2 lines appends the actions if the reward value is not -1.
            if (RSR[state][i] != -1):
                actions = np.append(actions, i) #This line appends the actions.       
        next_actions = np.array([]) #This line defines the next_actions variable that currently has an empty array and will find the next states.
        next_actions = np.append(next_actions, int(actions[0])) #This line appends the next actions and the current actions.
        #The for-loop with 6 lines checks if the values are required for the Q-Matrix.
        for action in np.delete(actions, 0):
            #The if-else statement with 4 lines adds the values of the Q-Matrix.
            if (QSR[state][int(action)] > QSR[state][int(next_actions[0])]): 
                next_actions = np.array([action]) #This line adds and considers the action if the value of the Q-Matrix is greater than the value of the next action.
            elif (QSR[state][int(action)] == QSR[state][int(next_actions[0])]):
                next_actions = np.append(next_actions, action) #This line adds and appends the action if the value of the Q-Matrix is equal to the action.
        action = int(random.choice(next_actions)) #This line randomly selects the next action of the next state.
        next_actions = np.array([]) #This line defines the next_actions variable that currently has an empty array and will find the next states.
        #The for-loop with 4 lines appends the values in the Q-Matrix.       
        for i in range(25):
            #The if-statement with 2 lines appends the next actions if the reward value is not -1.
            if (RSR[action][i] != -1):
                next_actions = np.append(next_actions, i) #This line appends the next actions.
        max = 0 #This lines defines the max variable that will find the max value of the values of the next states of the Q-Matrix.
        #The for-loop with 4 lines calculates the max value for the formula of SARSA Algorithm.
        for next in next_actions:
            #The if-statement with 2 lines calculates the max value if the Q-Matrix's next state's value is greater than the max value.
            if QSR[action][int(next)] > max:
                max = QSR[action][int(next)] #This line sets the max variable to the Q-Matrix's next state's value.
        QSR[state][action] = RSR[state][action] + gamma*max #This line uses the formula of Q(state, action) = R(state, action) + gamma(max[Q(next state, actions]) to calculate the value of the state.
        state = action #This line updates the state after the state's value is added in the Q-Matrix.
    QSR = np.around(QSR, decimals = 3) #This line rounds the values of the Q-Matrix to hundredths' place.
    #The if-statement with 4 lines uses the checkConvergenceSR() function to see if the iterable elements of the current Q-Matrix and the previous Q-Matrix are true. Then, it prints the Q-Matrix of the final episode/iteration after convergence.
    if checkConvergenceSR(copyQSR, QSR):
        print('\nEpisode', episode, '(Final Episode - SARSA Value Matrix (Q))') #This line makes a new line and prints the "Episode" message, its corresponding episode value, and the "(Final Episode - SARSA Value Matrix (Q))" message.
        print_matrix(QSR) #This line prints the Q-Matrix of SARSA Algorithm.
        break #This line uses the break statement to terminate the episode/iteration of the for loop since the Q-Matrix of the final epoch/iteration is printed.
    cum_rewardsSR = np.zeros((25,25)) #This line defines the cum_rewardsSR variable that creates a new array of the given shape with zeros to calculate the cumulative average reward values. The array's shape is (25,25) since the grid/array size is 25.
    cum_rewardsSR += QSR #This adds the values of the Q-Matrix of the episodes/iterations and stores them in the array.
    #The if-statement with 3 lines prints the matrices of Episode 1 and Episode 10.
    if episode in [1,10]:
        print("\nEpisode", episode, "- SARSA Value Matrix (Q)") #This line keeps on making new lines and printing the "Episode" messages, their corresponding episode values, and the "- SARSA Value Matrix (Q)" message due to the for-loop.
        print_matrix(QSR) #This line prints the Q-Matrix of SARSA Algorithm.
plot_graph(QSR.flatten(), 'Error Value SR') #This line converts the matrix into an array of data points by the flatten() function and plots the graph of the error value. This function is borrowed from PlottingCode.py file.

#PART 5: RL DECAYING EPSILON-GREEDY ALGORITHM CODE
print('\nRL DECAYING EPSILON-GREEDY ALGORITHM:') #This line makes a new line and prints the "RL DECAYING EPSILON-GREEDY ALGORITHM:" message.
actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]) #This line defines the actions variable that states the 4 actions of the policy array when [-1, 0], [1, 0], [0, 1], and [0, -1] represent Up, Down, Right, and Left Actions respectively.

QEG = np.zeros((25, 25)) #This line creates a new array of the given shape with zeros for the Q-Matrix/Value Matrix. The array's shape is (25,25) since the grid/array size is 25.
REG = np.full((25, 25), -1) #This line creates a new array of the given shape with zeros for the R-Matrix/Reward Matrix and fills this array with -1's. The array's shape is (25,25) since the grid/array size is 25.

#The for-loop with 10 lines initializes the values in the R-Matrix.
for state in range(25):
    #The if-statement with 2 lines sets the state to 100 as the reward value.
    if any(np.array_equal(x, [state//5, state%5]) for x in terminationStates):
        REG[int(state)][int(state)] = 100 #This line sets the reward value to 100 if the termination states and coordinate positions are equal.
    #The for-loop with 5 lines initializes the values before the values are printed.
    for action in actions:
        neighbor = np.add([state//5, state%5], action) #This line defines the neighbor variable that adds the coordinate position array and the action array.
        #The if-statement with 2 lines sets the reward value to 100 or 0 for certain states.
        if not(-1 in neighbor or 5 in neighbor): 
            REG[int(state)][int(neighbor[0]*5 + neighbor[1])] = 100 if any(np.array_equal(x, neighbor) for x in terminationStates) else 0 #This line sets the reward value of 100 if the neighbor state and the termination state are equal. If not, then the reward value is 0.    
            
print("Decaying Epsilon-Greedy Rewards Matrix (R)") #This line prints the "Decaying Epsilon-Greedy Rewards Matrix (R)" message.
print_matrix(REG) #This line prints the R-Matrix of Decaying Epsilon-Greedy Algorithm.
print("\nEpisode 0 (Initial Values - Decaying Epsilon-Greedy Value Matrix (Q))") #This line makes a new line and prints the "Episode 0 (Initial Values - Decaying Epsilon-Greedy Value Matrix (Q))" message.
print_matrix(QEG) #This line prints the Q-Matrix of Decaying Epsilon-Greedy Algorithm.

#The for-loop with 56 lines uses the Decaying Epsilon-Greedy Algorithm/the combination of Q-Learning and SARSA Algorithms to calculate the values of the states and prints the matrices of their corresponding episodes/iterations.
for episode in range(1, 100000):
    copyQEG = np.copy(QEG) #This line copies the previous values and makes a copy array of Q-Matrix to update the current iteration and check for convergence.
    state = random.randint(0, 24) #This line defines the state variable that randomly selects an initial state if it is not a termination state.
    epsilon = math.exp(-(episode - 1)/100) #This line uses the math.exp() function from the math library and defines the epsilon variable that uses the decaying function = e^(-(t-1)/c) when t and c are episode/iteration value and constant value/100 respectively.
    #The while loop with 31 lines calcultaes the values of the Q-Matrix if the states are not termination state.
    while not any(np.array_equal(x, [state//5, state%5]) for x in terminationStates): 
        actions = np.array([]) #This line defines the actions variable that currently has an empty array and will find the next states.
        #The if-else statement with 22 lines combines Q-Learning and SARSA Algorithms. It picks Q-Learning/Off-Policy if the uniform random number that is between 0 and 1 is less than the epsilon value. If not, then it picks SARSA/On-Policy.
        if (random.uniform(0, 1) < epsilon):
             #The for-loop with 4 lines appends the values in the Q-Matrix.
             for i in range(25):
               #The if-statement with 2 lines appends the actions if the reward value is not -1.
               if (REG[state][i] != -1):
                   actions = np.append(actions, i) #This line appends the actions. 
             action = int(random.choice(actions)) #This line randomly selects the next state.
        else:
             #The for-loop with 4 lines appends the values in the Q-Matrix.
             for i in range(25):
               #The if-statement with 2 lines appends the actions if the reward value is not -1.
               if (REG[state][i] != -1):
                   actions = np.append(actions, i) #This line appends the actions.       
             next_actions = np.array([]) #This line defines the next_actions variable that currently has an empty array and will find the next states.
             next_actions = np.append(next_actions, int(actions[0])) #This line appends the next actions and the current actions.
             for action in np.delete(actions, 0):
                 #The if-else statement with 4 lines adds the values of the Q-Matrix.
                 if (QEG[state][int(action)] > QEG[state][int(next_actions[0])]): 
                     next_actions = np.array([action]) #This line adds and considers the action if the value of the Q-Matrix is greater than the value of the next action.
                 elif (QEG[state][int(action)] == QEG[state][int(next_actions[0])]):
                     next_actions = np.append(next_actions, action) #This line adds and appends the action if the value of the Q-Matrix is equal to the action.
             action = int(random.choice(next_actions))
        next_actions = np.array([]) #This line defines the next_actions variable that currently has an empty array and will find the next states.
        #The for-loop with 4 lines appends the values in the Q-Matrix.       
        for i in range(25):
            #The if-statement with 2 lines appends the next actions if the reward value is not -1.
            if (REG[action][i] != -1):
                next_actions = np.append(next_actions, i) #This line appends the next actions.
        max = 0 #This lines defines the max variable that will find the max value of the values of the next states of the Q-Matrix.
        #The for-loop with 4 lines calculates the max value for the formula of Decaying Epsilon-Greedy Algorithm.
        for next in next_actions:
            #The if-statement with 2 lines calculates the max value if the Q-Matrix's next state's value is greater than the max value.
            if QEG[action][int(next)] > max:
                max = QEG[action][int(next)] #This line sets the max variable to the Q-Matrix's next state's value.
        QEG[state][action] = REG[state][action] + gamma*max #This line uses the formula of Q(state, action) = R(state, action) + gamma(max[Q(next state, actions]) to calculate the value of the state.
        state = action #This line updates the state after the state's value is added in the Q-Matrix.
    QEG = np.around(QEG, decimals = 3) #This line rounds the values of the Q-Matrix to hundredths' place.
    #The if-statement with 4 lines uses the checkConvergenceEG() function to see if the iterable elements of the current Q-Matrix and the previous Q-Matrix are true. Then, it prints the Q-Matrix of the final episode/iteration after convergence.
    if checkConvergenceEG(copyQEG, QEG):
        print('\nEpisode', episode, '(Final Episode - Decaying Epsilon-Greedy Value Matrix (Q))') #This line makes a new line and prints the "Episode" message, its corresponding episode value, and the "(Final Episode - Decaying Epsilon-Greedy Value Matrix (Q))" message.
        print_matrix(QEG) #This line prints the Q-Matrix of Decaying Epsilon-Greedy Algorithm..
        break #This line uses the break statement to terminate the episode/iteration of the for loop since the Q-Matrix of the final epoch/iteration is printed.
    cum_rewardsEG = np.zeros((25,25)) #This line defines the cum_rewardsEG variable that creates a new array of the given shape with zeros to calculate the cumulative average reward values. The array's shape is (25,25) since the grid/array size is 25.
    cum_rewardsEG += QEG #This adds the values of the Q-Matrix of the episodes/iterations and stores them in the array.
    #The if-statement with 3 lines prints the matrices of Episode 1 and Episode 10.
    if episode in [1,10]:
        print("\nEpisode", episode, "- Decaying Epsilon-Greedy Value Matrix (Q)") #This line keeps on making new lines and printing the "Episode" messages, their corresponding episode values, and the "- Decaying Epsilon-Greedy Value Matrix (Q)" message due to the for-loop.
        print_matrix(QEG) #This line prints the Q-Matrix of Decaying Epsilon-Greedy Algorithm. 
plot_graph(QEG.flatten(), 'Error Value EG') #This line converts the matrix into an array of data points by the flatten() function and plots the graph of the error value. This function is borrowed from PlottingCode.py file.

#PART 6: CUMULATIVE AVERAGE REWARD COMPARISON CODE
QL_cum_avg_reward = np.divide(np.cumsum(cum_rewardsQL),np.arange(1,626)) #This line defines the QL_cum_avg_reward variable that calculates the cumulative average of the values of Q-Learning Algorithm.
SR_cum_avg_reward = np.divide(np.cumsum(cum_rewardsSR),np.arange(1,626)) #This line defines the SR_cum_avg_reward variable that calculates the cumulative average of the values of SARSA Algorithm.
EG_cum_avg_reward = np.divide(np.cumsum(cum_rewardsEG),np.arange(1,626)) #This line defines the EG_cum_avg_reward variable that calculates the cumulative average of the values of Decaying Epsilon-Greedy Algorithm.
plot_cum_avg_reward(QL_cum_avg_reward.flatten(), SR_cum_avg_reward.flatten(), EG_cum_avg_reward.flatten(), 'Cumulative Average Reward Comparison') #This line converts the 3 matrices into arrays of data points by the flatten() function and plots the graph of these 3 arrays of the 3 algorithms for the cumulative average reward comparison. This function is borrowed from PlottingCode.py file.
