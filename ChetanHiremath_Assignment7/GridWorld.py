#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 7's GridWorld.py
#Description of the Program: This program used the Reinforcement Learning's Policy Iteration and Value Iteration Algorithms to generate their respective iterations' policy arrays or grids and graphs.
#Inputs of the Program: This program has various functions, Numpy and Matplotlib libraries, and parameters that are used in the program.
#Outputs of the Program: This program prints the Policy and the Value Iterations' policy arrays or grids that are represented as 5X5 matrices and plots the graphs of their respective iterations.
#Creation Date of the Program: November 12, 2023
#Collaborator/Collaborators of the Program: Ayaan Lakhani
#Source/Sources of the Program: RL Policy Iteration.pdf, RL Value Iteration.pdf, and PlottingCode.py (The code is used from these sources.). The background information of the links that are below this multi-line comment is applied for this program.
"""
Numpy.add's source link- https://numpy.org/doc/stable/reference/generated/numpy.add.html
Numpy.zeros' source link- https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
Numpy.copy's source link- https://numpy.org/doc/stable/reference/generated/numpy.copy.html
Numpy.empty's source link- https://numpy.org/doc/stable/reference/generated/numpy.empty.html
Numpy.insert's source link- https://numpy.org/doc/stable/reference/generated/numpy.insert.html
Numpy.max's source link- https://numpy.org/doc/stable/reference/generated/numpy.max.html
Python all()'s source link- https://www.programiz.com/python-programming/methods/built-in/all
Numpy.all's source link- https://numpy.org/doc/stable/reference/generated/numpy.all.html
Numpy.matrix.flatten's source link- https://numpy.org/doc/stable/reference/generated/numpy.matrix.flatten.html
"""

# Load Libraries
import numpy as np #This line imports the module from Numpy since this module is mandatory for the calculations of this program.
from matplotlib import pyplot as plt #This line imports the module from Matplotlib since this module is used to plot the graph of data points.
"""
These 10 lines that are below this multi-line comment define the actionReward() function that returns the final position and the reward.
"""
def actionReward(initialPosition, action):
    #The if-statement with 2 lines keeps the reward from returning to one of the 2 termination states at 0 and won't update the value of the position if the initial position is in one of the 2 termination states.
    if initialPosition in terminationStates:
        return (initialPosition, 0) #This line returns the initial position the reward value of 0 when it is in one of the 2 termination states.
    reward = -1 #This line sets the reward variable to -1 that defines the value of the reward.
    finalPosition = np.add(initialPosition, action) #This line defines the finalPosition variable that calculates the sum of the initial position array and the action array.
    #The if-statement with 2 lines returns the initial position to exit the policy grid, so it will be the final position.
    if reward in finalPosition or 5 in finalPosition: 
        finalPosition = initialPosition #This line sets the finalPosition variable to the initialPosition variable since the final position is considered as the initial position.    
    return (finalPosition, reward) #This line returns the final position and its reward value.
"""
These 7 lines that are below this multi-line comment define the plot_graph() function that plots and shows a graph. This function is borrowed from PlottingCode.py file.
"""    
def plot_graph(arr, name):
    plt.plot(range(1, 26), arr, marker='o')
    plt.title(name + ' vs. t')
    plt.xlabel('t')
    plt.xticks(np.arange(1, 26, 1))
    plt.ylabel(name)
    plt.show()
    
#PART 1: RL POLICY ITERATION ALGORITHM CODE
print('RL POLICY ITERATION ALGORITHM:') #This line prints the "RL POLICY ITERATION ALGORITHM:" message.
gamma = 1 #This line sets the gamma variable to 1 that defines the value of the discount rate.
terminationStates = [[0, 0], [4, 4]] #This line defines the terminationStates variable that represents the termination states of the policy array.
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]] #This line defines the actions variable that states the 4 actions of the policy array when [-1, 0], [1, 0], [0, 1], and [0, -1] represent Up, Down, Right, and Left Actions respectively.
maximumIterations = 5000 #This line sets the maximumIterations variable to 5000 that represents the maximum count of the iterations.

policyArray = np.zeros((5, 5)) #This line creates a new array of the given shape with zeros. The policy array's shape is (5,5) since the grid/array size is 5.
states = [[i, j] for i in range(5) for j in range(5)] #This line defines the states variable that creates and shows the 25 states of the policy array.  

print('Iteration 0 (The Initial Values)') #This line prints the "Iteration 0 (The Initial Values)" message.
print(policyArray) #This line prints the policy array of the first iteration.

#The for-loop with 18 lines uses the Policy Iteration Algorithm to calculate the values of the states and prints the policy arrays of their corresponding iterations.
for iteration in range(maximumIterations):
    copyPolicyArray = np.copy(policyArray) #This line copies the previous values and makes a copy array to update the current iteration and check for convergence.
    for state in states:
        rewards = 0 #This line sets the rewards variable to 0, which is the current value of cell of the next iteration. Then, the algorithm will eventually update the cell value.
        for action in actions:
            finalPosition, reward = actionReward(state, action) #This line calculates the positions and the rewards of the correspoding actions by the actionReward() function.
            rewards += (1 / len(actions)) * (reward + (gamma * policyArray[finalPosition[0], finalPosition[1]])) #This line uses the Policy Iteration formula to calculate the rewards because of their respective actions.
        copyPolicyArray[state[0], state[1]] = rewards #This line sets the cell value for the next iteration by the copied states.
    #The if-statement with 4 lines uses the all() function to see if the iterable elements of the current policy array and the previous policy array are true. Then, it prints the policy array of the final iteration after convergence.
    if (policyArray == copyPolicyArray).all():
        print('\nIteration', (iteration + 1), '(Final Iteration)') #This line makes a new line and prints the "Iteration" messages, its corresponding iteration value, and the "(Final Iteration)" message.
        print(policyArray, '\n') #This line prints the policy array of the final iteration and makes a new line.
        break #This line uses the break statement to terminate the iteration of the for loop since the policy array of the final iteration is printed.
    policyArray = copyPolicyArray #This line updates the policy array.
    #The if-statement with 3 lines prints the policy arrays of Iteration 1 and Iteration 10.
    if ((iteration == 0) or (iteration == 9)):
        print('\nIteration', (iteration + 1)) #This line keeps on making new lines and printing the "Iteration" messages and their corresponding iteration values due to the for-loop.
        print(policyArray) #This line keeps on printing the policy arrays of their respective iterations.
plot_graph(policyArray.flatten(), 'Error Value') #This line converts the matrix into an array of data points by the flatten() function and plots the graph of the error value. This function is borrowed from PlottingCode.py file.

#PART 2: RL VALUE ITERATION ALGORITHM CODE
print('RL VALUE ITERATION ALGORITHM:') #This line prints the "RL VALUE ITERATION ALGORITHM:" message.
policyArray = np.zeros((5, 5), dtype = int) #This line creates a new array of the given shape with zeros. The policy array's shape is (5,5) since the grid/array size is 5. The dtype paramter is int since it will convert the float values to integer values.
states = [[i, j] for i in range(5) for j in range(5)] #This line defines the states variable that creates and shows the 25 states of the policy array.

print('Iteration 0 (The Initial Values)') #This line prints the "Iteration 0 (The Initial Values)" message.
print(policyArray) #This line prints the policy array of the first iteration.

#The for-loop with 19 lines uses the Value Iteration Algorithm to calculate the values of the states and prints the policy arrays of their corresponding iterations.
for iteration in range(maximumIterations):
    copyPolicyArray = np.copy(policyArray) #This line copies the previous values and makes a copy array to update the current iteration and check for convergence.
    for state in states:
        rewards = np.empty(shape = 0) #This line sets the rewards variable to an empty array that will include the values of the cells. Then, the algorithm will eventually update the cell values.
        for action in actions:
            finalPosition, reward = actionReward(state, action) #This line calculates the positions and the rewards of the correspoding actions by the actionReward() function.
            calculatedReward = reward + (gamma * policyArray[finalPosition[0], finalPosition[1]]) #This line uses the Value Iteration formula to calculate the rewards because of their respective actions.
            rewards = np.insert(rewards, rewards.size, calculatedReward) #This line inserts a reward array in the given index.
        copyPolicyArray[state[0], state[1]] = np.max(rewards)  #This line finds the maximum values of the cell values for the next iteration by the copied states.
    #The if-statement with 4 lines uses the all() function to see if the iterable elements of the current policy array and the previous policy array are true. Then, it prints the policy array of the final iteration after convergence.
    if (policyArray == copyPolicyArray).all():
        print('\nIteration', (iteration + 1), '(Final Iteration)') #This line makes a new line and prints the "Iteration" messages, its corresponding iteration value, and the "(Final Iteration)" message.
        print(policyArray) #This line prints the policy array of the final iteration.
        break #This line uses the break statement to terminate the iteration of the for loop since the policy array of the final iteration is printed.
    policyArray = copyPolicyArray #This line updates the policy array.
    #The if-statement with 3 lines prints the policy arrays of Iteration 1 and Iteration 2.
    if ((iteration == 0) or (iteration == 1)):
        print('\nIteration', (iteration + 1)) #This line keeps on making new lines and printing the "Iteration" messages and their corresponding iteration values due to the for-loop.
        print(policyArray) #This line keeps on printing the policy arrays of their respective iterations.   
plot_graph(policyArray.flatten(), 'Error Value') #This line converts the matrix into an array of data points by the flatten() function and plots the graph of the error value. This function is borrowed from PlottingCode.py file.
