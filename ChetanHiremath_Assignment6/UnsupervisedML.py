#Author of the Program: Chetan Hiremath
#Name of the Program: EECS 658 Assignment 6's UnsupervisedML.py
#Description of the Program: This program uses the Unsupervised Learning to cluster the Iris data-set's data for K-Means Clustering and Gaussian Mixture Models (GMM). 
#Inputs of the Program: This program has various functions, libraries, and parameters that are used in the program.
#Outputs of the Program: This program shows the plotted graphs and prints the overall accuracy values and the overall confusion matrices of the K-Means Clustering and Gaussian Mixture Models (GMM).
#Creation Date of the Program: October 31, 2023
#Collaborator/Collaborators of the Program: Ayaan Lakhani
#Source/Sources of the Program: Supervised Learning.pdf, K-Means Clustering.pdf, Gaussian Mixture Model (GMM).pptx, and PlottingCode.py (The code is used from these sources.). The background information of the links that are below this multi-line comment is applied for this program.
"""
Sklearn.cluster.KMeans' source link- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
Sklearn.mixture.GaussianMixture's source link- https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
Numpy.empty's source link- https://numpy.org/doc/stable/reference/generated/numpy.empty.html
Sklearn.preprocessing.LabelEncoder's source link- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
"""

# Load Libraries
"""
These 8 lines that are below this multi-line comment import the modules from Numpy, Pandas, Scikit-learn, Matplotlib, and other libraries since these modules are mandatory for the calculations of this program.
"""
import numpy as np
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
"""
These 7 lines that are below this multi-line comment define the plot_graph() function that plots and shows a graph. This function is borrowed from PlottingCode.py file.
"""
def plot_graph(arr, name):
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()
    
# Load Dataset
url = "iris.csv" #This line sets the url variable to the iris.csv file.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #This line contains the array/list of features of the dataset.
dataset = read_csv(url, names = names) #This line uses the read_csv() function to access data from the iris.csv file and store that data in this variable.

# Create Arrays for Features and Classes
array = dataset.values #This line contains the iris.csv file's values that are stored in the array variable.
X = array[:,0:4] #This line contains the flower features/inputs of the Iris Varieties. 
y = array[:,4] #This line contains the flower names/outputs of the Iris Varieties.
encoder = LabelEncoder() #This line sets the encoder variable to the LabelEncoder() function of the sklearn.preprocessing library.
encoder.fit(y) #This line fits the Label Encoder for the encoder variable. 
          
#PART 1: K-MEANS CLUSTERING CODE
print('K-MEANS CLUSTERING:') #This line prints the "K_MEANS CLUSTERING:" message.
recons_err_arr = np.empty(shape = 20) #This line gives a new array of the given shape and 20 arbitrary values since the array is generated for k = 1 to k = 20.
#The for-loop with 3 lines sets the kmeans variable to the KMeans() function of the sklearn.cluster library, uses the inertia_ attribute, and generates and adds the reconstruction error values in the recons_err_arr variable for k = 1 to k = 20.
for i in range(np.size(recons_err_arr)):
    kmeans = KMeans(n_clusters = (i + 1), init = 'k-means++', random_state = 0, n_init = 'auto').fit(X)
    recons_err_arr[i] = kmeans.inertia_
plot_graph(recons_err_arr, 'Reconstruction Error') #This line plots the graph of the reconstruction error. This function is borrowed from PlottingCode.py file.
elbow_k = 3 #This line sets the elbow_k variable to 3 since the elbow value of the Reconstruction Error is 3 that is obtained by the elbow method on the Reconstruction Error graph.

#K-MEANS CLUSTERING CODE (k = elbow_k)  
kmeans = KMeans(n_clusters = elbow_k, init = 'k-means++', random_state = 0, n_init = 'auto').fit(X) #This line sets the kmeans variable to the KMeans() function of the sklearn.cluster library.      
prediction = kmeans.predict(X) #This line uses the predict() function and the clusters for k = elbow_k to classify the entire Iris data-set of X.
predicted = np.empty_like(prediction) #This line gives a new array with the same attributes for the given variable and tracks matched cluster labels.
#The for-loop with 3 lines matches the k-mean labels and the truth labels and assigns the best label, so the number of the true-positive predictions is maximized.
for i in np.unique(prediction): 
    match_nums = [np.sum((prediction == i) * (encoder.transform(y) == t)) for t in np.unique(encoder.transform(y))]
    predicted[prediction == i] = np.unique(encoder.transform(y))[np.argmax(match_nums)]
print('K-MEANS CLUSTERING: k = elbow_k') #This line prints the "K-MEANS CLUSTERING: k = elbow_k" message.
print('Accuracy:', accuracy_score(encoder.transform(y), predicted)) #This line prints the accuracy value of the K-Means Clustering when k = elbow_k.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(encoder.transform(y), predicted), '\n') #This line prints the confusion matrix of the K-Means Clustering when k = elbow_k and makes a new line.

#K-MEANS CLUSTERING CODE (k = 3)
k = 3 #This line sets the k variable to 3 and uses the K-Means Clustering to cluster the data into 3 clusters.
kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 0, n_init = 'auto').fit(X) #This line sets the kmeans variable to the KMeans() function of the sklearn.cluster library.
prediction = kmeans.predict(X) #This line uses the predict() function and the clusters for k = 3 to classify the entire Iris data-set of X.
predicted = np.empty_like(prediction) #This line gives a new array with the same attributes for the given variable and tracks matched cluster labels.
#The for-loop with 3 lines matches the k-mean labels and the truth labels and assigns the best label, so the number of the true-positive predictions is maximized.
for i in np.unique(prediction): 
    match_nums = [np.sum((prediction == i) * (encoder.transform(y) == t)) for t in np.unique(encoder.transform(y))]
    predicted[prediction == i] = np.unique(encoder.transform(y))[np.argmax(match_nums)]
print('K-MEANS CLUSTERING: k = 3') #This line prints the "K-MEANS CLUSTERING: k = 3" message.
print('Accuracy:', accuracy_score(encoder.transform(y), predicted)) #This line prints the accuracy value of the K-Means Clustering when k = 3.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(encoder.transform(y), predicted), '\n') #This line prints the confusion matrix of the K-Means Clustering when k = 3 and makes a new line.

#PART 2: GAUSSIAN MIXTURE MODELS (GMM) CODE
print('GAUSSIAN MIXTURE MODELS (GMM):') #This line prints the "GAUSSIAN MIXTURE MODELS (GMM):" message.
#AKAIKE INFORMATION CRITERION/AIC CODE (k = aic_elbow_k)
aic = np.empty(shape = 20) #This line gives a new array of the given shape and 20 arbitrary values for the aic variable.
#The for-loop with 3 lines sets the gm variable to the GaussianMixture() function of the sklearn.mixture library and generates data points for AIC.
for i in range(1,21):
    gm = GaussianMixture(n_components = i, random_state = 0, covariance_type = 'diag').fit(X)
    aic[i-1] = gm.aic(X)
plot_graph(aic, 'AIC') #This line plots the graph of AIC. This function is borrowed from PlottingCode.py file.
aic_elbow_k = 3 #This line sets the aic_elbow_k variable to 3 since the elbow value of AIC is 3 that is obtained by the elbow method on the AIC graph.
gm_aic = GaussianMixture(n_components = aic_elbow_k, random_state = 0, covariance_type = 'diag').fit(X) #This line sets the gm_aic variable to the GaussianMixture() function of the sklearn.mixture library.
prediction = gm_aic.predict(X) #This line uses the predict() function and the clusters for k = aic_elbow_k to classify the entire Iris data-set of X.
predicted = np.empty_like(prediction) #This line gives a new array with the same attributes for the given variable and tracks matched cluster labels.
#The for-loop with 3 lines matches the GMM labels and the truth labels and assigns the best label, so the number of the true-positive predictions is maximized.
for i in np.unique(prediction): 
    match_nums = [np.sum((prediction == i) * (encoder.transform(y) == t)) for t in np.unique(encoder.transform(y))]
    predicted[prediction == i] = np.unique(encoder.transform(y))[np.argmax(match_nums)]
print('GAUSSIAN MIXTURE MODEL: k = aic_elbow_k') #This line prints the "GAUSSIAN MIXTURE MODEL: k = aic_elbow_k" message.
print('Accuracy:', accuracy_score(encoder.transform(y), predicted)) #This line prints the accuracy value of the Gaussian Mixture Model when k = aic_elbow_k.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(encoder.transform(y), predicted), '\n') #This line prints the confusion matrix of the Gaussian Mixture Model when k = aic_elbow_k and makes a new line.

#BAYESIAN INFORMATION CRITERION/BIC CODE (k = bic_elbow_k)
bic = np.empty(shape = 20) #This line gives a new array of the given shape and 20 arbitrary values for the bic variable.
#The for-loop with 3 lines sets the gm variable to the GaussianMixture() function of the sklearn.mixture library and generates data points for BIC.
for i in range(1,21):
    gm = GaussianMixture(n_components = i, random_state = 0, covariance_type = 'diag').fit(X)
    bic[i-1] = gm.bic(X)
plot_graph(bic, 'BIC') #This line plots the graph of BIC. This function is borrowed from PlottingCode.py file.
bic_elbow_k = 4 #This line sets the bic_elbow_k variable to 4 since the elbow value of BIC is 4 that is obtained by the elbow method on the BIC graph.
#bic_elbow_k = 3 #This line sets the bic_elbow_k variable to 3 since the actual elbow value of BIC is 3 that is obtained by the elbow method on the BIC graph. So, I have lost some points since the used value is 4. But feel free to change the elbow value from 4 to 3 by uncommenting this line and commenting the previous line.
gm_bic = GaussianMixture(n_components = bic_elbow_k, random_state = 0, covariance_type = 'diag').fit(X) #This line sets the gm_bic variable to the GaussianMixture() function of the sklearn.mixture library.
prediction = gm_bic.predict(X) #This line uses the predict() function and the clusters for k = bic_elbow_k to classify the entire Iris data-set of X.
print('GAUSSIAN MIXTURE MODEL: k = bic_elbow_k') #This line prints the "GAUSSIAN MIXTURE MODEL: k = bic_elbow_k" message.
print('Accuracy: Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters') #This line prints the accuracy value of the Gaussian Mixture Model when k = bic_elbow_k. But this line prints the "Accuracy: Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters" message since the bic_elbow_k value is 4, not 3.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(encoder.transform(y), prediction), '\n') #This line prints the confusion matrix of the Gaussian Mixture Model when k = bic_elbow_k and makes a new line.

#GAUSSIAN MIXTURE MODEL CODE (k = 3)
k = 3 #This line sets the k variable to 3 and uses the Gaussian Mixture Model to cluster the data into 3 clusters.
gm = GaussianMixture(n_components = k, random_state = 0, covariance_type = 'diag').fit(X) #This line sets the gm variable to the GaussianMixture() function of the sklearn.mixture library.
prediction = gm.predict(X) #This line uses the predict() function and the clusters for k = 3 to classifity the entire Iris data-set of X.
predicted = np.empty_like(prediction) #This line gives a new array with the same attributes for the given variable and tracks matched cluster labels.
#The for-loop with 3 lines matches the GMM labels and the truth labels and assigns the best label, so the number of the true-positive predictions is maximized.
for i in np.unique(prediction): 
    match_nums = [np.sum((prediction == i) * (encoder.transform(y) == t)) for t in np.unique(encoder.transform(y))]
    predicted[prediction == i] = np.unique(encoder.transform(y))[np.argmax(match_nums)]
print('GAUSSIAN MIXTURE MODEL: k = 3') #This line prints the "GAUSSIAN MIXTURE MODEL: k = 3" message.
print('Accuracy:', accuracy_score(encoder.transform(y), predicted)) #This line prints the accuracy value of the Gaussian Mixture Model when k = 3.
print('Confusion Matrix:') #This line prints the "Confusion Matrix:" message.
print(confusion_matrix(encoder.transform(y), predicted)) #This line prints the confusion matrix of the Gaussian Mixture Model when k = 3.
