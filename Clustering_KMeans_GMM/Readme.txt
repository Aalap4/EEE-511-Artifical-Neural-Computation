#Midterm2 Competition - Clustering 
#EEE 511 ANC

This project has been implemented on Python3.7 using the numpy,pandas,seaborn and sklearn packages available in open source. 

Include the Dataset "Mall_Customers.csv" and the code "Clustering.py" in the same folder and Run the code. 

The Code will run and generate 3 graphs to get the good choice and that can be verified by looking at the plots.

It will then plot 8 plots. 4 plots for K-Means (4,6,8,10) clusters and 4 plots for (4,6,8,10) clusters.

We have used PCA to find the best features and then found out that Annual Income vs Spending Score so we used that to cluster.

Clustering algorithms such as K-Means and Gaussian Mixture Modelling using Expectation-Maximization  have been implemented to obtain models for 4,6,8 and 10 clusters respectively. 
The resulting cluster plots have been plotted while being validated on the consistency within clusters(K-Means), BIC Penalty(GMM) and Silhoutte Score.

The best clusters has been chosen on the basis of projected cluster centers using K-Means (Elbow method) , Bayesian Information Criterion Penalty and Silhouette Score. 

According to these methods, the best model for the given dataset consists of 6 clusters that can be verified from the plots.

Plots can be viewed in an enlarged image in a folder named Photos.

