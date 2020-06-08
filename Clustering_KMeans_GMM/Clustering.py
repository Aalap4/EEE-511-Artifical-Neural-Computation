##################################
#Libraries to be imported        #
#################################
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn.mixture.gaussian_mixture as GM
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


##################################
#Reading the Dataset and creating#
#a DataFrame                     #
##################################
filename = pd.read_csv('Mall_Customers.csv') # reading the dataset
df = pd.DataFrame(filename) #creating a DataFrame from read dataset
print(df) #displaying above dataframe
#print(df.isnull().sum())
#######################################
#Preparing the data for the clustering#
#algorithms to be implemented         #
#######################################
fig = plt.axes()
sns.heatmap(df.corr(), annot=True,cbar = False)
fig.set_title('Feature Correlation')
plt.show()
#print(X)

df_filtered = pd.get_dummies(df,prefix=["Gender"]) #One-hot encoding for the categorical variable

# We implemented PCA and found out the best features are Annual Income and Spending Score
X_f = df_filtered.iloc[:,1:6]
df1 = df_filtered.iloc[:,1:6]
princi = PCA(n_components = 2) #PCA to reduce dimensionality
X1=princi.fit_transform(X_f) 

pd.set_option('display.max_columns', None)
print(pd.DataFrame(princi.components_,columns=df1.columns,index = ['PC-1','PC-2']))

#On see the results of PC1 AND PC2 we see that Annual Income and Spending Score are the best features
X = df.iloc[:, [3, 4]].values             # Taking Annual Income and Spending Score on the original data and performing Clustering

# For Data Analysis Part 4
X_age_F =  df_filtered.iloc[:,4]
X_SC    =  df_filtered.iloc[:,3]
X_Annual_income   =  df_filtered.iloc[:,2]
plt.scatter(X_SC,X_age_F,color = "red")
plt.xlabel("Spending Score")
plt.ylabel("Male/Female")
plt.show()
plt.scatter(X_SC,X_Annual_income,color = "blue")
plt.xlabel("Spending Score")
plt.ylabel("Annual income")
plt.show()


#print(components)
########################################
#Implementing K-Means and GMM to obtain#
#4,6,8,10 clusters respectively        #
########################################
K = [4,6,8,10]
sil= []
wcss = [] #list to store the sum of squared distances to the cluster center
bics = [] #ist to store BIC penalty
for j in K:
    kmeans = KMeans(n_clusters = j, init = 'k-means++', max_iter = 1000, n_init = 10,random_state = 0 ,verbose = False,tol = 1e-4) # K-Means Algorithm
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X) # Prediction using K-Means
        #print(kmeans.labels_)
    a=silhouette_score(X, kmeans.labels_)
    sil.append(a)
    wcss.append(kmeans.inertia_) #appending the sum of squared distances to the cluster center
    gmm = GM.GaussianMixture(n_components=j, init_params='kmeans', max_iter=1000, covariance_type='full', tol=1e-04,random_state=0) # Gaussian Mixture Modelling
    gmm.fit(X)
    bics.append(gmm.bic(X)) # updating the BIC list

index_gmm = np.argmin(bics) # index of minimum BIC penalty
index_sil = np.argmax(sil)
#Plotting cost vs number of clusters
plt.figure("K-Means Clustering Analysis using Elbow Method")
plt.plot(K, wcss,'go--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()
#plt.show()

#Implementing Silhouette Score Method
plt.plot(K,sil,'go--')
plt.plot(K[index_sil],sil[index_sil],'ro')
plt.title('Silhouette score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Instantiate a scikit-learn K-Means model

#Plotting BIC penalty vs No. of clusters
plt.figure("Gaussian Mixture Modelling Analysis using BIC Penalty")        
plt.plot(K, bics,'go--')
plt.plot(K[index_gmm],bics[index_gmm],'ro')
plt.title('BIC Penalty')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.show()

######################################################
#Implementation of K-Means and GMM for the best model#
######################################################
X = np.array(X)
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 1000, n_init = 10)
y_kmeans = kmeans.fit_predict(X)


plt.figure("K-Means Clustering with 4 clusters")
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='orange', marker='+',edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='blue', marker='*', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='black', marker='o', edgecolor='black',
                    label='cluster 4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', marker='*', edgecolor='red',label='centroid')
plt.title('Kmeans 4')
plt.legend()
plt.show()

X = np.array(X)
kmeans = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 1000, n_init = 10)
y_kmeans = kmeans.fit_predict(X)

#Plotting the K-Means for 6 clusters
plt.figure("K-Means Clustering with 6 clusters")
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='orange', marker='+',edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='blue', marker='x', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='black', marker='o', edgecolor='black',
                    label='cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c='brown', marker='*', edgecolor='black',
                    label='cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=50, c='pink',marker='v', edgecolor='black',label='cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', marker='*', edgecolor='red',label='centroid')
plt.title('Kmeans 6')
plt.legend()
plt.show()

X = np.array(X)
kmeans = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 1000, n_init = 10)
y_kmeans = kmeans.fit_predict(X)

plt.figure("K-Means Clustering with 8 clusters")
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='orange', marker='+',edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='blue', marker='x', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='black', marker='o', edgecolor='black',
                    label='cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c='brown', marker='*', edgecolor='black',
                    label='cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=50, c='pink',marker='v', edgecolor='black',label='cluster 6')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s=50, c='yellow',marker='>', edgecolor='black',label='cluster 7')
plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s=50, c='darkgreen',marker='<', edgecolor='black',label='cluster 8')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', marker='*', edgecolor='red',label='centroid')
plt.title('Kmeans 8')
plt.legend()
plt.show()

X = np.array(X)
kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 1000, n_init = 10)
y_kmeans = kmeans.fit_predict(X)

plt.figure("K-Means Clustering with 10 clusters")
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='orange', marker='+',edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='blue', marker='x', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='black', marker='o', edgecolor='black',
                    label='cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c='brown', marker='*', edgecolor='black',
                    label='cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=50, c='pink',marker='v', edgecolor='black',label='cluster 6')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s=50, c='yellow',marker='>', edgecolor='black',label='cluster 7')
plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s=50, c='darkgreen',marker='<', edgecolor='black',label='cluster 8')
plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s=50, c='purple',marker='^', edgecolor='black',label='cluster 9')
plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9, 1], s=50, c='violet',marker='s', edgecolor='black',label='cluster 10')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', marker='*', edgecolor='red',label='centroid')
plt.title('Kmeans 10')
plt.legend()
plt.show()

gmm = GM.GaussianMixture(n_components=4, init_params='kmeans', max_iter=1000, covariance_type='full',
                              tol=1e-04,
                              random_state=0)
y_gmm = gmm.fit_predict(X)

#Plotting GMM with 4 clusters 
plt.figure("Gaussian Mixture Modelling with 4 components")
plt.scatter(X[y_gmm == 0, 0], X[y_gmm == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_gmm == 1, 0], X[y_gmm == 1, 1], s=50, c='orange', marker='v', edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_gmm == 2, 0], X[y_gmm == 2, 1], s=50, c='blue', marker='x', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_gmm == 3, 0], X[y_gmm == 3, 1], s=50, c='black', marker='*', edgecolor='black',
                    label='cluster 4')
plt.title('GMM 4')
plt.legend()
plt.show()

gmm = GM.GaussianMixture(n_components=6, init_params='kmeans', max_iter=1000, covariance_type='full',
                              tol=1e-04,
                              random_state=0)
y_gmm = gmm.fit_predict(X)

#Plotting GMM with 6 clusters 
plt.figure("Gaussian Mixture Modelling with 6 components")
plt.scatter(X[y_gmm == 0, 0], X[y_gmm == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_gmm == 1, 0], X[y_gmm == 1, 1], s=50, c='orange', marker='v', edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_gmm == 2, 0], X[y_gmm == 2, 1], s=50, c='blue', marker='x', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_gmm == 3, 0], X[y_gmm == 3, 1], s=50, c='black', marker='*', edgecolor='black',
                    label='cluster 4')
plt.scatter(X[y_gmm == 4, 0], X[y_gmm == 4, 1], s=50, c='red',marker='o', edgecolor='black', label='cluster 5')
plt.scatter(X[y_gmm == 5, 0], X[y_gmm == 5, 1], s=50, c='yellow',marker = '>', edgecolor='black', label='cluster 6')
plt.title('GMM 6')
plt.legend()
plt.show()

gmm = GM.GaussianMixture(n_components=8, init_params='kmeans', max_iter=1000, covariance_type='full',
                              tol=1e-04,
                              random_state=0)
y_gmm = gmm.fit_predict(X)

#Plotting GMM with 8 clusters 
plt.figure("Gaussian Mixture Modelling with 8 clusters")
plt.scatter(X[y_gmm == 0, 0], X[y_gmm == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_gmm == 1, 0], X[y_gmm == 1, 1], s=50, c='orange', marker='+',edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_gmm == 2, 0], X[y_gmm == 2, 1], s=50, c='blue', marker='x', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_gmm == 3, 0], X[y_gmm == 3, 1], s=50, c='black', marker='o', edgecolor='black',
                    label='cluster 4')
plt.scatter(X[y_gmm == 4, 0], X[y_gmm == 4, 1], s=50, c='brown', marker='*', edgecolor='black',
                    label='cluster 5')
plt.scatter(X[y_gmm == 5, 0], X[y_gmm == 5, 1], s=50, c='pink',marker='v', edgecolor='black',label='cluster 6')
plt.scatter(X[y_gmm == 6, 0], X[y_gmm == 6, 1], s=50, c='yellow',marker='>', edgecolor='black',label='cluster 7')
plt.scatter(X[y_gmm == 7, 0], X[y_gmm == 7, 1], s=50, c='darkgreen',marker='<', edgecolor='black',label='cluster 8')
plt.title('GMM 8')
plt.legend()
plt.show()

gmm = GM.GaussianMixture(n_components=10, init_params='kmeans', max_iter=1000, covariance_type='full',
                              tol=1e-04,
                              random_state=0)
y_gmm = gmm.fit_predict(X)

#Plotting GMM with 10 clusters 
plt.figure("Gaussian Mixture Modelling with 10 clusters")
plt.scatter(X[y_gmm == 0, 0], X[y_gmm == 0, 1], s=50, c='lightgreen', marker='.', edgecolor='black',
                    label='cluster 1')
plt.scatter(X[y_gmm == 1, 0], X[y_gmm == 1, 1], s=50, c='orange', marker='+',edgecolor='black',
                    label='cluster 2')
plt.scatter(X[y_gmm == 2, 0], X[y_gmm == 2, 1], s=50, c='blue', marker='x', edgecolor='black',
                    label='cluster 3')
plt.scatter(X[y_gmm == 3, 0], X[y_gmm == 3, 1], s=50, c='black', marker='o', edgecolor='black',
                    label='cluster 4')
plt.scatter(X[y_gmm == 4, 0], X[y_gmm == 4, 1], s=50, c='brown', marker='*', edgecolor='black',
                    label='cluster 5')
plt.scatter(X[y_gmm == 5, 0], X[y_gmm == 5, 1], s=50, c='pink',marker='v', edgecolor='black',label='cluster 6')
plt.scatter(X[y_gmm == 6, 0], X[y_gmm == 6, 1], s=50, c='yellow',marker='>', edgecolor='black',label='cluster 7')
plt.scatter(X[y_gmm == 7, 0], X[y_gmm == 7, 1], s=50, c='darkgreen',marker='<', edgecolor='black',label='cluster 8')
plt.scatter(X[y_gmm == 8, 0], X[y_gmm == 8, 1], s=50, c='purple',marker='^', edgecolor='black',label='cluster 9')
plt.scatter(X[y_gmm == 9, 0], X[y_gmm == 9, 1], s=50, c='violet',marker='s', edgecolor='black',label='cluster 10')
plt.title('GMM 10')
plt.legend()
plt.show()
#---- End of Code ----#
