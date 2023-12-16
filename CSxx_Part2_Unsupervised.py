#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:34:07 2020

@author: mihalis
"""

"""Unsupervised Learning"
print("Unsupervised Learning")"""

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
import pandas as pd
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn import cluster


print("read data")
br_data0 = pd.read_csv("br_data.csv")

print("show first and last 10 rows")
print(br_data0.head(10))  # check first 10 rows
print(br_data0.tail(10))
print("show shape of data")
print(br_data0.shape)
print("columns")
print(br_data0.columns)
print("checking for nulls")
print(br_data0.isnull())  # check if there are missing values.Returns Boolean [False=no null, True=null]
print(br_data0.isnull().any())  # returns which columns are intact and which not


print("Dropping column Id and slicing data prior to Clustering. Then work only with the 10 features and Iexclude the  columns with the statistics")
     
br_data1 = br_data0.drop("id", axis=1)
br_data1.head()
print(br_data1.shape)

subset1= br_data1.iloc[:, [ 21, 22,23,24,25,26,27,28,29,30]]
print(subset1.columns)
print(subset1.shape)

print("I set the diagnois columns as my target")
Y= br_data1.iloc[:, 0]

print("Y is" , Y.head)
print("Dropped column-Sliced Data Complete")


print("Maybe K-means will work this time")

print("Convert diagnosis column into numerical data for downstream clustering analysis")
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y)
print("Diagnosis column after encoding",Y1)


print("Scale data and find unique values of Y that we will use as k ")
from sklearn.preprocessing import scale

scaled_data1 = scale(subset1)
n_samples, n_features = br_data1.shape
n_digits = len(np.unique(Y1)) #It asks how many unique numbers== values you have in the dataset and uses that as n clusters"
print(n_digits)


print("Subset1: K-means clustering against Diagnosis for the 10 statistical significant variables")
kmeans = cluster.KMeans(n_clusters=2) #I set maually the N to 2 because there are two categories, benign and malignant
kmeans_model1 = kmeans.fit(scaled_data1)
print("K-means labels", kmeans.labels_, "k-means centroids", kmeans.cluster_centers_)
print("Subset1 : kmeans model", kmeans_model1)
print("silhouette_score = ", metrics.silhouette_score(subset1, kmeans.labels_))
print("completeness_score = ", metrics.completeness_score(Y1, kmeans.labels_))
print("homogeneity_score = ", metrics.homogeneity_score(Y1, kmeans.labels_))


print("Scatter plot showing clustering of Classification between Benign and Tumour")
plt.figure(figsize=(12,12))
plt.scatter(scaled_data1[:,0], scaled_data1[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='red')
plt.title("K-means Classification Against Diagnosis")
plt.grid(True)
plt.show()


print(" K-means calculation for different number of clusters and iterations. It takes sometime to run!")
print("I am now using statistically significant dataset trying to tune it.Somehow it underperforms the out of the box calculation of above")



result = []

for l in range(300,600):
    for i in range(2,10):
            model = cluster.KMeans(n_clusters = i, max_iter=l)
            model.fit(scaled_data1)
            result.append([l,i,metrics.silhouette_score(scaled_data1, model.labels_), metrics.completeness_score(Y1, model.labels_), metrics.homogeneity_score(Y1, model.labels_)])
print("results" , result) 
Result_big = pd.DataFrame(result, columns= ["No of iteration","No of Cluster", "Silhouette_score","Completeness_score","Homogeneity_score"]) 
print(Result_big) 
          

maxI = -1
maxV = 0
for i in range(0,len(result)):
  #print(result[i])
  if(result[i][2]>maxV):
    maxV = result[i][2]
    maxI = i
print("Max silhouette_score: ", result[maxI])




print("Plotting the Metrics for K-means")

plt.figure(figsize=(13,5))
plt.plot(Result_big["No of Cluster"], Result_big["Silhouette_score"], "go", label='Silhouette score')
plt.plot(Result_big["No of Cluster"], Result_big["Completeness_score"], "bo",label='Completeness score' )
plt.plot(Result_big["No of Cluster"], Result_big["Homogeneity_score"], "ro", label='Homogeneity score')
plt.grid(True)
plt.legend()
plt.xlabel('Number of clusters')
plt.title('Score Results of varying clusters by using K-Means ')
plt.show()










print("Performing Agglomerative clustering testing different conditions")
print("Agglomerative Custering")
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"] 
result1 = []
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
           continue
        else:
            modelA = cluster.AgglomerativeClustering(n_clusters=2, linkage=l, affinity=a)
            modelA.fit(scaled_data1)
            result1.append([a,l,metrics.silhouette_score(scaled_data1, modelA.labels_),metrics.completeness_score(Y1, modelA.labels_),metrics.homogeneity_score(Y1, modelA.labels_)])
Result_big1 = pd.DataFrame(result1, columns= ["Affinity","Linkage", "Silhouette_score","Completeness_score","Homogeneity_score"]) 
maxI = -1
maxV = 0
for i in range(0,len(result1)):
  print(result1[i])
  if(result1[i][2]>maxV):
    maxV = result1[i][2]
    maxI = i
print("Max silhouette_score: ", result1[maxI])
maxI = -1
maxV = 0










print("Performing PCA for the 10 features of our data")

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca_result = pca.fit_transform(scaled_data1)
df_pca = pd.DataFrame(columns = ['pc1','pc2', "pc3", "pc4", "pc5", "pc6"])
df_pca['pc1'] = pca_result[:,0]
df_pca['pc2'] = pca_result[:,1]
df_pca["pc3"] = pca_result[:,2]
df_pca["pc4"] = pca_result[:,3]
df_pca["pc5"] = pca_result[:,4]
df_pca["pc6"] = pca_result[:,5]
pca_variance = pd.DataFrame(pca.explained_variance_ratio_)

print ('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

print("Performing PCA and plotting its scatter plot ")

import matplotlib.patheffects as PathEffects

def fashion_viz(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
        txts.append(txt)

    return


df_pc_values = df_pca[['pc1','pc2']]       # Create a variable with the first and second PCs

print("Visualizing PCA result in scatter plot")
fashion_viz(df_pc_values.values, Y1)



print("Performing t-SNE and plotting scatter plot")
from sklearn.manifold import TSNE
rand_seed = 17
fashion_tsne = TSNE(random_state=rand_seed, n_iter=500).fit_transform(scaled_data1)
fashion_viz(fashion_tsne, Y1)



print ("Performing dimensionality reduction with PCA and then t-SNE ")



from sklearn.manifold import TSNE
pca_result = pca.fit_transform(scaled_data1)
random_seed =17
fashion_tsne = TSNE(n_components=2, n_iter=500, random_state=random_seed).fit_transform(pca_result)
fashion_viz(fashion_tsne, Y1)






print("Performing t-SNE in all the data set,32 columns to see how clusterint changes with all variables")

from sklearn.preprocessing import scale

br_data2=br_data1.drop("diagnosis", axis =1)
scaled_data2 = scale(br_data2)
n_samples, n_features = br_data1.shape
n_digits = len(np.unique(Y1)) #It asks how many unique numbers== values you have in the dataset and uses that as n clusters"
print(n_digits)




from sklearn.manifold import TSNE

rand_seed = 17
fashion_tsne = TSNE(n_components=2, n_iter=500, random_state=rand_seed).fit_transform(scaled_data2)
fashion_viz(fashion_tsne, Y1)


















