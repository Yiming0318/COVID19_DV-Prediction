# Yiming Ge
"""
Based on the latest 5.2 six countries data to do the classification
"""

'''
PCA 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel("F:/COVID19DV/dash-2019-coronavirus/5.2latest_data.xlsx")
df.head

"""
DO THE PCA
"""
#Use PCA to reduce dimension
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
variables = ['Confirmed','Recovered','Deaths']
df[variables] = scaler.fit_transform(df[variables])
X = df.drop(['Country'],axis=1)
y = df['Country']
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized')
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=2)
df_pca = pca_final.fit_transform(X)
df_pca = pd.DataFrame(df_pca)



"""
Hierarchical Clustering
"""
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
mergings = linkage(df_pca, method = "complete", metric='euclidean')
dendrogram(mergings)
plt.show()
#based on the graph, we can do 2 cluster divsion and also 4 cluster division 
#4 cluster
clusterCut = pd.Series(cut_tree(mergings, n_clusters = 4).reshape(-1,))
df_pca_hierarchical = pd.concat([df_pca, clusterCut], axis=1)
df_pca_hierarchical.columns = ["PC1","PC2","ClusterID"]
df_pca_hierarchical.head()
pca_cluster_hierarchical = pd.concat([df['Country'],df_pca_hierarchical], axis=1)
print(pca_cluster_hierarchical)
#2 cluster
clusterCut = pd.Series(cut_tree(mergings, n_clusters = 2).reshape(-1,))
df_pca_hierarchical = pd.concat([df_pca, clusterCut], axis=1)
df_pca_hierarchical.columns = ["PC1","PC2","ClusterID"]
pca_cluster_hierarchical = pd.concat([df['Country'],df_pca_hierarchical], axis=1)
print(pca_cluster_hierarchical)



