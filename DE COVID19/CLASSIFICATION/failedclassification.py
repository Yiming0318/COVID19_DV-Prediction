

'''
Import modules
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math 
import os
import matplotlib.pyplot as plt

'''
Generate data for analysis
'''
filename = os.listdir('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/')
sheet_name = [i.replace('.csv', '') for i in filename if i.endswith('.csv')]
dfs = {sheet_name: pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/{}.csv'.format(sheet_name))
          for sheet_name in sheet_name}

DailyData = dfs[sheet_name[0]][['date_day']]
for region in sheet_name:
    #DailyData[region] = dfs[region]['New']
    DailyData.loc[:, region] = dfs[region].loc[:,'New']
DailyData.set_index('date_day', inplace=True)
DailyData = DailyData.sort_index()
# Remove the latest day as it is not compeleted
DailyData = DailyData.drop(DailyData.iloc[-1].name)

DailyData['China']
DailyData.to_csv('F:/COVID19DV/DailyData.csv', index=True)


'''
Use fixed data
'''
DailyData = pd.read_csv('./DailyData.csv', index_col=0)
DailyData




# include all the sleceted countries 
# Data transformation to reduce the effect of data scale on pattern identification
# Square root transformation
DailyDataTrans = DailyData**0.5

# Normalisation column-wise
from sklearn import preprocessing

x = DailyDataTrans.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
DailyDataNorm = pd.DataFrame(x_scaled)
DailyDataNorm.columns = DailyData.columns

fig = plt.figure(figsize=(16,12), dpi=200, constrained_layout=True)

axs = fig.subplots(nrows=2, ncols=7)

for i in range(len(DailyDataNorm.columns)):
    axs.flat[i].plot(DailyDataNorm.index, DailyDataNorm.iloc[:,i], color='black')
    axs.flat[i].get_xaxis().set_ticks([])
    axs.flat[i].get_yaxis().set_ticks([])
    axs.flat[i].annotate(DailyDataNorm.iloc[:,i].name, (0.05, 0.8),xycoords='axes fraction', va='center', ha='left')
    


'''
PCA & LDA SVM CLASSIFY; RF CLASSIFY
'''
df = pd.read_excel("F:/COVID19DV/dash-2019-coronavirus/ALL.xlsx")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
out_pca = pca.fit_transform(df[['Confirmed',
                                'Recovered',
                                'Deaths']])

df_pca = pd.DataFrame(data = out_pca, columns = ['pca1', 'pca2'])
print(df_pca.head())

# This looks good, but we are missing the target or label column (species). 
# Let's add the column by concatenating with the original DataFrame. 
# This gives us a PCA DataFrame (df_pca) ready for downstream work and predictions. 
# Then, let's plot it and see what our transformed data looks like in two dimensions.

df_pca = pd.concat([df_pca, df[['Country']]], axis = 1)
print(df_pca.head())
sns.lmplot(x="pca1", y="pca2", hue="Country", data=df_pca, fit_reg=False)

#reduce dimensions with LDA - nothing else than a PCA with labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)

# format dataframe
out_lda = lda.fit_transform(X=df.iloc[:,:3], y=df['Country'])
df_lda = pd.DataFrame(data = out_lda, columns = ['lda1', 'lda2'])
df_lda = pd.concat([df_lda, df[['Country']]], axis = 1)

# sanity check
print(df_lda.head())

# plot
sns.lmplot(x="lda1", y="lda2", hue="Country", data=df_lda, fit_reg=False)

# The goal of PCA is to orient the data in the direction of the greatest variation. 
# However, it ignores some important information from our dataset – for instance, the labels are not used; 
# in some cases, we can extract even better transformation vectors if we include the labels. 
# The most popular labeled dimension-reduction technique is called linear discriminant analysis (LDA). 

# 10. comparison PCA vs. LDA - run separately 
sns.violinplot(x='Country',y='pca1', data=df_pca).set_title("Violin plot: Feature = PCA_1")
sns.violinplot(x='Country',y='lda1', data=df_lda).set_title("Violin plot: Feature = LDA_1")

# 11. k-means clustering and the silhouette score
# cluster With k-means and check silhouette score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# initialize k-means algo object
kmns = KMeans(n_clusters=3, random_state=42)

# fit algo to PCA and find silhouette score
out_kms_pca = kmns.fit_predict(out_pca)
silhouette = silhouette_score(out_pca, out_kms_pca)
print("PCA silhouette score = " + str(silhouette))

# fit algo to LDA and find silhouette score
out_kms_lda = kmns.fit_predict(out_lda)
silhouette = silhouette_score(out_lda, out_kms_lda)
print("LDA silhouette score = %2f " % silhouette)

# 12. making decisions
# IMPORTANT NOTE: 
# Before we make a decision, we need to separate our data into training and test sets. 
# Model validation is a large and very important topic that will be covered later, 
# but for the purpose of this end-to-end example, we will do a basic train-test split. 
# We will then build the decision model on the training data 
# and score it on the test data using the F1 score. 
# I recommend using a random seed for the most randomized data selection. 
# This seed tells the pseudo-random number generator where to begin its randomization routine. 
# The result is the same random choice every time. 
# In this example, I've used the random seed 42 when splitting into test and training sets. 
# Now, if I stop working on the project and pick it back up later, 
# I can split with the random seed and get the exact same training and test sets.

# Split into train/validation/test set
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_lda, test_size=0.3, random_state=42)

# Sanity check
print('train set shape = ' + str(df_train.shape))
print('test set shape = ' + str(df_test.shape))
print(df_train.head())

# classify with SVM
from sklearn.svm import SVC
from sklearn.metrics import f1_score
clf = SVC(kernel='rbf', C=0.8, gamma=10)
# C is a penalty term and is called a hyperparameter; this means that it is 
# a setting that an analyst can use to steer a fit in a certain direction. 
clf.fit(df_train[['lda1', 'lda2']], df_train['Country'])

# predict on test set
y_pred = clf.predict(df_test[['lda1', 'lda2']])
f1 = f1_score(df_test['Country'], y_pred, average='weighted')

# check prediction score
print("f1 score for SVM classifier = %2f " % f1)

# IMPORTANT NOTE:
# C is the penalty term in an SVM. It controls how large the penalty is 
# for a mis-classed example internally during the model fit. 
# For a utilitarian understanding, it is called the soft margin penalty 
# because it tunes how hard or soft the resulting separation line is drawn. 
# Common hyperparameters for SVMs will be covered in more detail later.

#  Let's change it from 0.8 to 1, which will effectively raise the penalty term.

# classify with SVM
from sklearn.svm import SVC
from sklearn.metrics import f1_score
clf = SVC(kernel='rbf', C=1, gamma=10)
clf.fit(df_train[['lda1', 'lda2']], df_train['Country'])
y_pred = clf.predict(df_test[['lda1', 'lda2']])
f1 = f1_score(df_test['Country'], y_pred, average='weighted')
print("f1 score for SVM classifier = %2f " % f1)

# The F1 score for this classifier is now 0.85. 
# The obvious next step is to tune the parameters and maximize the F1 score. 
# Of course, it will be very tedious to change a parameter (refit, analyze, and repeat). 
# Instead, you can employ a grid search to automate this parameterization. 
# Grid search and cross-validation will be covered in more detail later. 
# An alternative method to employing a grid search is to choose an algorithm that doesn't require tuning. 
# A popular algorithm that requires little-to-no tuning is Random Forest. 
# The forest refers to how the method adds together multiple decision trees into a voted prediction.

# classify with RF
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=2, random_state=42)
clf.fit(df_train[['lda1', 'lda2']], df_train['Country'])
y_pred = clf.predict(df_test[['lda1', 'lda2']])
f1 = f1_score(df_test['Country'], y_pred, average='weighted')

# check prediction score
print("f1 score for SVM classifier = %2f " % f1)



###############
#Kmean Cluster#
###############

'''
EAST
'''

df = pd.read_excel("F:/COVID19DV/dash-2019-coronavirus/East.xlsx")
df = df[df.columns[0:2]]
# import module and instantiate K-means object
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=5, tol=0.004, max_iter=300)

# fit to input data
clus.fit(df)

# get cluster assignments of input data and print first five labels
df['K-means Cluster Labels'] = clus.labels_
print(df['K-means Cluster Labels'][:5].tolist())

# Now, let's use Seaborn's scatter plot to visualize 
# the grouping of a blob set with the cluster labels displayed
sns.lmplot(x='Confirmed', y='Recovered', 
           hue="K-means Cluster Labels", data=df, fit_reg=False)



# finding k with silhouette

# find best value for k using silhouette score
# import metrics module
from sklearn import metrics

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for k in n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
    cluster_labels = kmeans.predict(df)
    S = metrics.silhouette_score(df, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(k, S))



'''
WEST
'''

df = pd.read_excel("F:/COVID19DV/dash-2019-coronavirus/West.xlsx")
df = df[df.columns[0:2]]
# import module and instantiate K-means object
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=5, tol=0.004, max_iter=300)

# fit to input data
clus.fit(df)

# get cluster assignments of input data and print first five labels
df['K-means Cluster Labels'] = clus.labels_
print(df['K-means Cluster Labels'][:5].tolist())

# Now, let's use Seaborn's scatter plot to visualize 
# the grouping of a blob set with the cluster labels displayed
sns.lmplot(x='Confirmed', y='Recovered', 
           hue="K-means Cluster Labels", data=df, fit_reg=False)



# finding k with silhouette

# find best value for k using silhouette score
# import metrics module
from sklearn import metrics

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for k in n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
    cluster_labels = kmeans.predict(df)
    S = metrics.silhouette_score(df, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(k, S))




'''
Developed
'''
df = pd.read_excel("F:/COVID19DV/dash-2019-coronavirus/Developed.xlsx")
df = df[df.columns[0:2]]
# import module and instantiate K-means object
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=5, tol=0.004, max_iter=300)

# fit to input data
clus.fit(df)

# get cluster assignments of input data and print first five labels
df['K-means Cluster Labels'] = clus.labels_
print(df['K-means Cluster Labels'][:5].tolist())

# Now, let's use Seaborn's scatter plot to visualize 
# the grouping of a blob set with the cluster labels displayed
sns.lmplot(x='Confirmed', y='Recovered', 
           hue="K-means Cluster Labels", data=df, fit_reg=False)



# finding k with silhouette

# find best value for k using silhouette score
# import metrics module
from sklearn import metrics

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for k in n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
    cluster_labels = kmeans.predict(df)
    S = metrics.silhouette_score(df, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(k, S))



'''
Developing
'''
df = pd.read_excel("F:/COVID19DV/dash-2019-coronavirus/Developing.xlsx")
df = df[df.columns[0:2]]
# import module and instantiate K-means object
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=5, tol=0.004, max_iter=300)

# fit to input data
clus.fit(df)

# get cluster assignments of input data and print first five labels
df['K-means Cluster Labels'] = clus.labels_
print(df['K-means Cluster Labels'][:5].tolist())

# Now, let's use Seaborn's scatter plot to visualize 
# the grouping of a blob set with the cluster labels displayed
sns.lmplot(x='Confirmed', y='Recovered', 
           hue="K-means Cluster Labels", data=df, fit_reg=False)



# finding k with silhouette

# find best value for k using silhouette score
# import metrics module
from sklearn import metrics

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for k in n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
    cluster_labels = kmeans.predict(df)
    S = metrics.silhouette_score(df, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(k, S))
    
    

    
    

