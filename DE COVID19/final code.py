#DE 
#GROUP 1
#Saily Jog, Kexin Li, Yao Zhang, Yiming Ge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
###########
####EDA####
###########

from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
import pip
pip.main(['install','plotly'])
full_table = pd.read_csv('./covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
full_table.sample(6)
full_table.info()
full_table.isna().sum()
full_table.shape

ship_rows = full_table['Province/State'].str.contains('Grand Princess') | full_table['Province/State'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('MS Zaandam')

# ship
ship = full_table[ship_rows]

# full table 
full_table = full_table[~(ship_rows)]

# Latest cases from the ships
ship_latest = ship[ship['Date']==max(ship['Date'])]

ship_latest.head()
ship_latest.shape

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']
full_table.head()

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)
full_table['Recovered'] = full_table['Recovered'].astype(int)
full_table.sample(6)

full_grouped = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

# new cases ======================================================
temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)
temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan
temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']
temp.head()

full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])
full_grouped = full_grouped.fillna(0)
cols = ['New cases', 'New deaths', 'New recovered']
full_grouped[cols] = full_grouped[cols].astype('int')
full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)

full_grouped.head()

day_wise = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()
day_wise['No. of countries'] = full_grouped[full_grouped['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len).values
day_wise.head()
country_wise = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)
country_wise = country_wise.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()
country_wise.head()

import plotly.express as px
temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case', height=600,
             title='Cases over time')
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()

temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case', height=600,
             title='Cases over time')
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_c = px.bar(day_wise, x="Date", y="Confirmed")
fig_d = px.bar(day_wise, x="Date", y="Deaths")

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,
                    subplot_titles=('Confirmed cases', 'Deaths reported'))

fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.update_layout(height=480)
fig.show()

fig_c = px.bar(day_wise, x="Date", y="New cases")
fig_d = px.bar(day_wise, x="Date", y="No. of countries")

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,
                    subplot_titles=('No. of new cases everyday', 'No. of countries'))

fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.update_layout(height=480)
fig.show()

fig_c = px.bar(country_wise.sort_values('Confirmed').tail(15), x="Confirmed", y="Country/Region", 
               text='Confirmed', orientation='h')
fig_d = px.bar(country_wise.sort_values('Deaths').tail(15), x="Deaths", y="Country/Region", 
               text='Deaths', orientation='h',color_discrete_sequence = ['red'])

# recovered - active
fig_r = px.bar(country_wise.sort_values('Recovered').tail(15), x="Recovered", y="Country/Region", 
               text='Recovered', orientation='h',color_discrete_sequence = ['green'])
fig_a = px.bar(country_wise.sort_values('Active').tail(15), x="Active", y="Country/Region", 
               text='Active', orientation='h', color_discrete_sequence = ['yellow'])
fig_c.show()
fig_d.show()
fig_r.show()
fig_a.show()

################
#failurecluster#
################
'''
PCA & LDA SVM CLASSIFY; RF CLASSIFY
'''
df = pd.read_excel("./ALL.xlsx")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
out_pca = pca.fit_transform(df[['Confirmed',
                                'Recovered',
                                'Deaths']])

df_pca = pd.DataFrame(data = out_pca, columns = ['pca1', 'pca2'])
print(df_pca.head())


colnames = df['Country']
pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':colnames})
pcs_df.head()

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

df = pd.read_excel("./East.xlsx")
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

df = pd.read_excel("./West.xlsx")
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
df = pd.read_excel("./Developed.xlsx")
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
df = pd.read_excel("./Developing.xlsx")
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
    

######################
##Successful Cluster##
######################
"""
Based on the latest 5.2 six countries data to do the classification
"""

'''
PCA 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel("./5.2latest_data.xlsx")
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

################
#NEW CASE TREND#
################
'''
Import modules
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

'''
Generate data for analysis
'''
filename = os.listdir('./cumulative_data/')
sheet_name = [i.replace('.csv', '') for i in filename if i.endswith('.csv')]
dfs = {sheet_name: pd.read_csv('./cumulative_data/{}.csv'.format(sheet_name))
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
DailyData.to_csv('./DailyData.csv', index=True)


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

##################
#CASES prediction#
##################

#logistic regression predict
from scipy.optimize import curve_fit
import urllib
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os


'''
Generate data for analysis
'''
# df = pd.read_csv('./China.csv')
# df = pd.read_csv('./Iran.csv')
# df = pd.read_csv('./Japan.csv')
# df = pd.read_csv('./Philippines.csv')
# df = pd.read_csv('./South Korea.csv')
# df = pd.read_csv('./Thailand.csv')
# df = pd.read_csv('./UK.csv')
"""
prediction
"""
def logistic_function(t, K, P0, r):
    t0 = 0
    exp = np.exp(r * (t - t0))
    return (K * exp * P0) / (K + (exp - 1) * P0)

def predict():
    # prediction days
    predict_days = 20
    date, confirm = df['date_day'].values, df['Confirmed'].values
    x = np.arange(len(confirm))
    # date_labels = get_date_list(4)
    # least squres curve fit
    popt, pcov = curve_fit(logistic_function, x, confirm)
    print(popt)
 

    predict_x = list(x) + [x[-1] + i for i in range(1, 1 + predict_days)]
    predict_x = np.array(predict_x)
    predict_y = logistic_function(predict_x, popt[0], popt[1], popt[2])
 
    #plot
    plt.figure(figsize=(15, 8))
    plt.plot(x, confirm, 's',label="confimed infected number")
    plt.plot(predict_x, predict_y, 's',label="predicted infected number")
 
    plt.suptitle("Logistic Fitting Curve for 2019-nCov infected numbers(Max = {},  r={:.3})".format(int(popt[0]), popt[2]), fontsize=16, fontweight="bold")
    plt.xlabel('days from 2020/1/21', fontsize=14)
    plt.ylabel('infected number', fontsize=14)
    plt.plot()
    
predict()    



#polynomial prediction

#df = pd.read_csv('./US.csv')
 
 
def f_3(x, A, B, C, D):  
    return A*x*x*x + B*x*x + C*x + D
 

def predict():
    # prediction days
    predict_days = 20
    date, confirm = df['date_day'].values, df['Confirmed'].values
    x = np.arange(len(confirm))
    popt, pcov = curve_fit(f_3, x, confirm)
    print(popt)

    predict_x = list(x) + [x[-1] + i for i in range(1, 1 + predict_days)]
    predict_x = np.array(predict_x)
    predict_y = f_3(predict_x, popt[0], popt[1], popt[2],popt[3])
    #plot
    plt.figure(figsize=(15, 8))
    plt.plot(x, confirm, 's',label="confimed infected number")
    plt.plot(predict_x, predict_y, 's',label="predicted infected number")
 
    plt.suptitle("Polynomial Fitting Curve for 2019-nCov infected numbers", fontsize=16, fontweight="bold")
    plt.xlabel('days from 2020/1/21', fontsize=14)
    plt.ylabel('infected number', fontsize=14)
    plt.plot()
    
predict()




#####################
#economic prediction#
#####################
import numpy as np
import pandas as pd
from sklearn import linear_model

''' read all the data set we have, And the data set is pre_selected with 
    all target countries: CHN,DEU,JPN,MEX,TUR,USA. Since GNI is yearly basis,
    and GDP is quarterly basis, we need to expand the data set into monthly basis,
    and combine all data into one dataset,in order for the following prediction.
'''
# read data set
xls = pd.ExcelFile('./CPI Data.xlsx')
xls.sheet_names
df1 = pd.read_excel(xls,"CPI")
df2 = pd.read_excel(xls,"GNI")
df3 = pd.read_excel(xls,"Quarterly GDP")
GNI = list(df2['GNI'])

# expand GNI data set into monthly basis
GNI_list = []
i = 0
while i < len(GNI):
    j = 0
    while j < 12:
        GNI_list.append(GNI[i]/12)
        j = j + 1
    else:
        i = i + 1     
list1 = pd.DataFrame(GNI_list)

# expand GDP data set into monthly basis
GDP = list(df3['GDP'])
GDP_list = []
i = 0
while i < len(GDP):
    j = 0
    while j < 3:
        GDP_list.append(GDP[i]/3)
        j = j + 1
    else:
        i = i + 1     
list2 = pd.DataFrame(GDP_list)
print(len(GDP_list),len(GNI_list))

#combine data set into one dataset
result = df1.values
total = pd.DataFrame(result)
total = pd.concat([total,list1,list2],axis = 1)
total.columns = ['Country','Time','CPI','GNI','GDP']
total = total[['Time','Country','CPI','GNI','GDP']]
total.to_csv('./eco_data.csv', index=True)

# convert Covid-19 to requried format
def covid_19(df):
    df.set_index("date_day",inplace = True)
    data = df.loc[['2020-01-31','2020-02-29','2020-03-31','2020-04-30'],['Confirmed','Recovered','Deaths']]
    #For each country we have 108 data point from 2011-01 to 2019-12
    zeros = np.zeros((108,3))
    result = pd.DataFrame(zeros)
    #Combine none covid_19 data with covid_19 data
    result = pd.concat([result,pd.DataFrame(data.values)], axis=0)
    #reset index for combined dataframe
    result.reset_index(drop=True,inplace=True)
    return result

#read COVID_19 raw data
CHN = pd.read_csv('./China.csv')
DEU = pd.read_csv('./Germany.csv')
JPN = pd.read_csv('./Japan.csv')
MEX = pd.read_csv('./Mexico.csv')
TUR = pd.read_csv('./Turkey.csv')
USA = pd.read_csv('./US.csv')

#get requried format
CHN1 =  covid_19(CHN)
DEU1 =  covid_19(DEU)
JPN1 =  covid_19(JPN)
MEX1 =  covid_19(MEX)
TUR1 =  covid_19(TUR)
USA1 =  covid_19(USA)

#group economics data into different contries
econ = pd.read_csv('./eco_data.csv')
grouped = econ.groupby(econ.Country)
CHN0 = grouped.get_group('CHN')
DEU0 = grouped.get_group('DEU')
JPN0 = grouped.get_group('JPN')
MEX0 = grouped.get_group('MEX')
TUR0 = grouped.get_group('TUR')
USA0 = grouped.get_group('USA')

print(CHN0.head())
#make any interval months prediction for GDP GNI CPI
def eco(df,feature,interval):
    x  = np.arange(0,len(df[feature]),1).reshape(-1,1)
    y = df[feature]
    lm = linear_model.LinearRegression()
    model = lm.fit(x,y)
    predict = lm.predict(np.arange(df.shape[0],df.shape[0]+interval,1).reshape(-1,1))
    return predict
eco(CHN0,'GDP',6)
#assemble predict econ data with existing data
def assemble(df,feature,interval):
    raw = np.array(df[feature])
    predict = eco(df,feature,interval)
    result = np.concatenate((raw,predict))
    df = pd.DataFrame(result,columns = [feature])
    return df

#compute impact factor by historical data
'''we compute the factor by comparing CHN and USA GDP from(2020 1st quarter) to 
GDP(regression result from historical data)with affected by COVID_19
'''
def factor(df1,df2,feature,interval):
    #without COVID-19
    data1 = np.array(assemble(df1,feature,interval))
    #impact by COVID_19
    data2 = np.array(df2)
    #computer factor by (data2-data1)/data1 and weighted by interval
    result = 0
    for i in range(interval):
        result = result + (data2[-i]-data1[-i])/data1[-i]
    return result


#import most recent data from USA and CHINA
xls = pd.ExcelFile('./recentdata.xlsx')
xls.sheet_names
recent_china = pd.read_excel(xls,"CHINA")
recent_usa = pd.read_excel(xls,"USA")
# expand GDP data set into monthly basis
def monthly(df):
    GDP = list(df['GDP'])
    GDP_list = []
    i = 0
    while i < len(GDP):
        j = 0
        while j < 3:
            GDP_list.append(GDP[i]/3)
            j = j + 1
        else:
            i = i + 1     
    list2 = pd.DataFrame(GDP_list)
    return list2
recent_china = monthly(recent_china)
recent_usa = monthly(recent_usa)

#Now we compute the factor from US and CHINA for the future prediction
#factor from US
const_US = factor(CHN0,recent_china,'GDP',3)
#factor from CHINA
const_CN = factor(USA0,recent_usa,'GDP',3)

'''
NOW we have both factor from CHINA and US,we can use these factor to predict
2020 JAN FEB MARCH GDP for remaining coutries: Germany, Turkey, Mexico, Japan,
according to their cluster
We assume the similarity with the same cluster is 1, and since US and CHINA have 
the most dissimilarity and we have 4 clusters in total
SO we get the similarity matrix as following:
            US      CHINA
US          1        0
Germany     0.7      0.3
JAPAN       0.3      0.7   
MEXICO      0.3      0.7
CHINA       0        1
TURKEY      0        1
multiply similarity with US and CHINA factor, we can predict the fucture GDP
by: 
factor[US CN] x similarity_matrix = Impact coefficient
GDP(without COVID_19) * Impact coefficient = GDP(with COVID_19)
'''
#compute impact coefficient for GEU JPN MEX TUR
IC_DEU = np.dot(np.array([const_US,const_CN]).reshape(1,2),np.array([[0.7],[0.3]]))
IC_JPN = np.dot(np.array([const_US,const_CN]).reshape(1,2),np.array([[0.3],[0.7]]))
IC_MEX = np.dot(np.array([const_US,const_CN]).reshape(1,2),np.array([[0.3],[0.7]]))
IC_TUR = np.dot(np.array([const_US,const_CN]).reshape(1,2),np.array([[0.0],[0.1]]))
IC_CHN = const_CN
IC_USA = const_US
#define a function to return affect GDP data
'''if we want to change the prediction period change interval here'''
def Affected(df,IC):
    interval = 6
    list1= df.values.tolist()
    for i in range(interval):
        list1[-i-1] = list1[-i-1]*IC
    result = pd.DataFrame(list1,columns = ['Affected_GDP'])
    return result
'''predict GDP growth rate for 2020 Jan Feb Mar Ari June July,change the interval for further prediction 
#Germany
'''
#Without COVID_19
Unaffected_DEU = assemble(DEU0,'GDP',6)
#With COVID_19
Affected_DEU = Affected(Unaffected_DEU,IC_DEU)
'''
Japan
'''
Unaffected_JPN = assemble(JPN0,'GDP',6)
Affected_JPN = Affected(Unaffected_JPN,IC_JPN)
'''
Mexico
'''
Unaffected_MEX = assemble(MEX0,'GDP',6)
Affected_MEX = Affected(Unaffected_MEX,IC_MEX)
'''
Turkey
'''
Unaffected_TUR = assemble(TUR0,'GDP',6)
Affected_TUR = Affected(Unaffected_TUR,IC_MEX)
'''China'''
Unaffected_CHN = assemble(CHN0,'GDP',6)
Affected_CHN = Affected(Unaffected_CHN,IC_CHN)
'''USA'''
Unaffected_USA = assemble(USA0,'GDP',6)
Affected_USA = Affected(Unaffected_USA,IC_USA)


"""
NOW PLOT the data!!!!!
"""
import matplotlib.pyplot as plt

def plot(data1,data2,country_name):
    date_rng = pd.period_range('1/1/2011',freq='M',periods=len(data1.values))
    df = pd.DataFrame(date_rng,columns=['date'])
    result = pd.concat([df,data1,data2],axis = 1)
    result.set_index('date',inplace=True,drop=True)
    #change the x_axis range from there, remember to corresponding to the interval in assemble function
    show = result['2018-6':'2020-6']
    plt.figure(figsize = (15,8))
    plt.plot(show.index.to_timestamp(),show['GDP'],label='Unaffected_GDP')
    plt.plot(show.index.to_timestamp(),show['Affected_GDP'],label='Affected_GDP')
    plt.legend()
    plt.title(country_name)
    plt.ylabel('GDP_growth_rate[%]')
    return
#plot the graph for Turkey,Germany,Mexico,Japan,China,USA
plot(Unaffected_TUR,Affected_TUR,'Turkey')
plt.savefig('./Turkey.png')

plot(Unaffected_DEU,Affected_DEU,'Germany')
plt.savefig('./Germany.png')

plot(Unaffected_MEX,Affected_MEX,'Mexico')
plt.savefig('./Mexico.png')

plot(Unaffected_JPN,Affected_JPN,'Japan')
plt.savefig('./Japan.png')

plot(Unaffected_CHN,Affected_CHN,'China')
plt.savefig('./China.png')

plot(Unaffected_USA,Affected_USA,'USA')
plt.savefig('./USA.png')


