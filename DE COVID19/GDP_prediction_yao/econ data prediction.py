#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:20:48 2020

@author: zhangyao
"""
import numpy as np
import pandas as pd
from sklearn import linear_model

''' read all the data set we have, And the data set is pre_selected with 
    all target countries: CHN,DEU,JPN,MEX,TUR,USA. Since GNI is yearly basis,
    and GDP is quarterly basis, we need to expand the data set into monthly basis,
    and combine all data into one dataset,in order for the following prediction.
'''
# read data set
xls = pd.ExcelFile('/Users/zhangyao/Downloads/DE COVID19/econ features/CPI Data.xlsx')
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
total.to_csv('/Users/zhangyao/Downloads/DE COVID19/eco_data.csv', index=True)

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
CHN = pd.read_csv('/Users/zhangyao/Downloads/DE COVID19/cumulative_data/China.csv')
DEU = pd.read_csv('/Users/zhangyao/Downloads/DE COVID19/cumulative_data/Germany.csv')
JPN = pd.read_csv('/Users/zhangyao/Downloads/DE COVID19/cumulative_data/Japan.csv')
MEX = pd.read_csv('/Users/zhangyao/Downloads/DE COVID19/cumulative_data/Mexico.csv')
TUR = pd.read_csv('/Users/zhangyao/Downloads/DE COVID19/cumulative_data/Turkey.csv')
USA = pd.read_csv('/Users/zhangyao/Downloads/DE COVID19/cumulative_data/US.csv')

#get requried format
CHN1 =  covid_19(CHN)
DEU1 =  covid_19(DEU)
JPN1 =  covid_19(JPN)
MEX1 =  covid_19(MEX)
TUR1 =  covid_19(TUR)
USA1 =  covid_19(USA)

#group economics data into different contries
econ = pd.read_csv('/Users/zhangyao/Downloads/DE COVID19/eco_data.csv')
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
xls = pd.ExcelFile('/Users/zhangyao/Downloads/DE COVID19/econ features/recentdata.xlsx')
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
plt.savefig('/Users/zhangyao/Downloads/DE COVID19/prediction_result/Turkey.png')

plot(Unaffected_DEU,Affected_DEU,'Germany')
plt.savefig('/Users/zhangyao/Downloads/DE COVID19/prediction_result/Germany.png')

plot(Unaffected_MEX,Affected_MEX,'Mexico')
plt.savefig('/Users/zhangyao/Downloads/DE COVID19/prediction_result/Mexico.png')

plot(Unaffected_JPN,Affected_JPN,'Japan')
plt.savefig('/Users/zhangyao/Downloads/DE COVID19/prediction_result/Japan.png')

plot(Unaffected_CHN,Affected_CHN,'China')
plt.savefig('/Users/zhangyao/Downloads/DE COVID19/prediction_result/China.png')

plot(Unaffected_USA,Affected_USA,'USA')
plt.savefig('/Users/zhangyao/Downloads/DE COVID19/prediction_result/USA.png')
