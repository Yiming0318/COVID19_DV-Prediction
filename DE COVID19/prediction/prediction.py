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
# df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/China.csv')
# df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/Iran.csv')
# df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/Japan.csv')
# df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/Philippines.csv')
# df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/South Korea.csv')
# df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/Thailand.csv')
# df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/UK.csv')
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
 


