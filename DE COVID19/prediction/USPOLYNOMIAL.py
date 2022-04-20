# -*- coding: utf-8 -*-
from scipy.optimize import curve_fit
import urllib
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import scipy as sp
from scipy.stats import norm
 
df = pd.read_csv('F:/COVID19DV/dash-2019-coronavirus/cumulative_data/US.csv')
 
 
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