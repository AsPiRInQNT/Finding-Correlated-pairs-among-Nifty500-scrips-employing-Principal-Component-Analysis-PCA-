#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv  
nifty=[]
with open('Nifty500List.csv','w') as output:
    with open('Nifty500.csv','r') as myFile:  
        output_data=csv.writer(output)
        data=csv.reader(myFile)
        for row in data:
            row.append('.NS')
            output_data.writerow(row)
#nifty=list(map(list, zip(*Equity1.csv)))

#nifty=nifty[0]
nifty=[]
with open('Nifty500List.csv','r') as myfile:
    for row in myfile:
        nifty.append(row.strip('\n'))
nifty=list(filter(str.strip, nifty))

nifty = [n.replace(',', '') for n in nifty]


# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime


# In[4]:


from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

# download dataframe using pandas_datareader
data = pdr.get_data_yahoo(nifty, start="2015-01-01", end="2020-04-30")
data=data['Adj Close'].pct_change().dropna()


# In[5]:


# portfolio pre-processing
dfP = data[(data.index >= "2019-01-01") & (data.index <= "2020-04-30")]
dfP = dfP.dropna(axis=1, how='any')
dfP.astype('float')


# In[6]:


m = dfP.mean(axis=0)
s = dfP.std(ddof=1, axis=0)
 
# normalised time-series as an input for PCA
dfPort = (dfP - m)/s
 
c = np.cov(dfP.values.T)     # covariance matrix
co = np.corrcoef(dfP.values.T)  # correlation matrix
 
tickers = list(dfP.columns)


# In[7]:


plt.figure(figsize=(15,15))
plt.imshow(co, cmap="RdGy", interpolation="nearest")
cb = plt.colorbar()
cb.set_label("Correlation Matrix Coefficients")
plt.title("Correlation Matrix", fontsize=14)
plt.xticks(np.arange(len(tickers)), tickers, rotation=90)
plt.yticks(np.arange(len(tickers)), tickers)
 
# perform PCA
w, v = np.linalg.eig(c)  
print(v.dtype)
ax = plt.figure(figsize=(15,15)).gca()
plt.imshow(np.real(v), cmap="bwr", interpolation="nearest")
cb = plt.colorbar()
plt.yticks(np.arange(len(tickers)), tickers)
plt.xlabel("PC Number")
plt.title("PCA", fontsize=14)
# force x-tickers to be displayed as integers (not floats)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# In[8]:


# choose PC-k numbers
k1 = -2  # the last PC column in 'v' PCA matrix
k2 = -3  # the second last PC column
 
# begin constructing bi-plot for PC(k1) and PC(k2)
# loadings
plt.figure(figsize=(7,7))
plt.grid()
 
# compute the distance from (0,0) point
dist = []
for i in range(v.shape[0]):
    x = v[i,k1]
    y = v[i,k2]
    plt.plot(x, y, '.k')
    plt.plot([0,x], [0,y], '-', color='grey')
    d = np.sqrt(x**2 + y**2)
    dist.append(d)


# In[9]:


# check and save membership of a coin to
# a quarter number 1, 2, 3 or 4 on the plane
quar = []
for i in range(v.shape[0]):
    x = v[i,k1]
    y = v[i,k2]
    d = np.sqrt(x**2 + y**2)
    if(d > np.mean(dist) + np.std(dist, ddof=1)) :
        plt.plot(x, y, '.r', markersize=10)
        plt.plot([0,x], [0,y], '-', color='grey')
        if((x > 0) and (y > 0)):
            quar.append((i, 1))
        elif((x < 0) and (y > 0)):
            quar.append((i, 2))
        elif((x < 0) and (y < 0)):
            quar.append((i, 3))
        elif((x > 0) and (y < 0)):
            quar.append((i, 4))
        plt.text(x, y, tickers[i], color='k')
 
plt.xlabel("PC-" + str(len(tickers)+k1+1))
plt.ylabel("PC-" + str(len(tickers)+k2+1))


# In[ ]:


for i in range(len(quar)):
    # Q1 vs Q3
    if(quar[i][1] == 1):
        for j in range(len(quar)):
            if(quar[j][1] == 3):
                plt.figure(figsize=(7,4))
                
                # highly correlated coins according to the PC analysis
                print(tickers[quar[i][0]], tickers[quar[j][0]])
                
                ts1 = dfP[tickers[quar[i][0]]]  # time-series
                ts2 = dfP[tickers[quar[j][0]]]
                
                # correlation metrics and their p_values
                slope, intercept, r2, pvalue, _ = stats.linregress(ts1, ts2)
                ktau, kpvalue = stats.kendalltau(ts1, ts2)
                print(r2, pvalue)
                print(ktau, kpvalue)
                
                plt.plot(ts1, ts2, '.k')
                xline = np.linspace(np.min(ts1), np.max(ts1), 100)
                yline = slope*xline + intercept
                plt.plot(xline, yline,'--', color='b')  # linear model fit
                plt.xlabel(tickers[quar[i][0]])
                plt.ylabel(tickers[quar[j][0]])
                plt.show()
    # Q2 vs Q4
    if(quar[i][1] == 2):
        for j in range(len(quar)):
            if(quar[j][1] == 4):
                plt.figure(figsize=(7,4))
                print(tickers[quar[i][0]], tickers[quar[j][0]])
                ts1 = dfP[tickers[quar[i][0]]]
                ts2 = dfP[tickers[quar[j][0]]]
                slope, intercept, r2, pvalue, _ = stats.linregress(ts1, ts2)
                ktau, kpvalue = stats.kendalltau(ts1, ts2)
                print(r2, pvalue)
                print(ktau, kpvalue)
                plt.plot(ts1, ts2, '.k')
                xline = np.linspace(np.min(ts1), np.max(ts1), 100)
                yline = slope*xline + intercept
                plt.plot(xline, yline,'--', color='b')
                plt.xlabel(tickers[quar[i][0]])
                plt.ylabel(tickers[quar[j][0]])
                plt.show()


# In[ ]:




