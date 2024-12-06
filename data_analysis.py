# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:37:21 2024

@author: XYZW
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
#import datetime as dt

def convert_format(string):
    """
    Converts the time format of string 'M/D/Y' to 'Y-M-D'
    """
    date = string.split('/')
    if int(date[1])<10:
        date[1]='0'+date[1]
    if int(date[0])<10:
        date[0] = '0'+date[0]
    return str(date[2]+'-'+date[0]+'-'+date[1])
#%%
"""
Remove unnecessary index column from data csv (containing the markers for Sales).

Remove duplicate(s) lines from the Revenues 

data_csv contains the Sales Markers
df_revenues (test_data2) contains the revenues values
"""
data_csv = pd.read_csv('Test_data1.csv')
data_csv=data_csv.drop(labels = data_csv.columns[0],axis = 1)
data_csv = data_csv.rename(columns = {'date':'Date','Sales':'Sales'})
f = open('Test_data2.txt','r') # the revenues
data_text= f.readlines()
sep_data = [data_text[i].rstrip('\n').split('|') for i in range(len(data_text))]
df_revenues = pd.DataFrame(sep_data[1:],columns = ['Index','Date','Revenue'])
df_revenues = df_revenues.drop_duplicates()
df_revenues.index = df_revenues['Index']
df_revenues = df_revenues.drop(labels = df_revenues.columns[0],axis = 1)
data_csv['Date'] = [convert_format(x) for x in data_csv['Date']]
merged_df = pd.merge(data_csv,df_revenues,how = 'outer',on = 'Date')
merged_df = merged_df[merged_df.iloc[:,1].isna()==0]
merged_df = merged_df[merged_df['Revenue']!='@#$%%#@$$']
merged_df['Revenue'] = [float(x) for x in merged_df['Revenue']]
#%%
"""
It is clear that the Red indicator for Sales imply strongly negative values for Revenues, 
the yellow are slightly around 0 (both positive and negative),
and Green indicator for Sales imply strongly positive vals

The red revenues have a left tail outlier

The green revenues have a right tail outlier + NaN

The yellow revenues contain one single negative value. 

"""

red_revs = sorted(merged_df[merged_df['Sales']=='Red']['Revenue'])
yellow_revs = sorted(merged_df[merged_df['Sales']=='Yellow']['Revenue'])
green_revs = sorted(merged_df[merged_df['Sales']=='Green']['Revenue'])

"""
1. For yellow revenues, except the negative value (outlier), I would assume a uniform distribution 
between theta1,theta2 (to be estimated) by using Maximum Likelihood estimators

2. For red revenues except the left outlier (-60.23), i would assume a uniform distribution between 
alpha1, alpha2 (to be estimated) 

3. For green revenues except the right outlier I would assume another uniform distribution 
between beta1, beta2 (also to be estimated)

Red Revenues: Unif(theta1,theta2)=Unif(-7.30,-2.50)

Yellow Revenues: Unif(alpha1,alpha2) = Unif(0,4.84)

Green Revenues: Unif(beta1,beta2) = Unif(17.0,22.5)
"""

theta1,theta2 = (min(red_revs[1:]),max(red_revs[1:]))

alpha1,alpha2 = (min(yellow_revs[1:]),max(yellow_revs[1:]))

beta1,beta2 = (min(green_revs),max(green_revs[0:-2]))

#%%
