#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:12:45 2019

@author: lukasbuhmann
"""

#Logistische Regression über gradient descent schätzen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from mpl_toolkits.mplot3d import Axes3D
#%%
#Datensets einlesen


#Dataset
dataset = pd.read_csv("/Users/lukasbuhmann/Documents/Python Projekte/Gradient_descent/AutoMPGdata/AutoMPG-Tabelle 1.csv", 
                      sep = ";", header = 0, decimal = ',')

dataset = dataset.assign(intercept=pd.Series(np.ones(392)).values)
#Intercept hinzufügen: Dieser besteht lediglich aus Einsen: Intuitiv ist vorstellbar, dass wir für jede Beobachtung ein 
#gleichgewichtetes Beta schätzen, dass im Gegensatz zu allen anderen Variablen keine zeilenweise variierenden Werte annimmt

#%% 
#create 0/1 column for logistic training and prediction
#print(dataset.loc[0,"Cylinders"])

binary_array = np.array([1 if dataset.loc[i,"Cylinders"] >= 6 else 0 for i in range(0,392)])
print(binary_array) 
        
dataset = dataset.assign(binary_var = pd.Series(binary_array))

#%%

Design_mat = dataset.loc[:,['intercept','MPG' , 'Year', 'Weight1000lb', 'Seconds0to60']].values
dep_var = dataset.loc[:, 'binary_var']

Design_mat.shape[0]
Design_mat[1,:]
#%%

def log_likelihood(y, X, betas):
    
    sum_likelihood = 0
    
    for i in range(0, X.shape[0]):
        
        X_i = X[i,:]
        y_i = y[i]
        
        prob_1 = np.exp(np.dot(X_i, betas))/(1 + np.exp(np.dot(X_i, betas)))
        
        log_likelihood = np.power(prob_1, y_i) + np.power(1 - prob_1, 1 - y_i)
                
        sum_likelihood = log_likelihood + sum_likelihood
        
        next
        
    return sum_likelihood
        
        
#%%
        
def der_log_likelihood(y, X, betas):
    
    X_i  = X[1,:]
    
    prob_tmp = np.exp(np.dot(X_i, betas))/(1 + np.exp(np.dot(X_i, betas)))
    
    vec_tmp = X[1,:] * (y[1] - prob_tmp)
    
    return vec_tmp
    
        
#%%

betas = [0.1 , 0.1 , 0.1, 0.1 , 0.1 ]

print(log_likelihood(dep_var, Design_mat, betas))
print(der_log_likelihood(dep_var, Design_mat, betas))
    
#%%