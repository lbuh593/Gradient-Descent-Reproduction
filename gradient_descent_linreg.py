#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:54:59 2019

@author: lukasbuhmann
"""

#Lineare Regression über gradient descent schätzen

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

#Aufteilung: y ist die Variable, die ich schätzen möchte. X_0 - X_3 sind mehrere Datensets, die ich zur Modellierung heranziehe
#(Bisher benutze ich nur X_0 und X_2)
y = dataset.loc[:,'GallonsPer100Miles'].values

X_0 = dataset.loc[:,['intercept','MPG']].values
X_1 = dataset.loc[:,['intercept','MPG', 'Cylinders']].values
X_2 = dataset.loc[:,['intercept','MPG', 'Cylinders', 'Year', 'Weight1000lb', 'Seconds0to60']].values
X_3 = dataset.loc[:,['MPG', 'Cylinders', 'Year', 'Weight1000lb', 'Seconds0to60']].values

#%%
#Definiere die Zielfunktion. Diese ist lediglich das wahre y abzüglich des geschätzten y:
# error_i = y_i - (beta_1 * x_1 + beta_2 * x_2 + ... + beta_n * x_n) 
# Als Vektorschreibweise: Error = y - X*beta, Ergebnis ist ein n x 1 Vektor mit den Errors für alle Zeilen: n ist hier die Anzahl
# der Zeilen von y und natürlich auch von allen X 

def error(y,X,betas):
    
    error = np.power(y - np.dot(X,betas),2)
    return error.sum()

#%%
    
#Im Folgenden modelliere ich eine Lineare Regression für X_0. Hierfür benutze ich das Gradient Descent Verfahren mit Armijo Schrittweite

#Die Armijo Schrittweite ist im Prinzip eine Versicherung, damit ich mit dem Gradientenverfahren keinen zu großene Schritt mache und
#und so möglicherweise in einer Divergenzdynamik rutsche
    
#Das Gradientenverfahren funktioniert so:

#Gegeben: min f(x)
#initialisiere x = x_null     

# Iterativ:
# x_neu = x_null - Schrittweite * Gradient (df/dx)
    
# Abbruchkritierium ist, dass die relative Änderung von f(x) von x_alt nach x_neu kleiner als ein gewisser Prozentsatz ist. 
    
#Ich schicke dir Links für diese Armijo Schrittweite

#Define Armijo-Schrittweite (Das ganze als 2 dimensionaler Vektor)

sigma = 0.1
theta = 0.2

def armijo_0(beta_old,X,y):
    
    t_null=1
    maxit = 100000
    it = 0
    
    while it < maxit:
        
        #print(it)
        
        left = error(y,X,beta_old - t_null * Ableitungen_0(beta_old))
        
        #print(left)
        
        right = error(y,X,beta_old) - sigma * t_null * np.dot(Ableitungen_0(beta_old).reshape(1,-1),Ableitungen_0(beta_old).reshape(-1,1))
        
        #print(right)
        
        if left > right:
        
            t_null = theta * t_null
            
            #print(t_null)
            
        else:
                
            break
        
        it = it +1
        next
        
    return t_null
    


print(armijo_0([0.1,0.1], X_0, y))


#%%

#Dies ist ein Python Lineare Regression Alghorithmus zum Vergleich der Resultate

regr = linear_model.LinearRegression(fit_intercept = False)

# Train the model using the training sets
regr.fit(X_0, y)

#Koeffizienten

coefs = regr.coef_



#%%
# Ableitungen über Differenzenqutient geschätzt (nicht gebraucht)

def Ableitungen_0_est(betas):
    
    der_b1 = (error(y, X_0, betas + np.array([0.0000001 , 0])) - error(y, X_0, betas))/0.0000001
    
    der_b2 = (error(y, X_0, betas + np.array([0 , 0.0000001])) - error(y, X_0, betas))/0.0000001
    
    return np.array([der_b1, der_b2]) 

#%%
# Analytische Ableitungen (als Matrizenobjekt)
    
#Error = (y - X*beta)^2
    
# dError / d beta = 2 * ( -X'y + X'X * beta) 

def Ableitungen_0(betas):
    
    der = 2*((-1) * np.dot(np.transpose(X_0), y) + np.dot(np.transpose(X_0), np.dot(X_0, betas)))
        
    return der

#%%
#initialize algorithm variables
    
iter = 0
change_y = 0.01
beta_0 = [0.1,0.1]

#gradient descent Verfahen

#Abbruchbedingungen
while change_y > 0.0000001 and iter < 200000:
    
    print(iter)
    
    #Berechnung der Schrittweite
    schrittweite = armijo_0(beta_0, X_0, y)
    
    print(schrittweite)
    
    #Gradientenverfahren iterativer Prozess 
    beta_tmp = beta_0 - schrittweite * Ableitungen_0(beta_0)
    
    print(beta_tmp)
    
    #Berechnung Abbruchkriterium
    change_y = abs((error(y,X_0,beta_tmp) - error(y,X_0,beta_0))/error(y,X_0,beta_0))
    
    beta_0 = beta_tmp
    
    iter = iter + 1
    
    next


print(beta_0)
########################################################################################

#Jetzt das ganz mit anderem X Dataset: Nicht 2-dimensional, sondern 6-dimensional

########################################################################################
#%%

# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept = False)

# Train the model using the training sets
regr.fit(X_2, y)

regr.coef_

#%%
    

def Ableitungen_2_est(X, betas):
       
    der_b1 = (error(y,X, betas+ np.array([0.0000001,0,0,0,0,0])) - error(y,X, betas))/0.0000001
    der_b2 = (error(y,X, betas+ np.array([0,0.0000001,0,0,0,0])) - error(y,X, betas))/0.0000001
    der_b3 = (error(y,X, betas+ np.array([0,0,0.0000001,0,0,0])) - error(y,X, betas))/0.0000001
    der_b4 = (error(y,X, betas+ np.array([0,0,0,0.0000001,0,0])) - error(y,X, betas))/0.0000001
    der_b5 = (error(y,X, betas+ np.array([0,0,0,0,0.0000001,0])) - error(y,X, betas))/0.0000001
    der_b6 = (error(y,X, betas+ np.array([0,0,0,0,0,0.0000001])) - error(y,X, betas))/0.0000001
    
    return np.array([der_b1, der_b2, der_b3, der_b4, der_b5, der_b6])

Ableitungen_2_est(X_2, [9.21346688, -0.10864554,  0.09530809, -0.05471369,  0.64840218, -0.01135981])

#%%

def Ableitungen_2(betas):
    
    der = 2*((-1) * np.dot(np.transpose(X_2), y) + np.dot(np.transpose(X_2), np.dot(X_2, betas)))
        
    return der

Ableitungen_2([9.21346688, -0.10864554,  0.09530809, -0.05471369,  0.64840218, -0.01135981])    
#%%

#Define Armijo-Schrittweite

sigma = 0.5
theta = 0.5

def armijo_2(beta_old,X,y):
    
    t_null=1
    maxit = 100000
    it = 0
    
    while it < maxit:
    
        left = error(y,X,beta_old - t_null * Ableitungen_2(beta_old))
        
        #print(left)
        
        #print(Derivatives(beta_old))
        right = error(y,X,beta_old) - sigma * t_null * np.dot(Ableitungen_2(beta_old).reshape(1,-1),Ableitungen_2(beta_old).reshape(-1,1))
        
        #print(right)
        
        if left > right:
        
            t_null = theta * t_null
            
            #print(t_null)
            
        else:
                
            break
            
        next
        
    return t_null
    
#print(error(y,X_0,beta_0)+ sigma * 0.5 * np.dot(Derivatives(beta_0).reshape(1,-1),Derivatives(beta_0)))

print(armijo_2(beta_old, X_2, y))
    
#%%

beta_old = np.array([8, 0.01, 0.01 ,0.01 ,0.01, 0.01])
iter = 0
change_y = 0.01

while change_y > 0.000001 and iter < 1000000:
    
    schrittweite = armijo_2(beta_old, X_2, y)
    
    beta_new = beta_old - schrittweite * Ableitungen_2(beta_old)
    
    change_y = abs((error(y,X_2,beta_new) - error(y,X_2,beta_old))/error(y,X_2,beta_old))
    
    beta_old = beta_new
    
    iter = iter + 1
    
    print(beta_old)
    print(iter)
    next
    
print(beta_old)