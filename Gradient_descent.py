#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:23:50 2019

@author: lukasbuhmann
"""

"""

UNIVARIAT GRADIENT DESCENT

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

#Choosing a simple quadratic functional form


def function(x1):
    
    y = ((x1+3)**2 + 3)
    
    return y

#Graphical Analysis

x_range = np.arange(-20,21,1)
x_range

plt.plot(x_range, function(x_range))

#define slope

def function_delta(x_val):
    
    y = (function(x_val + 0.001) - function(x_val))/0.001
    
    return y 


#define gradient descent step

def gradient_descent(x_val, conv_speed):
    
    y = x_val - conv_speed*function_delta(x_val)
    
    return y



#update process

change = 0.02
x_null = 10

while change > 0.00000010101:
    
    
    x_new = gradient_descent(x_null, 0.011)
    
    change = abs(x_new - x_null)/abs(x_null)
    
    x_null = x_new
    print(x_null)
    
    next


#3D Function
  #%%  
#import 3D plotting tool    
from mpl_toolkits.mplot3d import Axes3D

from decimal import Decimal
#define function

def function_3d(x_val, y_val):
    
    y = np.power(x_val - 2, 2) + np.power(y_val + 5, 2) 
    
    return y 

#plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X_val_3d = np.arange(-20, +20, 1)
Y_val_3d = np.arange(-20, +20, 1)

X_val_3d_1, Y_val_3d_1 = np.meshgrid(X_val_3d, Y_val_3d)


Z_val = function_3d(X_val_3d_1,Y_val_3d_1)

ax.plot_surface(X_val_3d_1, Y_val_3d_1, Z_val, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');
                
plt.show()
#%%
#define derivations

def function_delta1(x_val,y_val):
    
    y = (function_3d(x_val + 0.01, y_val) - function_3d(x_val, y_val))/0.01
    
    return y 

def function_delta2(x_val, y_val):
    
    y = (function_3d(x_val, y_val + 0.01) - function_3d(x_val, y_val))/0.01
    
    return y 

# define gradient descent

def gradient_descent1(x_val, y_val, conv_speed):
    
    y = x_val - conv_speed*function_delta1(x_val, y_val)
    
    return y

def gradient_descent2(x_val, y_val, conv_speed):
    
    y = y_val - conv_speed*function_delta2(x_val ,y_val)
    
    return y


change_x = 0.02
change_y = 0.02

x_null = 3
y_null = 3
iter = 0

while change_x > 0.00001 and change_y > 0.000000001 and iter < 1000000:

    x_new = gradient_descent1(x_null, y_null, 0.01)
    y_new = gradient_descent2(x_null, y_null , 0.01)
    
    change_x = abs((x_new - x_null)/x_null)
    change_y = abs((y_new - y_null)/y_null)
    
    #print(change_x, change_y)
    
    x_null = x_new
    y_null = y_new
    
    iter= iter + 1
    
    print(iter)
    next
    
print(x_null, y_null, function_3d(x_null, y_null))



    
    


