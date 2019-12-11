#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:50:36 2019

@author: rplazevedo
"""

#import numpy as np
import matplotlib.pyplot as plt
import os
import configparser

config = configparser.ConfigParser()
config.read('input.ini')
name = config['Parameters']['name']
   
name= str(input('Name of the data to plot?\n'))

plt.figure()
number = 0
while os.path.isfile(str(name) +'_'+str(number)+ '_v.data'):
    T, V = [], []
    for line in open(str(name) +'_'+str(number)+ '_v.data',"r"):
        x, y = line.split()
        T.append(float(x))
        V.append(float(y))
    plt.plot(T,V)
    number += 1
plt.xscale('log')
plt.show()
    