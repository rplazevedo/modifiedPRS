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

def run():
    config = configparser.ConfigParser()
    config.read('input.ini')
    name_def = config['Parameters']['name']

    name = name_def       
    
    # name = str(input('Name of the data to plot?\n'))
    # if name == '':
    #     name = name_def
    
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
    
    plt.figure()
    number = 0
    while os.path.isfile(str(name) +'_'+str(number)+ '_exc_pts.data'):
        T, P = [], []
        for line in open(str(name) +'_'+str(number)+ '_exc_pts.data',"r"):
            x, y = line.split()
            T.append(float(x))
            P.append(float(y))
        plt.plot(T,P)
        number += 1
    plt.xscale('log')
    plt.show()
# run()