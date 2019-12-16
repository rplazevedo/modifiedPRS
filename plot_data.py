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
import numpy as np

def run():
    config = configparser.ConfigParser()
    config.read('input.ini')
    name_def = config['Parameters']['name']
    t0 = float(config['PRS']['initial_t'])
    w0 = float(config['PRS']['initial_width'])
    phi_0 = float(config['PRS']['initial_phi'])
    alpha = float(config['PRS']['alpha'])
    dt = float(config['PRS']['dt'])
    delta_x = float(config['PRS']['dx'])
    delta_y = float(config['PRS']['dy'])
    alpha_1 = float(config['Parameters']['alpha_1'])
    alpha_2 = float(config['Parameters']['alpha_2'])
    alpha_e = float(config['Parameters']['alpha_e'])
    Nw_alpha = float(config['Parameters']['Nw_alpha'])
    daeta = int(config['Parameters']['dlna/dlneta'])
    name = name_def       
    
    # name = str(input('Name of the data to plot?\n'))
    # if name == '':
    #     name = name_def
    
    lbl = ['sPRS', r'$\eta_m=50$', r'$\eta_m=35$', r'$\eta_m=20$', r'$\eta_m=5$']
    
    # V plots 
    plt.figure()
    number = 0
    while os.path.isfile(str(name) +'_'+str(number)+ '_v.data'):
        T, V = [], []
        for line in open(str(name) +'_'+str(number)+ '_v.data',"r"):
            x, y = line.split()
            T.append(float(x))
            V.append(float(y))
        plt.plot(T, V, label=lbl[number])         
        if number == 0:
            V0 = V
        number += 1
    plt.xscale('log')
    plt.title(fr"$w_0={w0:G}$, $\phi_0={phi_0:G}$, $dt={dt:G}$, $\alpha_w={Nw_alpha:G}$, $\alpha_1={alpha_1:G}$, $\alpha_2={alpha_2:G}$, $\alpha_\rho={alpha_e:G}$")
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\gamma^2 v^2$')
    plt.xlim(5,250)
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    number = 0
    V0a = np.array(V0)
    while os.path.isfile(str(name) +'_'+str(number)+ '_v.data'):
        T, V = [], []
        for line in open(str(name) +'_'+str(number)+ '_v.data',"r"):
            x, y = line.split()
            T.append(float(x))
            V.append(float(y))
        Va = np.array(V)
        plt.plot(T, np.abs(Va-V0a)/V0a, label=lbl[number])
        number += 1
    plt.xscale('log')
    plt.title(fr"$w_0={w0:G}$, $\phi_0={phi_0:G}$, $dt={dt:G}$, $\alpha_w={Nw_alpha:G}$, $\alpha_1={alpha_1:G}$, $\alpha_2={alpha_2:G}$, $\alpha_\rho={alpha_e:G}$")
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\gamma^2 v^2$ (frac. diff.)')
    plt.xlim(5,250)
    plt.grid()
    plt.legend()
    plt.show()
    
    # Number of walls
    plt.figure()
    number = 0
    while os.path.isfile(str(name) +'_'+str(number)+ '_nwalls.data'):
        T, Nw = [], []
        for line in open(str(name) +'_'+str(number)+ '_nwalls.data',"r"):
            x, y = line.split()
            T.append(float(x))
            Nw.append(float(y))
        plt.plot(T, Nw, label=lbl[number])         
        if number == 0:
            Nw0 = Nw
        number += 1
    plt.xscale('log')
    plt.title(fr"$w_0={w0:G}$, $\phi_0={phi_0:G}$, $dt={dt:G}$, $\alpha_w={Nw_alpha:G}$, $\alpha_1={alpha_1:G}$, $\alpha_2={alpha_2:G}$, $\alpha_\rho={alpha_e:G}$")
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$N_{walls}$')
    plt.xlim(5,250)
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    number = 0
    Nw0a = np.array(Nw0)
    while os.path.isfile(str(name) +'_'+str(number)+ '_nwalls.data'):
        T, Nw = [], []
        for line in open(str(name) +'_'+str(number)+ '_nwalls.data',"r"):
            x, y = line.split()
            T.append(float(x))
            Nw.append(float(y))
        Nwa = np.array(Nw)
        plt.plot(T, np.abs(Nwa-Nw0a)/Nw0a, label=lbl[number])
        number += 1
    plt.title(fr"$w_0={w0:G}$, $\phi_0={phi_0:G}$, $dt={dt:G}$, $\alpha_w={Nw_alpha:G}$, $\alpha_1={alpha_1:G}$, $\alpha_2={alpha_2:G}$, $\alpha_\rho={alpha_e:G}$")
    plt.xscale('log')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$N_{walls}$ (frac. diff.)')
    plt.xlim(5,250)
    plt.grid()
    plt.legend()
    plt.show()
    
    # Vacuum plots
    plt.figure()
    number = 0
    while os.path.isfile(str(name) +'_'+str(number)+ '_exc_pts.data'):
        T, P = [], []
        for line in open(str(name) +'_'+str(number)+ '_exc_pts.data',"r"):
            x, y = line.split()
            T.append(float(x))
            P.append(float(y))
            
        plt.plot(T, P, label=lbl[number])
        number += 1
    plt.title(fr"$w_0={w0:G}$, $\phi_0={phi_0:G}$, $dt={dt:G}$, $\alpha_w={Nw_alpha:G}$, $\alpha_1={alpha_1:G}$, $\alpha_2={alpha_2:G}$, $\alpha_\rho={alpha_e:G}$")
    plt.xscale('log')
    plt.xlabel(r'$\eta$')
    plt.ylabel('Frac. of vacua')
    plt.xlim(5,250)
    plt.grid()
    plt.legend()
    plt.show()

# Run the function
run()