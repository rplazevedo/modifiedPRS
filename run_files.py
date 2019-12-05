#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:27:50 2019

@author: rplazevedo
"""

#import matplotlib.pyplot as plt
import configparser
import reduce_data
import cleanup
import os
        

config = configparser.ConfigParser()
config.read('input.ini')
all_phi = config['Parameters']['all_phi']


if input('Run cleanup? y/(n)\n') in ['y','Yes']:
    cleanup.run()

run_init = input('Run initial sPRS? (y)/n \n')
run_sPRS_cont = input('Run full sPRS? (y)/n \n')
run_mPRS = input('Run mPRS? (y)/n \n')

os.system("python3 setup.py build_ext --inplace")

import sprs_init   
import sprs_cont_vect
import mprs_cont_vect
import merge_v
import merge_phi

if run_init in ['','y','Y']:
    print('\n---Running initial sPRS---' )
    sprs_init.run()

if run_sPRS_cont in ['','y','Y']:
    print('\n---Running rest of sPRS---' )
    sprs_cont_vect.run()
    merge_v.run()
    reduce_data.run()
    if all_phi in ['yes', 'YES', 'some', 'SOME']:
        merge_phi.run()

if run_mPRS in ['','y','Y']:
    print('\n---Running mPRS---')
    mprs_cont_vect.run()
    merge_v.run()
    reduce_data.run()
    if all_phi in ['yes', 'YES', 'some', 'SOME']:
        merge_phi.run()

print('---Done!---')




# plot the velocity
        
#print('---Plotting (to take over the world)---')
#
#X, Y = [], []
#for line in open("v.data","r"):
#    x, y = line.split()
#    X.append(float(x))
#    Y.append(float(y))
#
#plt.figure()
#plt.plot(X,Y)
#plt.show()
