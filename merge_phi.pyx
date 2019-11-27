import numpy as np
import random
import sys
import time
import os
import configparser

cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

config = configparser.ConfigParser()
config.read('input.ini')

def run():    
    cdef int Nx = int(config['PRS']['Lattice_size'])
    cdef int Nt_int = int(config['PRS']['points_per_part'])
    cdef int parts = int(config['PRS']['sPRS_parts'])+int(config['PRS']['mPRS_parts'])
    cdef str name = config['Parameters']['name']
    
    #cdef int parts = 150
    #cdef int Nt_int = 10
    
    cdef int Nt=parts*Nt_int
    #Nx = 2**6
    
    cdef int c=0
    
    # Deletes old memory-map and creates new one
    try:
        os.remove(str(name)+'_phi.mymemmap')
    except FileNotFoundError:
        pass
    cdef float [:,:,:] phi = np.memmap(str(name)+'_phi.mymemmap', dtype='float32', mode='w+', shape=(Nt,Nx,Nx))
    
    cdef float [:,:,:] phi_r
    
    cdef int start_n=1
    cdef int p
    
    
    
    
    for p in range(1, parts+1):
        # Load the data files
        phi_fname = str(name) + '_phi_data'+str(p)+'.npy'
        phi_r = np.load(phi_fname)
    
        #print (phi.shape[0])
        try:
            if (p==1):
                for i in range (0, phi_r.shape[0]):
    #               print (c,i)
                    phi[c,:,:]=phi_r[i,:,:]
                    c = c+1
                        
            else:
                for i in range (0, phi_r.shape[0]-1):
    #               print(c,i+1)
                    phi[c,:,:]=phi_r[i+1,:,:]
                    c=c+1
        except:
            print("Error in the loop")

            

