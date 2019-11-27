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
    cdef int Nt_int = int(config['PRS']['points_per_part'])
    cdef int parts = int(config['PRS']['sPRS_parts'])+int(config['PRS']['mPRS_parts'])
    cdef str name = str(config['Parameters']['name'])
    
    Nt=parts*Nt_int
    
    cdef int c=0
    
    # Creates memory-maps
    cdef float [:] v = np.memmap(str(name) + '_v.mymemmap', dtype='float32', mode='w+', shape=(Nt))
    cdef float [:] nwalls = np.memmap(str(name) + '_nwalls.mymemmap', dtype='float32', mode='w+', shape=(Nt))
    cdef float [:] nop = np.memmap(str(name) + '_nop.mymemmap', dtype='float32', mode='w+', shape=(Nt))
    
    
    cdef float [:] v_r
    cdef float [:] nwalls_r
    cdef float [:] nop_r
    
    
    cdef int start_n=1
    cdef int p
    
    
    
    for p in range(1, parts+1):
        # Load the data files
        v_fname = str(name) + '_vdata' +str(p)+'.npy'
        v_r = np.load(v_fname)
        nwalls_fname = str(name) + '_nwalls_data' + str(p) + '.npy'
        nwalls_r = np.load(nwalls_fname)
        try:
            nop_fname = str(name) + '_nop_data' + str(p) + '.npy'
            nop_r = np.load(nop_fname)
        except:
            pass
    
        if (p==1):
            for i in range (0, v_r.shape[0]):
    #           print (c,i)
                v[c]=v_r[i]
                nwalls[c]=nwalls_r[i]
                try:
                    nop[c] = nop_r[i]
                except: 
                    pass
                c = c+1
                        
        else:
            for i in range (0, v_r.shape[0]-1):
    #           print(c,i+1)
    
                v[c]=v_r[i+1]
                nwalls[c]=nwalls_r[i]
                try:
                    nop[c]=nop_r[i]
                except:
                    pass
    
                c=c+1
        
    # Save new .txt files
    number = 0
    while os.path.isfile(str(name) +'_'+str(number)+ '_v.txt'):
        number+=1
    np.savetxt(str(name) +'_'+str(number)+ '_v.txt', v)
    np.savetxt(str(name) +'_'+str(number)+ '_nwalls.txt', nwalls)
    try:
        np.savetxt(str(name) +'_'+str(number)+ '_nop.txt', nop)
    except:
        pass


