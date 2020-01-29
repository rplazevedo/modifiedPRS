#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:45:43 2019

@authors: rplazevedo, fferreira
"""

import numpy as np
import sys
import time
import configparser
import scipy.sparse

DTYPE = np.float32
def roll_sparse(x, shift, axis=0):
    ''' (IMPORTED FORM THE LIBROSA PACKAGE) Sparse matrix roll

    This operation is equivalent to ``numpy.roll``, but operates on sparse matrices.

    .. warning:: This function is deprecated in version 0.7.1.
                 It will be removed in version 0.8.0.

    Parameters
    ----------
    x : scipy.sparse.spmatrix or np.ndarray
        The sparse matrix input

    shift : int
        The number of positions to roll the specified axis

    axis : (0, 1, -1)
        The axis along which to roll.

    Returns
    -------
    x_rolled : same type as `x`
        The rolled matrix, with the same format as `x`

    See Also
    --------
    numpy.roll

    Examples
    --------
    >>> # Generate a random sparse binary matrix
    >>> X = scipy.sparse.lil_matrix(np.random.randint(0, 2, size=(5,5)))
    >>> X_roll = roll_sparse(X, 2, axis=0)  # Roll by 2 on the first axis
    >>> X_dense_r = roll_sparse(X.toarray(), 2, axis=0)  # Equivalent dense roll
    >>> np.allclose(X_roll, X_dense_r.toarray())
    True
    '''
    if not scipy.sparse.isspmatrix(x):
        return np.roll(x, shift, axis=axis)

    # shift-mod-length lets us have shift > x.shape[axis]
    if axis not in [0, 1, -1]:
        raise ValueError('axis must be one of (0, 1, -1)')

    shift = np.mod(shift, x.shape[axis])

    if shift == 0:
        return x.copy()

    fmt = x.format
    if axis == 0:
        x = x.tocsc()
    elif axis in (-1, 1):
        x = x.tocsr()

    # lil matrix to start
    x_r = scipy.sparse.lil_matrix(x.shape, dtype=x.dtype)

    idx_in = [slice(None)] * x.ndim
    idx_out = [slice(None)] * x_r.ndim

    idx_in[axis] = slice(0, -shift)
    idx_out[axis] = slice(shift, None)
    x_r[tuple(idx_out)] = x[tuple(idx_in)]

    idx_out[axis] = slice(0, shift)
    idx_in[axis] = slice(-shift, None)
    x_r[tuple(idx_out)] = x[tuple(idx_in)]

    return x_r.asformat(fmt)


def run():
    
    # read paratmeters from the config file
    config = configparser.ConfigParser()
    config.read('input.ini')
    
    Nx = int(config['PRS']['Lattice_size'])
    t0 = float(config['PRS']['initial_t'])
    w0 = float(config['PRS']['initial_width'])
    phi_0 = float(config['PRS']['initial_phi'])
    alpha = float(config['PRS']['alpha'])
    dt = float(config['PRS']['dt'])
    delta_x = float(config['PRS']['dx'])
    delta_y = float(config['PRS']['dy'])
    Nw_alpha = float(config['Parameters']['Nw_alpha'])
    alpha_1 = float(config['Parameters']['alpha_1'])
    alpha_2 = float(config['Parameters']['alpha_2'])
    alpha_e = float(config['Parameters']['alpha_e'])
    daeta = int(config['Parameters']['dlna/dlneta'])
    name = config['Parameters']['name']
    all_phi = config['Parameters']['all_phi']
        
    V0 = (np.pi**2)*(phi_0)**2/(2*w0**2)
    count_op = 0
    Nw = 0
        
    last_part = int(config['PRS']['sPRS_parts'])
    Nt = int(config['PRS']['points_per_part'])
    nparts = int(config['PRS']['mPRS_parts'])
    
    # initializes the arrays
    tspan= np.zeros((Nt),dtype=DTYPE)
    phi = []
    nabla_phi = []
    d_phi = []
    d_V = []
    #phi_w = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    v = np.zeros((Nt),dtype=DTYPE)
    N_walls = np.zeros((Nt),dtype=DTYPE)
    n_op = np.zeros((Nt),dtype=DTYPE)
    cond_pts = np.zeros((Nt),dtype=DTYPE)

    
    # loads previous data
    phi_r = np.load(str(name)+'_phi_data'+str(last_part)+'.npy')[-1]
    d_phi_r = np.load(str(name)+'_d_phi_data'+str(last_part)+'.npy')[-1]
    tspan_r = np.load(str(name)+'_tspan_data'+str(last_part)+'.npy')[-1]
    v_r = np.load(str(name)+'_vdata'+str(last_part)+'.npy')[-1]
    N_walls_r = np.load(str(name)+'_nwalls_data'+str(last_part)+'.npy')[-1]
    n_op_r = np.load(str(name)+'_nop_data'+str(last_part)+'.npy')[-1]
    
    
    start_time = time.time()    # initial time for measuring speed of calculation
    
    # orig_stdout = sys.stdout
    # f = open('out.txt', 'a')
    # sys.stdout = f
    
    
    print ("Running mPRS")
    print ("Reading the last part and continue the calulation")
    print("CONT:GRID POINTS on space interval (Nx=Ny): " + str(Nx))
    print("CONT:GRID POINTS on time interval (Nt): " + str(Nt))
    
    print("w0 = ", w0, ", V0 = ", V0, ", phi_0 = ", phi_0, ", t_0 =", t0, ", dt = ", dt,", alpha 1 =", alpha_1,
            ', alpha 2 =', alpha_2, ", alpha e =", alpha_e)
        
    tspan[0] = tspan_r
    v[0] = v_r
    N_walls[0] = N_walls_r
    n_op[0] = n_op_r
     
    # phi_r = None
    # d_phi_r = None
    tspan_r = None
    v_r = None
    N_walls_r = None
    n_op_r = None
    
    for i in range(1,Nt):
        tspan[i] = tspan[0]+dt*i
        #print(tspan[i])
        
    count_op_last = int(n_op[0])
    
    V = V0*((phi_r/phi_0)**2-1)**2      #potential
    cond = (np.abs(d_phi_r**2 
            +(0.5*(np.roll(phi_r,1,axis=0)-np.roll(phi_r,-1,axis=0))/delta_x)**2
            +(0.5*(np.roll(phi_r,1,axis=1)-np.roll(phi_r,-1,axis=1))/delta_y)**2
            +V) >= V0*alpha_e).nonzero()   
    
    phi_nonvac = phi_r[cond]
    
    # -1 goes to 0, all other val go to val+1
    phi_r = np.where(phi_r<0, 0.0, 2.0)
    phi_r[cond] = phi_nonvac + 1.0
    # save the array as a compressed sparse row matrix
    phi.append(scipy.sparse.csr_matrix(phi_r, dtype=DTYPE))
    # save the last d_phi as a compressed sparse row matrix
    d_phi.append(scipy.sparse.csr_matrix((d_phi_r[cond],cond), shape=(Nx,Nx)))
    
    phi_r = None
    d_phi_r = None
    
    print("Initialization is done--- %s seconds ---" % (time.time() - start_time))  
    
   
    for part in range(1,nparts+1):
        for n in range(0,Nt-1):
              
            step_start_time = time.time()    
            
            if part != 1 or n != 0:                            
            # Check where the energy is too low -> vacuum                 
                V = V0*((phi[n]/phi_0)**2-1)**2      #potential
                cond = (np.abs(d_phi[n]**2 
                        +(0.5*(roll_sparse(phi[n],1,axis=0)-roll_sparse(phi[n],-1,axis=0))/delta_x)**2
                        +(0.5*(roll_sparse(phi[n],1,axis=1)-roll_sparse(phi[n],-1,axis=1))/delta_y)**2
                        +V) >= V0*alpha_e).nonzero()   
                
                phi_nonvac = phi[n][cond]
                
                phi.append(scipy.sparse.csr_matrix(np.where(phi[n]<0, 0.0, 2.0)))
                phi[n][cond] = phi_nonvac + 1.0
                
            cond_pts[n] = cond[0].shape[0]
            
            # Calculates the gradient of phi   
            nabla_phi.append( ( (roll_sparse(phi[n],1,axis=0)+roll_sparse(phi[n],-1,axis=0)-2*phi[n])/(delta_x**2)+
                              (roll_sparse(phi[n],1,axis=1)+roll_sparse(phi[n],-1,axis=1)-2*phi[n])/(delta_y**2)))                       
                      
            delta = 0.5*alpha*dt*(daeta)/(tspan[n])                
            d_V.append(V0*4*( (phi[n]**3)/(phi_0**3)-(phi[n]/phi_0) ))
            #calculates d_phi
            d_phi.append(((1-delta)*d_phi[n]+dt*(nabla_phi[n]-d_V[n]))/(1+delta) )            

            phi.append(phi[n])
            phi[n+1] = phi[n] + dt*d_phi[n+1]
            # print(phi[n+1])
            # print(abs(phi[n+1]) <= Nw_alpha)
            wall_loc = (abs(phi[n+1]) <= Nw_alpha).nonzero()    #condition for a pt = wall
            # print(wall_loc)
            Nw = wall_loc[0].shape[0]                   # number of walls
            # print(Nw)
            # average velocity of the walls
            v[n+1] = (1.0/(2.0*Nw))*np.sum(
                                 ((d_phi[n+1][wall_loc])**2/(1.0))/
                                 (V0*( (phi[n+1][wall_loc]**2)/(phi_0**2)-1)**2))                
            
            count_op = count_op+1    
            N_walls[n+1] = Nw
        
            # print("n: " , n, "count: ", count_op, "part = ", part)
            # print("--- %s seconds ---" % (time.time() - step_start_time))
            
            n_op[n+1] =count_op + count_op_last
    
            #print("n_op[n+1]", n_op[n+1])
    
    #    print("Part " + str(part+int(last_part))+" is done--- %s seconds ---" % (time.time() - start_time))
    
        #numpart = part-1
    
    
        if (all_phi=='yes' or all_phi=='YES'):
    
            np.save(str(name) + '_phi_data' + str (part+int(last_part)) +'.npy', phi)
            np.save(str(name) + '_d_phi_data' + str(part+int(last_part))+ '.npy', d_phi)
    
    
        if (all_phi=='some' or all_phi=='SOME'):
    
            if (part+int(last_part)==int(nparts*0.1) or part+int(last_part)==int(nparts*0.2)
                or part+int(last_part)==int(nparts*0.4) or part+int(last_part)==int(nparts*0.30)
                or part+int(last_part)==int(nparts*0.5) or part+int(last_part)==int(nparts*0.6)
                or part+int(last_part)==int(nparts*0.7) or part+int(last_part)==int(nparts*0.8)
                or part+int(last_part)==int(nparts*0.9) or part+int(last_part)==int(nparts*1.0)):
    
                np.save(str(name) + '_phi_data' + str (part+int(last_part)) +'.npy', phi)
                np.save(str(name) + '_d_phi_data' + str(part+int(last_part))+ '.npy', d_phi)
    
    
        else:
    
            if (part == nparts):
    
                np.save(str(name) + '_phi_data' + str (part+int(last_part)) +'.npy', phi)
                np.save(str(name) + '_d_phi_data' + str(part+int(last_part))+ '.npy', d_phi)
    
        
    
        # np.save('d_V_data' + str(part+int(last_part)) + '.npy', d_V)
        np.save(str(name) + '_tspan_data' + str(part+int(last_part)) + '.npy', tspan)
        np.save(str(name) + '_vdata' + str(part+int(last_part)) + '.npy', v)
        np.save(str(name) + '_nwalls_data' + str(part+int(last_part)) + '.npy', N_walls)
        np.save(str(name) + '_nop_data' + str(part+int(last_part)) + '.npy', n_op)
        np.save(str(name) + '_exc_pts_data' + str(part+int(last_part)) + '.npy', -cond_pts/Nx**2+1)

    
    #    print("Part " + str(part+int(last_part))+" is saved--- %s seconds ---" % (time.time() - start_time))
    
    
        #Lets initialize the t again from the last point
        tspan[0] = tspan[-1]
        tspan[1:] = 0
    
        for i in range(1,Nt):
            tspan[i] = tspan[0]+dt*i
    
    
        d_V = []
    
        d_phi = [d_phi[-1]]
    
        phi = [phi[-1]]
    
        nabla_phi = []
        
        v[0] = v[-1]
        v[1:] = 0
    
        N_walls[0] = N_walls[-1]
        N_walls[1:] = 0
        
        cond_pts[0] = cond_pts[-1]
        cond_pts[1:] = 0
    
        n_op[0] = n_op[-1]
        n_op[1:] = 0
    
    
    #    print("Going to Part " + str(part+1)+"--- %s seconds ---" % (time.time() - start_time))
    
    
    print('Count= ', count_op)
    print("--- %s VERSION2 seconds ---" % (time.time() - start_time))
    
    
    # sys.stdout = orig_stdout
    # f.close()
