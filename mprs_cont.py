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

DTYPE = np.float32

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
    phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cond = np.zeros((Nx,Nx),dtype=DTYPE)
    nabla_phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    d_phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    d_V = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    #phi_w = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    v = np.zeros((Nt),dtype=DTYPE)
    N_walls = np.zeros((Nt),dtype=DTYPE)
    n_op = np.zeros((Nt),dtype=DTYPE)
    cond_pts = np.zeros((Nt),dtype=DTYPE)

    
    # loads previous data
    phi_r = np.load(str(name)+'_phi_data'+str(last_part)+'.npy')
    d_phi_r = np.load(str(name)+'_d_phi_data'+str(last_part)+'.npy')
    tspan_r = np.load(str(name)+'_tspan_data'+str(last_part)+'.npy')
    v_r = np.load(str(name)+'_vdata'+str(last_part)+'.npy')
    N_walls_r = np.load(str(name)+'_nwalls_data'+str(last_part)+'.npy')
    n_op_r = np.load(str(name)+'_nop_data'+str(last_part)+'.npy')
    
    
    start_time = time.time()    # initial time for measuring speed of calculation
    
    orig_stdout = sys.stdout
    f = open('out.txt', 'a')
    sys.stdout = f
    
    
    print ("Running mPRS")
    print ("Reading the last part and continue the calulation")
    print("CONT:GRID POINTS on space interval (Nx=Ny): " + str(Nx))
    print("CONT:GRID POINTS on time interval (Nt): " + str(Nt))
    
    print("w0 = ", w0, ", V0 = ", V0, ", phi_0 = ", phi_0, ", t_0 =", t0, ", dt = ", dt,", alpha 1 =", alpha_1,
            ', alpha 2 =', alpha_2, ", alpha e =", alpha_e)
        
    phi[0] = phi_r[-1]
    d_phi[0] = d_phi_r[-1]
    tspan[0] = tspan_r[-1]
    v[0] = v_r[-1]
    N_walls[0] = N_walls_r[-1]
    n_op[0] = n_op_r[-1]
     
    phi_r = None
    d_phi_r = None
    tspan_r = None
    v_r = None
    N_walls_r = None
    n_op_r = None
    
    for i in range(1,Nt):
        tspan[i] = tspan[0]+dt*i
        #print(tspan[i])
        
    count_op_last = int(n_op[0])
        
    print("Initialization is done--- %s seconds ---" % (time.time() - start_time))  
    
    for part in range(1,nparts+1):
        for n in range(0,Nt-1):
              
            step_start_time = time.time()    
                    
   
            # Check where the energy is too low -> vacuum                 
            V = V0*((phi[n]/phi_0)**2-1)**2      #potential
            cond = (np.abs(d_phi[n]**2 
                    +(0.5*(np.roll(phi[n],1,axis=0)-np.roll(phi[n],-1,axis=0))/delta_x)**2
                    +(0.5*(np.roll(phi[n],1,axis=1)-np.roll(phi[n],-1,axis=1))/delta_y)**2
                    +V) >= V0*alpha_e).nonzero()   
            
            # phi_nonvac = phi[n][cond]
            
            # phi[n] = np.where(phi[n]<0,-1,1)
            # phi[n][cond] = phi_nonvac
            
            # Old conditions - ONLY USE FOR COMPARISONS 
            # cond_1 = (np.abs(phi[n]) > alpha_1)*1
            # cond_2 = (np.abs(d_phi[n]**2 +(0.5*(np.roll(phi[n],1,axis=0)-np.roll(phi[n],-1,axis=0))/delta_x)**2
            #                     +(0.5*(np.roll(phi[n],1,axis=1)-np.roll(phi[n],-1,axis=1))/delta_y)**2) < alpha_2)*1
            # cond_1_xf = (np.abs(np.roll(phi[n],1,axis=0)) > alpha_1)*1
            # cond_1_xb = (np.abs(np.roll(phi[n],-1,axis=0)) > alpha_1)*1
            # cond_1_yf = (np.abs(np.roll(phi[n],1,axis=1)) > alpha_1)*1
            # cond_1_yb = (np.abs(np.roll(phi[n],-1,axis=1)) > alpha_1)*1               
            # cond = cond_1 * cond_2 * cond_1_xf * cond_1_xb * cond_1_yf * cond_1_yb
            
            cond_pts[n] = cond[0].shape[0]
            
            # Calculates the gradient of phi   
            nabla_phi[n][cond] = ( (np.roll(phi[n],1,axis=0)[cond]+np.roll(phi[n],-1,axis=0)[cond]-2*phi[n][cond])/(delta_x**2)+
                              (np.roll(phi[n],1,axis=1)[cond]+np.roll(phi[n],-1,axis=1)[cond]-2*phi[n][cond])/(delta_y**2))                          
                      
            delta = 0.5*alpha*dt*(daeta)/(tspan[n])                
            d_V[n][cond]=V0*4*(phi[n][cond]/phi_0)*( (phi[n][cond]**2)/(phi_0**2)-1 )
            #calculates d_phi where the condition is met
            d_phi[n+1][cond] = ((1-delta)*d_phi[n][cond]
                                      +dt*(nabla_phi[n][cond]-d_V[n][cond]))/(1+delta)                

            phi[n+1] = phi[n]
            phi[n+1][cond] = phi[n][cond] + dt*d_phi[n+1][cond]
            
            wall_loc = (abs(phi[n+1]) <= Nw_alpha).nonzero()    #condition for a pt = wall
            Nw = wall_loc[0].shape[0]                   # number of walls
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
    
    
        d_V[:,:] = 0
    
        d_phi[0,:,:] =  d_phi[-1,:,:]
        d_phi[1:,:,:] =  0
    
        phi[0,:,:] =  phi[-1,:,:]
        phi[1:,:,:] =  0
    
        nabla_phi[:,:] = 0
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
    
    
    sys.stdout = orig_stdout
    f.close()
