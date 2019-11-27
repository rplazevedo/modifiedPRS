#VERSION 1.2
#07/06/2019
#Author: Fabio

import numpy as np
import random
import sys
import time
import configparser
    
    
cimport numpy as np    
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def run():
  
    config = configparser.ConfigParser()
    config.read('input.ini') 
    
    cdef int Nx = int(config['PRS']['Lattice_size'])
    cdef float t0 = float(config['PRS']['initial_t'])
    cdef float w0 = float(config['PRS']['initial_width'])
    cdef float phi_0 = float(config['PRS']['initial_phi'])
    cdef float alpha = float(config['PRS']['alpha'])
    cdef float dt = float(config['PRS']['dt'])
    cdef float delta_x = float(config['PRS']['dx'])
    cdef float delta_y = float(config['PRS']['dy'])
    cdef float Nw_alpha = float(config['Parameters']['Nw_alpha'])
    cdef int daeta = int(config['Parameters']['dlna/dlneta'])
    cdef str name = config['Parameters']['name']
    all_phi = config['Parameters']['all_phi']
    
    
    cdef float pi = np.pi
    cdef float V0 = (np.pi**2)*(phi_0)**2/(2*w0**2)
    cdef float delta
    cdef long long int count_op = 0
    cdef int i,j,n,parti,start,w
    cdef float Nw = 0
    
    cdef int Nt = int(config['PRS']['points_per_part'])
    cdef int nparts = int(config['PRS']['sPRS_parts'])
    cdef float alpha2 = float(config['Parameters']['alpha2'])
    
    cdef float[:] tspan= np.zeros((Nt),dtype=DTYPE)
    cdef float[:, :, :] phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:, :, :] nabla_phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:, :, :] d_phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:, :, :] d_V = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    #cdef float[:, :, :] phi_w = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:] v = np.zeros((Nt),dtype=DTYPE)
    cdef float[:] N_walls = np.zeros((Nt),dtype=DTYPE)
    cdef float[:] n_op = np.zeros((Nt),dtype=DTYPE)
    
    
    start_time = time.time()
    
    orig_stdout = sys.stdout
    f = open('out.txt', 'a')
    sys.stdout = f
    
    print ("Running with cython")
    print("GRID POINTS on space interval (Nx=Ny): " + str(Nx))
    print("GRID POINTS on time interval (Nt): " + str(Nt))
    print("w0 = ", w0, ", V0 = ", V0, ", phi_0 = ", phi_0, "t_0 =", t0, "dt = ", dt)
    
    
    tspan[0]=1
    
    for i in range(1,Nt):
        tspan[i] = t0+dt*i
    
    
    #cdef float[:, :, :] phi = np.memmap('phi.mymemmap', dtype='float32', mode='w+', shape=(Nt,Nx,Nx)) 
    #cdef float[:, :, :] nabla_phi = np.memmap('nabla_phi.mymemmap', dtype='float32', mode='w+', shape=(Nt,Nx,Nx))
    #cdef float[:, :, :] d_phi = np.memmap('d_phi.mymemmap', dtype='float32', mode='w+', shape=(Nt,Nx,Nx))
    #cdef float[:, :, :] d_V=  np.memmap('dV.mymemmap', dtype='float32', mode='w+', shape=(Nt,Nx,Nx))
    #cdef float[:] Ek = np.memmap('Ek.mymemmap', dtype='float32', mode='w+', shape=(Nt))
    
    
    
    #Condition: initial value of phi_ij lies between -1 and 1
    for i in range(0,Nx):
        for j in range(0,Nx):
            phi[0,i,j] = random.uniform(-1,1)
            
            if (abs(phi[0,i,j])<= Nw_alpha):
                #phi_w[0,i,j]=1
                v[0] = v[0] + ((d_phi[0,i,j])**2/(1))/(V0*(phi[0,i,j]**2-1)**2)
                Nw=Nw+1
    
                #print(((d_phi[0,i,j])**2/(8*pi)))
                #print((V0*(phi[0,i,j]**2-1))**2)
    
            #else:
                #phi_w[0,i,j]=0
    
    
            #phi[0,i,j] = t0
            #print("Phi", 0, i, j, phi[0,i,j], sep="  ")
            #print(phi)
    #phi[0,-1,:]=phi[0,0,:]
    #phi[0,:,-1]=phi[0,:,0]
    
    N_walls[0] = Nw
    n_op[0] = 1
    v[0] = (1/(2*Nw))*v[0]
    Nw=0
    
    #Lets calculate the initial values for nabla(phi_ij)
    for n in range(0,1):
        for i in range(0,Nx):
            for j in range(0,Nx):
                if (i==0 and j==0):
                    #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,Nx-1,j]+phi[n,i,Nx-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,Nx-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,Nx-1])/(delta_y**2) 
                
                if (i==0 and j>0 and j<Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,Nx-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,Nx-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                
                if (j==0 and i>0 and i<Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,Nx-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,Nx-1])/(delta_y**2)
    
                if (j==Nx-1 and i>0 and i<Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,0]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,0]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                if (i==Nx-1 and j>0 and j<Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,0,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,0,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                if (i==Nx-1 and j==Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,0,j]+phi[n,i,0]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,0,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,0]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
    
                if (i>0 and i<Nx-1 and j>0 and j<Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                if (i==0 and j==Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,0]+phi[n,Nx-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,Nx-1,j])/(delta_x**2) + (phi[n,i,0]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                if (j==0 and i==Nx-1):
                    #nabla_phi[n,i,j] =  phi[n,0,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,Nx-1]-4*phi[n,i,j]
                    nabla_phi[n,i,j] = (phi[n,0,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,Nx-1])/(delta_y**2)
    
    
                #print("Nabla_Phi", n, i, j, nabla_phi[n,i,j], sep="  ")
    
    
    #phi[0,-1,:]=0
    #phi[0,:,-1]=0
    
    #Initial value of d_phi is zero
    
    
    #Computing the next d_phi
    for n in range(1,2):
        for i in range (0,Nx):
            for j in range (0,Nx):
                delta = 0.5*alpha*dt*(daeta)/(tspan[n-1])
                d_V[n-1,i,j]=V0*4*(phi[n-1,i,j]/phi_0)*( (phi[n-1,i,j]**2)/(phi_0**2)-1 )
                d_phi[n,i,j] = ((1-delta)*d_phi[n-1,i,j]+dt*(nabla_phi[n-1,i,j]-d_V[n-1,i,j]))/(1+delta)
                #Ek[n] = Ek[n] + 1/(8*pi) * d_phi[n,i,j]**2
    
    
    #Computing the next phi_ij
    for n in range(0,1):
        for i in range (0,Nx):
            for j in range (0,Nx):
                phi[n+1,i,j] = phi[n,i,j] + dt*d_phi[n+1,i,j]
    
                if (abs(phi[n+1,i,j])<= Nw_alpha):
    
                    v[n+1] = v[n+1] + ((d_phi[n+1,i,j])**2/(1.0))/(V0*( (phi[n+1,i,j]**2)/(phi_0**2)-1)**2)
                    Nw=Nw+1
                    #print('dphi**2=',((d_phi[n+1,i,j])**2/(8*pi)))
                    #print('num =',V0*(phi[n+1,i,j]**2-1)**2)
    
                #else:
                    #phi_w[n+1,i,j]=0
    
        v[n+1]=(1/(2*Nw))*v[n+1]    
    
        N_walls[n+1] = Nw
        n_op[n+1] = n_op[n]+1
    
    
    Nw=0
    
    print("Initialization is done--- %s seconds ---" % (time.time() - start_time))
    
    
    #Loop
    for part in range(1,nparts+1):
        
        if (part ==1):
            start=1
        else:
            start=0
    
        for n in range(start,Nt-1):
            for i in range(0,Nx):
                for j in range(0,Nx):
                
                    if (i==0 and j==0):
                        #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,Nx-1,j]+phi[n,i,Nx-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,Nx-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,Nx-1])/(delta_y**2)
    
                    if (i==0 and j>0 and j<Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,Nx-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,Nx-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                    if (j==0 and i>0 and i<Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,Nx-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,Nx-1])/(delta_y**2)
    
                    if (j==Nx-1 and i>0 and i<Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,0]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,0]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                    if (i==Nx-1 and j>0 and j<Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,0,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,0,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                    if (i==Nx-1 and j==Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,0,j]+phi[n,i,0]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,0,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,0]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                    if (i>0 and i<Nx-1 and j>0 and j<Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                         nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                    if (i==0 and j==Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,i+1,j]+phi[n,i,0]+phi[n,Nx-1,j]+phi[n,i,j-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,Nx-1,j])/(delta_x**2) + (phi[n,i,0]-2*phi[n,i,j]+phi[n,i,j-1])/(delta_y**2)
    
                    if (j==0 and i==Nx-1):
                        #nabla_phi[n,i,j] =  phi[n,0,j]+phi[n,i,j+1]+phi[n,i-1,j]+phi[n,i,Nx-1]-4*phi[n,i,j]
                        nabla_phi[n,i,j] = (phi[n,0,j]-2*phi[n,i,j]+phi[n,i-1,j])/(delta_x**2) + (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,Nx-1])/(delta_y**2)
    
    
                    delta = 0.5*alpha*dt*(daeta)/(tspan[n])
                    d_V[n,i,j]=V0*4*(phi[n,i,j]/(phi_0**2))*( (phi[n,i,j]**2)/(phi_0**2)-1 )
                    d_phi[n+1,i,j] = ((1-delta)*d_phi[n,i,j]+dt*(nabla_phi[n,i,j]-d_V[n,i,j]) )/(1+delta)
    
                    #Ek[n+1] = Ek[n+1] +  1/(8*pi)*(d_phi[n+1,i,j])**2
                    phi[n+1,i,j] = phi[n,i,j] + dt*d_phi[n+1,i,j]
                
                    if (abs(phi[n+1,i,j])<= Nw_alpha):
                        #phi_w[n+1,i,j]=1
                        v[n+1] = v[n+1] + ((d_phi[n+1,i,j])**2/(1.0))/(V0*( (phi[n+1,i,j]**2)/(phi_0**2)-1)**2)
                        Nw=Nw+1
    
                    #else:
                        #phi_w[n+1,i,j]=0
    
                    count_op = count_op +1
    
            #print(Nw)
            try:
                v[n+1]=(1.0/(2.0*Nw))*v[n+1]    
            except:
                print("Warning at n = ", part)
                print("There are no walls")
                v[n+1]=0
    
            N_walls[n+1]=Nw
            n_op[n+1] = 2 + count_op
            Nw=0
           
            #print("count_op: ", count_op)
            #print("n_op[n+1] = ", n_op[n+1], "part= ", part)
    
    #    print("Part " + str(part)+" is done--- %s seconds ---" % (time.time() - start_time))
    
    
        if (all_phi=='yes' or all_phi=='YES'):
    
            np.save(str(name) + '_phi_data' + str (part) +'.npy', phi)
            np.save(str(name) + '_d_phi_data' + str(part) + '.npy', d_phi)
    
        else:
    
            if (part == nparts):
    
                np.save(str(name) + '_phi_data' + str (part) +'.npy', phi)
                np.save(str(name) + '_d_phi_data' + str(part)+ '.npy', d_phi)
            
        
    
        #np.save('d_V_data' + str(part) + '.npy', d_V)
        np.save(str(name) + '_tspan_data' + str(part) + '.npy', tspan)
        #np.save('phiw_data' + str (part) +'.npy', phi_w)
        np.save(str(name) + '_vdata' + str(part) +'.npy', v)
        np.save(str(name) + '_nwalls_data' + str(part) + '.npy', N_walls)
        np.save(str(name) + '_nop_data' + str(part) + '.npy', n_op)
    
    #    print("Part " + str(part)+" is saved--- %s seconds ---" % (time.time() - start_time))
    
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
    
        n_op[0] = n_op[-1]
        n_op[1:] = 0
    
    #    print("Going to Part " + str(part+1)+"--- %s seconds ---" % (time.time() - start_time))
    
    #np.save(str(name) + '_Ek_data.npy', Ek)
    
    print('Count= ', count_op)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    sys.stdout = orig_stdout
    f.close()
