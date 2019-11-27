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
    cdef float alpha1 = float(config['Parameters']['alpha1'])
    cdef float alpha2 = float(config['Parameters']['alpha2'])
    cdef int daeta = int(config['Parameters']['dlna/dlneta'])
    cdef str name = config['Parameters']['name']
    all_phi = config['Parameters']['all_phi']
    
    
    cdef float pi = np.pi
    cdef float V0 = (np.pi**2)*(phi_0)**2/(2*w0**2)
    cdef float delta
    cdef long long int  count_op = 0
    cdef long long int count_op_last
    cdef int i,j,n,part
    cdef float Nw = 0
    
    
    cdef int last_part = int(config['PRS']['sPRS_parts'])
    cdef int Nt = int(config['PRS']['points_per_part'])
    cdef int nparts = int(config['PRS']['mPRS_parts'])
    
    cdef str string_phi = str(name)+'_phi_data'+str(last_part)+'.npy'
    cdef str string_dphi = str(name)+'_d_phi_data'+str(last_part)+'.npy'
    
    
    cdef float[:] tspan= np.zeros((Nt),dtype=DTYPE)
    cdef float[:, :, :] phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:, :, :] nabla_phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:, :, :] d_phi = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:, :, :] d_V = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    #cdef float[:, :, :] phi_w = np.zeros((Nt,Nx,Nx),dtype=DTYPE)
    cdef float[:] v = np.zeros((Nt),dtype=DTYPE)
    cdef float[:] N_walls = np.zeros((Nt),dtype=DTYPE)
    cdef float[:] n_op = np.zeros((Nt),dtype=DTYPE)
    
    
    cdef float[:, :, :] phi_r = np.load(str(name)+'_phi_data'+str(last_part)+'.npy')
    cdef float[:, :, :] d_phi_r = np.load(str(name)+'_d_phi_data'+str(last_part)+'.npy')
    cdef float[:] tspan_r = np.load(str(name)+'_tspan_data'+str(last_part)+'.npy')
    cdef float[:] v_r = np.load(str(name)+'_vdata'+str(last_part)+'.npy')
    cdef float[:] N_walls_r = np.load(str(name)+'_nwalls_data'+str(last_part)+'.npy')
    cdef float[:] n_op_r = np.load(str(name)+'_nop_data'+str(last_part)+'.npy')
    
    
    start_time = time.time()
    
    orig_stdout = sys.stdout
    f = open('out.txt', 'a')
    sys.stdout = f
    
    
    print ("Running with cython")
    print ("Reading the last part and continue the calulation (optimized)")
    print("CONT:GRID POINTS on space interval (Nx=Ny): " + str(Nx))
    print("CONT:GRID POINTS on time interval (Nt): " + str(Nt))
    
    print("w0 = ", w0, ", V0 = ", V0, ", phi_0 = ", phi_0, "t_0 =", t0, "dt = ", dt, "alpha 1 =", alpha1,
            'alpha 2 =', alpha2)
    
    
    phi = phi_r
    nabla_phi[:,:]=0
    d_V[:,:] = 0
    d_phi = d_phi_r
    tspan = tspan_r
    v = v_r
    N_walls = N_walls_r
    n_op = n_op_r
    
    
    phi_r = None
    d_phi_r = None
    tspan_r = None
    v_r = None
    N_walls_r = None
    n_op_r = None
    
    tspan[0] = tspan[-1]
    tspan[1:] = 0
    
    phi[0] = phi[-1]
    phi[1:] = 0
    
    d_phi[0] = d_phi[-1]
    d_phi[1:] = 0
    
    v[0] = v[-1]
    v[1:] = 0
    
    N_walls[0] = N_walls[-1]
    N_walls[1:] = 0
    
    n_op[0] = n_op[-1]
    n_op[1:] = 0
    
    
    
    for i in range(1,Nt):
        tspan[i] = tspan[0]+dt*i
        #print(tspan[i])
    
    Nw=0
    
    count_op_last = int(n_op[0])
    
    print("Initialization is done--- %s seconds ---" % (time.time() - start_time))
    
    
    for part in range(1,nparts+1):
        for n in range(0,Nt-1):
            for i in range(0,Nx):
                for j in range(0,Nx):
    
                    if (phi[n,i,j] <=alpha1 and phi[n,i,j] >=-alpha1):
                        
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
                        d_V[n,i,j]=V0*4*(phi[n,i,j]/phi_0)*( (phi[n,i,j]**2)/(phi_0**2)-1 )
                        d_phi[n+1,i,j] = ((1-delta)*d_phi[n,i,j]+dt*(nabla_phi[n,i,j]-d_V[n,i,j]))/(1+delta)
                        
                        #Ek[n+1] = Ek[n+1] +  1/(8*pi)*(d_phi[n+1,i,j])**2
                        phi[n+1,i,j] = phi[n,i,j] + dt*d_phi[n+1,i,j]
    
    
    
                        if (abs(phi[n+1,i,j])<= Nw_alpha):
                            #phi_w[n+1,i,j]=1
                            v[n+1] = v[n+1] + ((d_phi[n+1,i,j])**2/(1.0))/(V0*( (phi[n+1,i,j]**2)/(phi_0**2)-1)**2)
                            Nw=Nw+1
    
                    #else:
                        #phi_w[n+1,i,j]=0
    
    
                        count_op = count_op+1
    
                    #if (phi[n,i,j]>alpha1 or phi[n,i,j]<-alpha1):
                    elif (abs(phi[n,i,j])>alpha1):
                       
    
                        '''
                        try:
                            if (n>30 and n<50):
                                    print(i,j,n,d_phi[n,i,j]**2, (phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j]), (phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1]))
                                    print(abs(d_phi[n,i,j]**2+(phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])+(phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1]))<alpha2)
                        except:
                            pass
                        '''
    
    
                        
                        #(phi[n,i+1,j] - phi[n,i-1,j] + phi[n,i,j+1] - phi[n,i,j-1])**2
    
    
                       # phi[n+1,i,j] = phi[n,i,j]
    
                        if (i==0 and j==0  
                                and abs(phi[n,Nx-1,j])>alpha1 
                                and abs(phi[n,i+1,j])>alpha1
                                and abs(phi[n,i,Nx-1])>alpha1 
                                and abs(phi[n,i,j+1])>alpha1
                                and (abs(d_phi[n,i,j]**2 + (((1.0/2)*(phi[n,i+1,j]-phi[n,Nx-1,j]))**2+((1.0/2)*(phi[n,i,j+1]-phi[n,i,Nx-1]))**2))<alpha2) ):
    
    
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
     
                            
                            else:
                                phi[n+1,i,j] = -1.0
    
    
                            d_phi[n+1,i,j] = 0.0
                        
                        
                        #Ek[n+1] = Ek[n+1]+1/(8*pi)*(d_phi[n+1,i,j])**2
                
        
                        elif (i==0 and j>0 and j<Nx-1 
                                and abs(phi[n,Nx-1,j])>alpha1 
                                and abs(phi[n,i+1,j])>alpha1
                                and abs(phi[n,i,j-1])>alpha1 
                                and abs(phi[n,i,j+1])>alpha1
                                and (abs(d_phi[n,i,j]**2 +( ((1.0/2)*(phi[n,i+1,j]-phi[n,Nx-1,j]))**2+((1.0/2)*(phi[n,i,j+1]-phi[n,i,j-1]))**2))<alpha2)  ):
    
                   
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
    
    
                            else:
                                phi[n+1,i,j] = -1.0
    
                            
                            d_phi[n+1,i,j] = 0.0
                        
                        
    
                        elif (j==0 and i>0 and i<Nx-1 
                                and abs(phi[n,i-1,j])>alpha1 
                                and abs(phi[n,i+1,j])>alpha1 
                                and abs(phi[n,i,Nx-1])>alpha1 
                                and abs(phi[n,i,j+1])>alpha1
                                and (abs(d_phi[n,i,j]**2 +( ((1.0/2)*(phi[n,i+1,j]-phi[n,i-1,j]))**2+((1.0/2)*(phi[n,i,j+1]-phi[n,i,Nx-1]))**2))<alpha2)  ):
    
    
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
     
     
                            else:
                                phi[n+1,i,j] = -1.0
    
                            d_phi[n+1,i,j] = 0.0
                        
    
                        elif (i==Nx-1 and j>0 and j<Nx-1 
                                and abs(phi[n,i-1,j])>alpha1 
                                and abs(phi[n,0,j])>alpha1 
                                and abs(phi[n,i,j-1])>alpha1 
                                and abs(phi[n,i,j+1])>alpha1 
                                and (abs(d_phi[n,i,j]**2 +( ((1.0/2)*(phi[n,0,j]-phi[n,i-1,j]))**2 +((1.0/2)*(phi[n,i,j+1]-phi[n,i,j-1]))**2))<alpha2)  ):
    
                            
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
    
                            else:
                                phi[n+1,i,j] = -1.0
    
                            
                            d_phi[n+1,i,j] = 0.0        
                        
                        
                        
                        elif (j==Nx-1 and i>0 and i<Nx-1 
                                and abs(phi[n,i-1,j])>alpha1 
                                and abs(phi[n,i+1,j])>alpha1 
                                and abs(phi[n,i,j-1])>alpha1 
                                and abs(phi[n,i,0])>alpha1
                                and (abs(d_phi[n,i,j]**2 +( ((1.0/2)*(phi[n,i+1,j]-phi[n,i-1,j]))**2 +((1.0/2)*(phi[n,i,0]-phi[n,i,j-1]))**2))<alpha2)  ):
    
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
     
    
                            else:
                                phi[n+1,i,j] = -1.0
    
                            
                            d_phi[n+1,i,j] = 0.0
                        
                        
    
                        elif (i==Nx-1 and j==Nx-1 
                            and abs(phi[n,i-1,j])>alpha1 
                            and abs(phi[n,0,j])>alpha1 
                            and abs(phi[n,i,j-1])>alpha1 
                            and abs(phi[n,i,0])>alpha1
                            and (abs(d_phi[n,i,j]**2 +( ((1.0/2)*(phi[n,0,j]-phi[n,i-1,j]))**2 +((1.0/2)*(phi[n,i,0]-phi[n,i,j-1]))**2))<alpha2)  ):
    
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
    
    
                            else:
                                phi[n+1,i,j] = -1.0
    
    
                            d_phi[n+1,i,j] = 0.0
                           
                        
                        
                        
                        elif (i>0 and i<Nx-1 and j>0 and j<Nx-1 
                                and abs(phi[n,i-1,j])>alpha1 
                                and abs(phi[n,i+1,j])>alpha1 
                                and abs(phi[n,i,j-1])>alpha1 
                                and abs(phi[n,i,j+1])>alpha1
                                and (abs(d_phi[n,i,j]**2 + ( ((1.0/2)*(phi[n,i+1,j]-phi[n,i-1,j]))**2+((1.0/2)*(phi[n,i,j+1]-phi[n,i,j-1]))**2))<alpha2)  ):
                                
    
                                    
                            
    
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
    
    
                            else:
    
                                phi[n+1,i,j] = -1.0
                          
                            d_phi[n+1,i,j] = 0.0
    
                            #print("COND:", abs(d_phi[n,i,j]**2)+(phi[n,i+1,j]-2*phi[n,i,j]+phi[n,i-1,j])+(phi[n,i,j+1]-2*phi[n,i,j]+phi[n,i,j-1]))
                            
                    
                        elif (i==0 and j==Nx-1 
                                and abs(phi[n,Nx-1,j])>alpha1 
                                and abs(phi[n,i+1,j])>alpha1 
                                and abs(phi[n,i,j-1])>alpha1 
                                and abs(phi[n,i,0])>alpha1
                                and(abs(d_phi[n,i,j]**2 + ( ((1.0/2)*(phi[n,i+1,j]-phi[n,Nx-1,j]))**2+((1.0/2)*(phi[n,i,0]-phi[n,i,j-1]))**2))<alpha2)  ):
                            
                            
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
     
                        
                            else:
                                phi[n+1,i,j] = -1.0
    
                            
                            d_phi[n+1,i,j] = 0.0
                        
                            
                            
                        elif (j==0 and i==Nx-1 
                                and abs(phi[n,i-1,j])>alpha1 
                                and abs(phi[n,0,j])>alpha1 
                                and abs(phi[n,i,Nx-1])>alpha1 
                                and abs(phi[n,i,j+1])>alpha1 
                                and (abs(d_phi[n,i,j]**2 +( ((1.0/2)*(phi[n,0,j]-phi[n,i-1,j]))**2+((1.0/2)*(phi[n,i,j+1]-phi[n,i,Nx-1]))**2))<alpha2)  ):
                                                
                            
                            if (phi[n,i,j] >0):
    
                                phi[n+1,i,j] = 1.0
     
                        
                            else:
                                phi[n+1,i,j] = -1.0
    
                            
                            d_phi[n+1,i,j] = 0.0
    
     
                        
                        else:
    
                            count_op=count_op+1
    
    
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
                            d_V[n,i,j]=V0*4*(phi[n,i,j]/phi_0)*( (phi[n,i,j]**2)/(phi_0**2)-1 )
                            d_phi[n+1,i,j] = ((1-delta)*d_phi[n,i,j]+dt*(nabla_phi[n,i,j]-d_V[n,i,j]))/(1+delta)
                            #Ek[n+1] = Ek[n+1] +  1/(8*pi)*(d_phi[n+1,i,j])**2
                            phi[n+1,i,j] = phi[n,i,j] + dt*d_phi[n+1,i,j]
    
    
    
                        if (abs(phi[n+1,i,j])<= Nw_alpha):
                            #phi_w[n+1,i,j]=1
                        
                            v[n+1] = v[n+1] + ((d_phi[n+1,i,j])**2/(1.0))/(V0*( (phi[n+1,i,j]**2)/(phi_0**2)-1)**2)
                            Nw=Nw+1
    
    
                        #else:
                            #phi_w[n+1,i,j]=0
    
                    
            v[n+1] = (1.0/(2.0*Nw))*v[n+1]
    
            N_walls[n+1] = Nw
    
            Nw=0
        
            #print("n: " , n, "count: ", count_op, "part = ", part)
          
            
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
    
        n_op[0] = n_op[-1]
        n_op[1:] = 0
    
    
    #    print("Going to Part " + str(part+1)+"--- %s seconds ---" % (time.time() - start_time))
    
    
    print('Count= ', count_op)
    print("--- %s VERSION2 seconds ---" % (time.time() - start_time))
    
    
    sys.stdout = orig_stdout
    f.close()
