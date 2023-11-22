#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.fftpack as fft
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

"""
Description: Calculates x component (w1) of v^(n) in Eq. 14.42 in Peskin 1996.
Arguments:
    u: MxN numpy array
    v: MxN numpy array
    f1: MxN numpy array
    f2: MxN numpy array
    rho: 1x1 float (constant density)
    dx: 1x1 float
    dt: 1x1 float
Returns:
    w1: MxN numpy array

"""

def calculate_w1(u, v, f1, f2, rho, dx, dt):
    M, N = u.shape
    w1 = 1j*np.zeros((M,N))
    
    i = 0
    while (i < M):
        j = 0
        while (j < N):
            i_minus_1 = i-1
            i_plus_1 = i+1
            j_minus_1 = j-1
            j_plus_1 = j+1
            
            # Implements periodic boundary conditions
            if (j == N-1):
                j_plus_1 = 0
            if (j == 0):
                j_minus_1 = N-1
            if (i == M-1):
                i_plus_1 = 0
            if (i == 0):
                i_minus_1 = M-1
            
            # Define upwind finite differences of u and v based on Eq. 14.34
            # in Peskin 1996.
            
            Du1 = 0
            Du2 = 0
            
            if (u[i, j] < 0):
                Du1 = (u[i_plus_1, j] - u[i,j])/dx
                Du2 = (u[i, j_plus_1] - u[i,j])/dx
            else:
                Du1 = (u[i,j] - u[i_minus_1,j])/dx
                Du2 = (u[i,j] - u[i,j_minus_1])/dx
        
            
            w1[i,j] = u[i, j] - (dt/rho)*(u[i,j]*Du1 + v[i,j]*Du2)+(dt/rho)*f1[i,j]
            j+=1
            
        i+=1
    
    return w1

"""
Description: Calculates y component (w2) of v^(n) in Eq. 14.42 in Peskin 1996.
Arguments:
    u: MxN numpy array
    v: MxN numpy array
    f1: MxN numpy array
    f2: MxN numpy array
    rho: 1x1 float (constant density)
    dx: 1x1 float
    dt: 1x1 float
Returns:
    w1: MxN numpy array

"""

def calculate_w2(u, v, f1, f2, rho, dx, dt):
    M, N = u.shape
    w2 = 1j*np.zeros((M,N))
    
    i = 0
    while (i < M):
        j = 0
        while (j < N):
            i_minus_1 = i-1
            i_plus_1 = i+1
            j_minus_1 = j-1
            j_plus_1 = j+1
            
            # Implements periodic boundary conditions
            if (j == N-1):
                j_plus_1 = 0
            if (j == 0):
                j_minus_1 = N-1
            if (i == M-1):
                i_plus_1 = 0
            if (i == 0):
                i_minus_1 = M-1
            
            # Define upwind finite differences of u and v based on Eq. 14.34
            # in Peskin 1996.
            
            Dv1 = 0
            Dv2 = 0
            
            if (v[i,j] < 0):
                Dv1 = (v[i_plus_1,j]-v[i,j])/dx
                Dv2 = (v[i, j_plus_1]-v[i,j])/dx
            else:
                Dv1 = (v[i,j] - v[i_minus_1,j])/dx
                Dv2 = (v[i,j]-v[i,j_minus_1])/dx
            
            w2[i,j] = v[i, j] - (dt/rho)*(u[i,j]*Dv1 + v[i,j]*Dv2)+(dt/rho)*f2[i,j]
            j+=1
            
        i+=1
    
    return w2

"""
Description: Calculates u and v for a full step of the NS Equations
Arguments:
    u: NxN numpy array
    v: NxN numpy array
    f1: NxN numpy array
    f2: NxN numpy array
    rho: 1x1 float (constant density)
    mu: 1x1 float (constant kinematic viscosity)
    dx: 1x1 float
    dt: 1x1 float
Returns:
    u: NxN numpy array
    v: NxN numpy array
"""
def calculate_step(u, v, f1, f2, rho, mu, dx, dt):
    M, N = u.shape
    if (M != N):
        print("Grid must be square.")
        exit()
        
    A = 1j*np.zeros((N,N))
    k1 = 0
    while (k1 < N):
        k2 = 0
        while (k2 < N):
            A[k1,k2] = 1 + 4*mu*dt/(rho*dx**2)*(np.sin(np.pi*k1/N)**2+np.sin(np.pi*k2/N)**2)
            k2+=1
        k1+=1
    
    #Calculate w1 and w2

    w1 = calculate_w1(u, v, f1, f2, rho, dx, dt)
    w2 = calculate_w2(u, v, f1, f2, rho, dx, dt)
    
    #Calculate Fourier transformed w1 and w2
    w_hat_1 = fft.fft2(w1)
    w_hat_2 = fft.fft2(w2)
    
    #Calculate transformed pressure
    p_hat = 1j*np.zeros((N,N))
    
    k1 = 0
    while (k1 < N):
        k2 = 0
        while (k2 < N):
            #Handles special values
            if (k1 == 0 or k2 == 0 or k1 == N // 2 or k2 == N // 2):
                p_hat[k1, k2] = 0
            #Handles other values
            else:
                top = ((1j/dx)*(np.sin(2*np.pi/N * k1))*w_hat_1[k1,k2] + (1j/dx)*(np.sin(2*np.pi/N * k2))*w_hat_2[k1,k2])
                bottom = (-dt/(rho*dx**2) * ((np.sin(2*np.pi/N*k1))**2 + (np.sin(2*np.pi/N*k2))**2)) 
                
                if (np.abs(bottom) < 0.000001): 
                    print("Warning: Division by zero")
                    exit()
                    
                p_hat[k1,k2] = top / bottom
            k2+=1
        k1+=1
    
    #Calculate transformed velocity in next timestep
    u_hat = 1j*np.zeros((N, N))
    v_hat = 1j*np.zeros((N, N))
    
    k1 = 0
    while (k1 < N):
        k2 = 0
        while (k2 < N):
            #Handles special values
            if (k1 == 0 or k2 == 0 or k1 == N // 2 or k2 == N // 2):
                u_hat[k1,k2] = w_hat_1[k1,k2]/A[k1,k2]
                v_hat[k1,k2] = w_hat_2[k1,k2]/A[k1,k2]
            #Handles other values
            else:
                u_hat[k1,k2] = (w_hat_1[k1,k2] - 1j*dt/(rho*dx)*np.sin(2*np.pi*k1/N)*p_hat[k1,k2]) / A[k1,k2]
                v_hat[k1,k2] = (w_hat_2[k1,k2] - 1j*dt/(rho*dx)*np.sin(2*np.pi*k2/N)*p_hat[k1,k2]) / A[k1,k2]
            k2+=1
        k1+=1
    
    
    #Transform velocity back
    u = np.real(fft.ifft2(u_hat))
    v = np.real(fft.ifft2(v_hat))
    
    return u, v

"""
Description: Initializes and solves the Navier stokes solver with a constant 
force density.
Arguments:
    L: 1x1 integer
        - Length of the fluid domain. The solver will generate a grid of length 
        L x L and discretize it.
    dx: 1x1 double
        - the grid size
    Niter: 1x1 integer
        - the number of iterations
"""
def solve_ns_const_force_1(L, dx, Niter):
    N = int(L/dx)
    
    u = np.zeros((N, N, Niter))
    v = np.ones((N, N, Niter))
    f1 = np.ones((N, N))
    f2 = np.zeros((N, N))
    
    rho = 1
    mu = 1
    dx = dx
    dt = 0.2
    
    ii = 0
    while (ii < Niter-1):
        u[:,:,ii+1], v[:,:,ii+1] = calculate_step(u[:,:,ii], v[:,:,ii], f1, f2, rho, mu, dx, dt)
        ii+=1

    args = (rho, mu, dx, dt, Niter)
    arg_list = ("rho: ", "mu: ", "dx: ", "dt: ", "Niter: ")
    params = []
    
    for i in range(len(args)):
        params.append(str(arg_list[i]) + str(args[i]))

    anim_u = plot_animated_heatmap(u, "u", params)
    anim_v = plot_animated_heatmap(v, "v", params)
    
    return anim_u, anim_v

"""
Description: Initializes the force density vector for the explosion problem.
Arguments:
    f1: NxN numpy array
        - The x component of force density
    f2: NxN numpy array
        - The y component of force density
    icenter: 1x1 integer (nonnegative)
        - The x coordinate of the center gridpoint of the explosion
    jcenter: 1x1 integer (nonnegative)
        - The y coordinate of the center gridpoint of the explosion
    magx: 1x1 double (positive values only)
        - The magnitude of the x-component of the force
    magy: 1x1 double (positive values only)
        - The magnitude of the y-component of the force
    radius: 1x1 double (positive values only)
        - The radius of the explosion
    tstart: 1x1 integer (nonnegative)
        - The starting timestep
    tend: 1x1 integer (nonnegative)
        - The ending timestep
    explosion_type: 0 or 1
        - 0 is for a 3DD Gaussian
        - 1 is for a regularized Dirac delta
Warnings:
    * Be careful not to let the explosion radius interact with the boundaries.
    This solver uses periodic boundary conditions, so if any of your waves
    reach the boundary, they can cause self-interference and your velocities
    will increase rapidly until you get overflow errors.
    Your results will literally explode and not in a good way! Either 
    keep the radius small or make the grid size very large to counteract this.
    
    ** If the impulse (force * (tstart - tend)*dt) is too big, it will create
    a wave that travels all the way to the boundary. Once again, this causes
    self-interference and will mess with your results.
    
"""
def init_explosion(f1, f2, icenter, jcenter, magx, magy, radius, tstart, tend, explosion_type):
    f1[icenter,jcenter,tstart:tend] = 0
    f2[icenter,jcenter,tstart:tend] = 0
    M, N, T = f1.shape;
    
    def gauss_3d(i, j, t):
        return magx*np.exp(-1*( (i-icenter)**2 + (j-jcenter)**2 + (t-tstart)**2)), magy*np.exp(-1*( (i-icenter)**2 + (j-jcenter)**2 + (t-tstart)**2))
    
    i = 0
    while (i < N):
        j = 0
        while (j < N):
            k = 0
            while (k < T):
                if ((i-icenter)**2 + (j-jcenter)**2 <= radius**2 and k >= tstart and k <= tend):  
                    if (explosion_type == 0):
                        f1[i, j, k], f2[i, j, k] = gauss_3d(i, j, k)
                    elif (explosion_type == 1):
                        #Implement this (TODO)
                        pass
                    else:
                        print("Invalid explosion type")
                        exit()
                k+=1
            j+=1
        i+=1
    return f1, f2

"""
Description: Initializes and solves the Navier stokes solver with an explosion.
Arguments:
    rho: 1x1 double
        - Density of the fluid
    mu: 1x1 double
        - Viscosity (Newtonian fluid)
    L: 1x1 integer
        - Length of the fluid domain. The solver will generate a grid of length 
        L x L and discretize it.
    dx: 1x1 double
        - the grid size
    dt: 1x1 ddouble
        - the timestep size
    Niter: 1x1 integer
        - the number of iterations
    icenter: 1x1 integer (nonnegative)
        - The x coordinate of the center gridpoint of the explosion
    jcenter: 1x1 integer (nonnegative)
        - The y coordinate of the center gridpoint of the explosion
    magx: 1x1 double (positive values only)
        - The magnitude of the x-component of the force
    magy: 1x1 double (positive values only)
        - The magnitude of the y-component of the force
    radius: 1x1 double (positive values only)
        - The radius of the explosion
    tstart: 1x1 integer (nonnegative)
        - The starting timestep
    tend: 1x1 integer (nonnegative)
        - The ending timestep
    explosion_type: 0 or 1
        - 0 is for a 3D Gaussian
        - 1 is for a regularized Dirac delta
"""
def solve_ns_explosion(rho, mu, L, dx, dt, Niter, icenter, jcenter, magx, magy, radius, tstart, tend, explosion_type): 
    N = int(L/dx)
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.zeros((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    
    #Initialize f1 and f2 for explosion
    f1, f2 = init_explosion(f1, f2, icenter, jcenter, magx, magy, radius, tstart, tend, 0)
    
    ii = 0
    while (ii < Niter-1):
        u[:,:,ii+1], v[:,:,ii+1] = calculate_step(u[:,:,ii], v[:,:,ii], f1[:,:,ii], f2[:,:,ii], rho, mu, dx, dt)
        ii+=1
    
    args = (rho, mu, dx, dt, Niter, magx, magy, radius)
    arg_list = ("rho: ", "mu: ", "dx: ", "dt: ", "Niter: ", "magx: ", "magy: ", "radius: ")
    params = []
    
    for i in range(len(args)):
        params.append(str(arg_list[i]) + str(args[i]))
    
    #Display results with animated heat map
    anim_u = plot_animated_heatmap(u, "u", params);
    anim_v = plot_animated_heatmap(v, "v", params);

    return anim_u, anim_v

"""
Description: A plotting function
"""
def plot_animated_heatmap(u, variable, params):
    _, _, Niter = u.shape
    def init_heatmap(i):
        ax.cla()
        sns.heatmap(u[:,:,i],
                    ax = ax,
                    cbar = True,
                    cbar_ax = cbar_ax,
                    vmin = u.min(),
                    vmax = u.max())
    
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (10, 8))
    fig.suptitle("Velocity " + variable + " Parameter List: " + str(params))
    anim = animation.FuncAnimation(fig = fig, func = init_heatmap, frames = Niter, interval = 50, blit = False)
    return anim

def main():
    # Run
    N = int(50/0.3)
    t0 = time.perf_counter()
    anim_u, anim_v = solve_ns_explosion(1, 100, 50, 0.3, 0.02, 20, N//2, N//2, 1000, 1000, 20, 2, 6, 0)
    tf = time.perf_counter()
    print("time to run: " + str(round(100*(tf-t0))/100.0) + " (s)")
    #
    
    writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
    anim_u.save('u_plot.gif', writer=writer)
    anim_v.save('v_plot.gif', writer=writer)
    plt.show()
    plt.close()

main();



