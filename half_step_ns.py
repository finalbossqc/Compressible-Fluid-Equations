#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.fftpack as fft
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_w(u, v, f1, f2, rho, mu, dx, dt, half):
    M, N = u.shape
    
    if (M != N):
        print("Grid length and width must be the same")
        
    w1 = np.zeros((N,N))
    w2 = np.zeros((N,N))
    
    i = 0
    while (i < N):
        j = 0
        while (j < N):
            i_plus_1 = i+1
            i_minus_1 = i-1
            j_plus_1 = j+1
            j_minus_1 = j-1
            
            #Implement Periodic Boundary Conditions
            if (i == N-1):
                i_plus_1 = 0
            if (i == 0):
                i_minus_1 = N-1
            if (j == N-1):
                j_plus_1 = 0
            if (j == 0):
                j_minus_1 = N-1
            
            #Calculate finite differences
            
            Du1 = (u[i_plus_1,j]-u[i_minus_1,j])/(2*dx)
            Du2 = (u[i, j_plus_1] - u[i, j_minus_1])/(2*dx)
            Dv1 = (v[i_plus_1,j]-u[i_minus_1,j])/(2*dx)
            Dv2 = (v[i,j_plus_1]-v[i,j_minus_1])/(2*dx)
            
            Duu1 = (u[i_plus_1,j]**2 - u[i_minus_1,j]**2)/(2*dx)
            Duv2 = (u[i, j_plus_1]*v[i, j_plus_1] - u[i, j_minus_1]*v[i,j_minus_1])/(2*dx)
            Duv1 = (u[i_plus_1,j]*v[i_plus_1,j] - u[i_minus_1,j]*v[i_minus_1,j])/(2*dx)
            Dvv2 = (v[i,j_plus_1]**2 - v[i, j_minus_1]**2)/(2*dx)
            
            S1 = 1/2*(u[i,j]*Du1 + v[i,j]*Du2) + 1/2*(Duu1 + Duv2)
            S2 = 1/2*(u[i,j]*Dv1 + v[i,j]*Dv2) + 1/2*(Duv1 + Dvv2)
            
            w1[i,j] = u[i,j]
            w2[i,j] = v[i,j]
            #Calculate w1 and w2
            if (half == 0):
                w1[i,j] -= (dt/2)*S1 + (dt/(2*rho))*f1[i,j]
                w2[i,j] -= (dt/2)*S2 + (dt/(2*rho))*f2[i,j]
            elif (half == 1):
                Lu = (1/(dx**2))*(u[i_plus_1,j] + u[i_minus_1,j] + 4*u[i,j] + u[i,j_plus_1] + u[i,j_minus_1])
                Lv = (1/(dx**2))*(v[i_plus_1,j] + v[i_minus_1,j] + 4*v[i,j] + v[i,j_plus_1] + v[i,j_minus_1])
                
                w1[i,j] -= dt*S1 + (dt/rho)*f1[i,j] + (dt*mu/(2*rho))*Lu
                w2[i,j] -= dt*S2 + (dt/rho)*f2[i,j] + (dt*mu/(2*rho))*Lv
            j+=1
        i+=1
    
    return w1, w2

def calculate_step(u, v, f1, f2, rho, mu, dx, dt):
    M, N = u.shape
    if (M != N):
        print("Grid must be square.")
    
    #__________________________________________________________________________
    #Half step:
    #Calculate w1 and w2
    w1, w2 = calculate_w(u, v, f1, f2, rho, mu, dx, dt, 0)
    
    #Take Fourier transforms
    w_hat_1 = fft.fft2(w1)
    w_hat_2 = fft.fft2(w2)
    
    #Initialize velocity and pressure
    u_hat = 1j*np.zeros((N,N))
    v_hat = 1j*np.zeros((N,N))
    q_hat = 1j*np.zeros((N,N))
    
    #Initilize matrices and transformed operators
    I = np.eye(N)
    D_hat_1 = 1j*np.zeros((N,N))
    D_hat_2 = 1j*np.zeros((N,N))
    L_hat = 1j*np.zeros((N,N))
    
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            D_hat_1[m1,m2] = (1j/dx)*np.sin(2*np.pi*m1*dx/N)
            D_hat_2[m1,m2] = (1j/dx)*np.sin(2*np.pi*m2*dx/N)
            L_hat[m1,m2] = (-4/(dx**2))*(np.sin(np.pi*m1*dx/N) + np.sin(np.pi*m2*dx/N))
            m2 += 1
        m1+=1
    
    #Calculate pressure
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            q_hat[m1,m2] = (D_hat_1[m1,m2]*w_hat_1[m1,m2]+D_hat_2[m1,m2]*w_hat_2[m1,m2]) / ((dt/rho)*(D_hat_1[m1,m2]**2 + D_hat_2[m1,m2]**2))
            m2 += 1
        m1+=1
    
    #Calculate velocity
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            u_hat[m1,m2] = (w_hat_1[m1,m2] - (dt/rho) * D_hat_1[m1,m2] * q_hat[m1,m2]) / (I[m1,m2] - (dt*mu)/(2*rho)*L_hat[m1,m2])
            v_hat[m1,m2] = (w_hat_2[m1,m2] - (dt/rho) * D_hat_2[m1,m2] * q_hat[m1,m2]) / (I[m1,m2] - (dt*mu)/(2*rho)*L_hat[m1,m2])
            m2 += 1
        m1+=1
    
    #Transform velocity back
    
    u = fft.ifft2(u_hat)
    v = fft.ifft2(v_hat)
    
    #__________________________________________________________________________
    #Full step
    #Calculate w1 and w2
    w1, w2 = calculate_w(u, v, f1, f2, rho, mu, dx, dt, 1)
    
    #Take Fourier transforms
    w_hat_1 = fft.fft2(w1)
    w_hat_2 = fft.fft2(w2)
    
    #Initialize velocity and pressure
    u_hat = 1j*np.zeros((N,N))
    v_hat = 1j*np.zeros((N,N))
    q_hat = 1j*np.zeros((N,N))
    
    #Initilize matrices and transformed operators
    D_hat_1 = 1j*np.zeros((N,N))
    D_hat_2 = 1j*np.zeros((N,N))
    L_hat = 1j*np.zeros((N,N))
    
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            D_hat_1[m1,m2] = (1j/dx)*np.sin(2*np.pi*m1*dx/N)
            D_hat_2[m1,m2] = (1j/dx)*np.sin(2*np.pi*m2*dx/N)
            L_hat[m1,m2] = (-4/(dx**2))*(np.sin(np.pi*m1*dx/N) + np.sin(np.pi*m2*dx/N))
            m2 += 1
        m1+=1
    
    #Calculate pressure
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            q_hat[m1,m2] = (D_hat_1[m1,m2]*w_hat_1[m1,m2]+D_hat_2[m1,m2]*w_hat_2[m1,m2]) / ((dt/rho)*(D_hat_1[m1,m2]**2 + D_hat_2[m1,m2]**2))
            m2 += 1
        m1+=1
    
    #Calculate velocity
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            u_hat[m1,m2] = (w_hat_1[m1,m2] - (dt/rho) * D_hat_1[m1,m2] * q_hat[m1,m2]) / (I[m1,m2] - (dt*mu)/(2*rho)*L_hat[m1,m2])
            v_hat[m1,m2] = (w_hat_2[m1,m2] - (dt/rho) * D_hat_2[m1,m2] * q_hat[m1,m2]) / (I[m1,m2] - (dt*mu)/(2*rho)*L_hat[m1,m2])
            m2 += 1
        m1+=1
    
    #Transform velocity back
    
    u = fft.ifft2(u_hat)
    v = fft.ifft2(v_hat)
    
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

    #Refactor this code such that the plotting capability is in its own
    #Display results with animated heat map
    
    def init_heatmap(i):
        ax.cla()
        ax.set_title("u")
        sns.heatmap(u[:,:,i],
                    ax = ax,
                    cbar = True,
                    cbar_ax = cbar_ax,
                    vmin = u.min(),
                    vmax = u.max())
    
    #Create the heatmap for v (TODO)
    
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (8, 5))
    anim = FuncAnimation(fig = fig, func = init_heatmap, frames = Niter, interval = 50, blit = False)
    
    #Create the vector plot for u and v (TODO)

    plt.show()


solve_ns_const_force_1(50, 0.3, 20)