#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.fftpack as fft

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
    w1 = np.zeros((M,N))
    
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
    w2 = np.zeros((M,N))
    
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
def calculate_full_step(u, v, f1, f2, rho, mu, dx, dt):
    M, N = u.shape
    if (M != N):
        print("Grid must be square.")
        exit()
        
    A = np.zeros((N,N))
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
    p_hat = np.zeros((N,N))
    
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
                
                if (bottom == 0): 
                    print("Warning: Division by zero")
                    
                p_hat[k1,k2] = top / bottom
            k2+=1
        k1+=1
    
    #print(p_hat)
    
    #Calculate transformed velocity in next timestep
    u_hat = np.zeros((N, N))
    v_hat = np.zeros((N, N))
    
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


def fft1D(x):
    N = len(x)
    
    if N <= 1:
        return x
    
    even = fft1D(x[0::2])
    odd = fft1D(x[1::2])
    
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    
    return np.concatenate([even + T, even - T])

def fft2D(x):
    rows, cols = x.shape
    
    # FFT along the rows
    for i in range(rows):
        x[i, :] = fft1D(x[i, :])
    
    # FFT along the columns
    for j in range(cols):
        x[:, j] = fft1D(x[:, j])
    
    return x

def ifft1D(x):
    N = len(x)
    
    if N <= 1:
        return x
    
    even = ifft1D(x[0::2])
    odd = ifft1D(x[1::2])
    
    T = [np.exp(2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    
    return np.concatenate([even + T, even - T])

def ifft2D(x):
    rows, cols = x.shape
    
    # IFFT along the rows
    for i in range(rows):
        x[i, :] = ifft1D(x[i, :])
    
    # IFFT along the columns
    for j in range(cols):
        x[:, j] = ifft1D(x[:, j])
    
    return x  # Normalize by the number of elements