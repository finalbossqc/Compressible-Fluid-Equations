#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from argparse import RawTextHelpFormatter
import plot_ns
import time

def calculate_w(u, v, w, f1, f2, f3, rho, mu, dx, dt, half):
    _, _, N = u.shape
        
    w1 = 1j*np.zeros((N,N,N))
    w2 = 1j*np.zeros((N,N,N))
    w3 = 1j*np.zeros((N,N,N))
    
    i = 0
    while (i < N):
        j = 0
        while (j < N):
            k = 0
            while (k < N):
                i_plus_1 = i+1
                i_minus_1 = i-1
                j_plus_1 = j+1
                j_minus_1 = j-1
                k_plus_1 = k+1
                k_minus_1 = k-1
                
                #Implement Periodic Boundary Conditions
                if (i == N-1):
                    i_plus_1 = 0
                if (i == 0):
                    i_minus_1 = N-1
                if (j == N-1):
                    j_plus_1 = 0
                if (j == 0):
                    j_minus_1 = N-1
                if (k == N-1):
                    k_plus_1 = 0
                if (k == 0):
                    k_minus_1 = N-1
                
                #Find upwind finite differences
                
                Du1 = 0
                Du2 = 0
                Du3 = 0
                Dv1 = 0
                Dv2 = 0
                Dv3 = 0
                Dw1 = 0
                Dw2 = 0
                Dw3 = 0
                
                if (u[i, j, k] < 0):
                    Du1 = (u[i_plus_1, j, k] - u[i,j,k])/dx
                    Dv1 = (v[i_plus_1,j,k]- v[i,j,k])/dx
                    Dw1 = (w[i_plus_1,j,k] - w[i,j,k])/dx
                else:
                    Du1 = (u[i,j,k] - u[i_minus_1,j,k])/dx
                    Dv1 = (v[i,j,k] - v[i_minus_1,j,k])/dx
                    Dw1 = (w[i,j,k] - w[i_minus_1,j,k])/dx
                if (v[i, j, k] < 0):
                    Du2 = (u[i, j_plus_1, k] - u[i,j,k])/dx
                    Dv2 = (v[i, j_plus_1, k] - v[i,j,k])/dx
                    Dw2 = (w[i, j_plus_1, k] - w[i,j,k])/dx
                else:
                    Du2 = (u[i,j,k] - u[i,j_minus_1,k])/dx
                    Dv2 = (v[i,j,k] - v[i,j_minus_1,k])/dx
                    Dw2 = (w[i,j,k] - w[i,j_minus_1,k])/dx
                if (w[i, j, k] < 0):
                    Du3 = (u[i, j, k_plus_1] - u[i, j, k])
                    Dv3 = (v[i, j, k_plus_1] - v[i, j, k])
                    Dw3 = (w[i, j, k_plus_1] - w[i, j, k])
                else:
                    Du3 = (u[i, j, k] - u[i, j, k_minus_1])
                    Dv3 = (v[i, j, k] - v[i, j, k_minus_1])
                    Dw3 = (w[i, j, k] - w[i, j, k_minus_1])
                
                S1 = u[i,j,k]*Du1 + v[i,j,k]*Du2 + w[i,j,k]*Du3
                S2 = u[i,j,k]*Dv1 + v[i,j,k]*Dv2 + w[i,j,k]*Dv3
                S3 = u[i,j,k]*Dw1 + v[i,j,k]*Dw2 + w[i,j,k]*Dw3
                
                w1[i,j,k] = u[i,j,k]
                w2[i,j,k] = v[i,j,k]
                w3[i,j,k] = w[i,j,k]
                
                #Calculate w1 and w2
                if (half == 0):
                    w1[i,j,k] -= (dt/2)*S1 - (dt/(2*rho))*f1[i,j,k]
                    w2[i,j,k] -= (dt/2)*S2 - (dt/(2*rho))*f2[i,j,k]
                    w3[i,j,k] -= (dt/2)*S3 - (dt/(2*rho))*f3[i,j,k]
                elif (half == 1):
                    Lu = (1/(dx**2))*(u[i_plus_1,j,k] + u[i_minus_1,j,k] + u[i,j_plus_1,k] + u[i,j_minus_1,k] + u[i,j,k_plus_1] + u[i,j,k_minus_1] - 6*u[i,j,k])
                    Lv = (1/(dx**2))*(v[i_plus_1,j,k] + v[i_minus_1,j,k] + v[i,j_plus_1,k] + v[i,j_minus_1,k] + v[i,j,k_plus_1] + v[i,j,k_minus_1] - 6*v[i,j,k])
                    Lw = (1/(dx**2))*(w[i_plus_1,j,k] + w[i_minus_1,j,k] + w[i,j_plus_1,k] + w[i,j_minus_1,k] + w[i,j,k_plus_1] + w[i,j,k_minus_1] - 6*w[i,j,k])
                    
                    w1[i,j,k] -= dt*S1 - (dt/rho)*f1[i,j,k] - (dt*mu/(2*rho))*Lu
                    w2[i,j,k] -= dt*S2 - (dt/rho)*f2[i,j,k] - (dt*mu/(2*rho))*Lv
                    w3[i,j,k] -= dt*S3 - (dt/rho)*f3[i,j,k] - (dt*mu/(2*rho))*Lw
                k+=1
            j+=1
        i+=1
    
    return w1, w2, w3

def calculate_step(u, v, w, f1, f2, f3, rho, mu, dx, dt):
    _, _, N = u.shape
    
    #__________________________________________________________________________
    #Half step:
    #Calculate w1 and w2
    w1, w2, w3 = calculate_w(u, v, w, f1, f2, f3, rho, mu, dx, dt, 0)
    
    #Take Fourier transforms
    w_hat_1 = fft.fft2(w1)
    w_hat_2 = fft.fft2(w2)
    w_hat_3 = fft.fft2(w3)
    
    #Initialize velocity and pressure
    u_hat = 1j*np.zeros((N,N,N))
    v_hat = 1j*np.zeros((N,N,N))
    w_hat = 1j*np.zeros((N,N,N))
    
    #Initilize matrices and transformed operators
    D_hat_1 = 1j*np.zeros((N,N,N))
    D_hat_2 = 1j*np.zeros((N,N,N))
    D_hat_3 = 1j*np.zeros((N,N,N))
    L_hat = 1j*np.zeros((N,N,N))
    
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            m3 = 0
            while (m3 < N):
                D_hat_1[m1,m2,m3] = (1j/dx)*np.sin(2*np.pi*m1*dx/N)
                D_hat_2[m1,m2,m3] = (1j/dx)*np.sin(2*np.pi*m2*dx/N)
                D_hat_3[m1,m2,m3] = (1j/dx)*np.sin(2*np.pi*m3*dx/N)
                L_hat[m1,m2,m3] = (-4/(dx**2))*(np.sin(np.pi*m1*dx/N)**2 + np.sin(np.pi*m2*dx/N)**2 + np.sin(np.pi*m3*dx/N)**2)
                m3 += 1
            m2 += 1
        m1+=1
    
    #Calculate transformed velocity
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            m3 = 0
            while (m3 < N):
                a11 = 0*1j
                a12 = 0*1j
                a13 = 0*1j
                a21 = 0*1j
                a22 = 0*1j
                a23 = 0*1j
                a31 = 0*1j
                a32 = 0*1j
                a33 = 0*1j
                
                if (np.abs(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2) < 0.00000001):
                    a11 = 1
                    a12 = 0
                    a13 = 0
                    a21 = 0
                    a22 = 1
                    a23 = 0
                    a31 = 0
                    a32 = 0
                    a33 = 1
                else:
                    a11 = 1 - D_hat_1[m1,m2,m3]**2/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a12 = -D_hat_1[m1,m2,m3]*D_hat_2[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a13 = -D_hat_1[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a21 = -D_hat_1[m1,m2,m3]*D_hat_2[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a22 = 1 - D_hat_2[m1,m2,m3]**2/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a23 = -D_hat_2[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a31 = -D_hat_1[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a32 = -D_hat_2[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a33 = 1 - D_hat_3[m1,m2,m3]**2/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                
                u_hat[m1,m2,m3] = (w_hat_1[m1,m2,m3]*a11 + w_hat_2[m1,m2,m3]*a12 + w_hat_3[m1,m2,m3]*a13) / (1 - mu*dt/(2*rho)*L_hat[m1,m2,m3])
                v_hat[m1,m2,m3] = (w_hat_1[m1,m2,m3]*a21 + w_hat_2[m1,m2,m3]*a22 + w_hat_3[m1,m2,m3]*a23) / (1 - mu*dt/(2*rho)*L_hat[m1,m2,m3])
                w_hat[m1,m2,m3] = (w_hat_1[m1,m2,m3]*a31 + w_hat_2[m1,m2,m3]*a32 + w_hat_3[m1,m2,m3]*a33) / (1 - mu*dt/(2*rho)*L_hat[m1,m2,m3])
                m3+=1
            m2+=1
        m1+=1
    
    #Transform velocity back
    
    u = np.real(fft.ifft2(u_hat))
    v = np.real(fft.ifft2(v_hat))
    w = np.real(fft.ifft2(w_hat))
    
    #__________________________________________________________________________
    #Full step
    #Calculate w1 and w2
    w1, w2, w3 = calculate_w(u, v, w, f1, f2, f3, rho, mu, dx, dt, 1)
    
    #Take Fourier transforms
    w_hat_1 = fft.fft2(w1)
    w_hat_2 = fft.fft2(w2)
    w_hat_3 = fft.fft2(w3)
    
    #Initialize velocity and pressure
    u_hat = 1j*np.zeros((N,N,N))
    v_hat = 1j*np.zeros((N,N,N))
    w_hat = 1j*np.zeros((N,N,N))
    
    #Calculate transformed velocity
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            m3 = 0
            while (m3 < N):
                a11 = 0*1j
                a12 = 0*1j
                a13 = 0*1j
                a21 = 0*1j
                a22 = 0*1j
                a23 = 0*1j
                a31 = 0*1j
                a32 = 0*1j
                a33 = 0*1j
                
                if (np.abs(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2) < 0.00000001):
                    a11 = 1
                    a12 = 0
                    a13 = 0
                    a21 = 0
                    a22 = 1
                    a23 = 0
                    a31 = 0
                    a32 = 0
                    a33 = 1
                else:
                    a11 = 1 - D_hat_1[m1,m2,m3]**2/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a12 = -D_hat_1[m1,m2,m3]*D_hat_2[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a13 = -D_hat_1[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a21 = -D_hat_1[m1,m2,m3]*D_hat_2[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a22 = 1 - D_hat_2[m1,m2,m3]**2/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a23 = -D_hat_2[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a31 = -D_hat_1[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a32 = -D_hat_2[m1,m2,m3]*D_hat_3[m1,m2,m3]/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                    a33 = 1 - D_hat_3[m1,m2,m3]**2/(D_hat_1[m1,m2,m3]**2 + D_hat_2[m1,m2,m3]**2 + D_hat_3[m1,m2,m3]**2)
                
                u_hat[m1,m2,m3] = (w_hat_1[m1,m2,m3]*a11 + w_hat_2[m1,m2,m3]*a12 + w_hat_3[m1,m2,m3]*a13) / (1 - mu*dt/(2*rho)*L_hat[m1,m2,m3])
                v_hat[m1,m2,m3] = (w_hat_1[m1,m2,m3]*a21 + w_hat_2[m1,m2,m3]*a22 + w_hat_3[m1,m2,m3]*a23) / (1 - mu*dt/(2*rho)*L_hat[m1,m2,m3])
                w_hat[m1,m2,m3] = (w_hat_1[m1,m2,m3]*a31 + w_hat_2[m1,m2,m3]*a32 + w_hat_3[m1,m2,m3]*a33) / (1 - mu*dt/(2*rho)*L_hat[m1,m2,m3])
                m3+=1
            m2+=1
        m1+=1
    
    #Transform velocity back
    
    u = np.real(fft.ifft2(u_hat))
    v = np.real(fft.ifft2(v_hat))
    w = np.real(fft.ifft2(w_hat))
    
    return u, v, w


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
def solve_ns_const_force_1(rho, mu, L, dx, dt, Niter):
    N = int(L/dx)

    u = np.zeros((N, N, N, Niter))
    v = np.zeros((N, N, N, Niter))
    w = np.zeros((N, N, N, Niter))
    f1 = np.ones((N, N, N, Niter))
    f2 = np.zeros((N, N, N, Niter))
    f3 = np.zeros((N, N, N, Niter))
    
    ii = 0
    while (ii < Niter-1):
        print("iteration: ", ii)
        u[:, :, :, ii+1], v[:, :, :, ii +1], w[:, :,:, ii+1] = calculate_step(u[:, :, :, ii], v[:, :, :, ii], w[:,:,:,ii], f1[:, :, :, ii], f2[:, :, :, ii], f3[:, :, :, ii], rho, mu, dx, dt)
        ii += 1

    args = (rho, mu, dx, dt, Niter)
    arg_list = ("rho: ", "mu: ", "dx: ", "dt: ", "Niter: ")
    params = []

    for i in range(len(args)):
        params.append(str(arg_list[i]) + str(args[i]))

    #anim_u = plot_ns.animated_heatmap(u, "u", params)
    #anim_v = plot_ns.animated_heatmap(v, "v", params)

solve_ns_const_force_1(1, 1, 1, 0.1, 0.1, 100)



