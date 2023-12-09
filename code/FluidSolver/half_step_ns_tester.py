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


def calculate_w(u, v, f1, f2, rho, mu, dx, dt, half):
    M, N = u.shape

    if (M != N):
        print("Grid length and width must be the same")
        exit()

    w1 = 1j*np.zeros((N, N))
    w2 = 1j*np.zeros((N, N))

    i = 0
    while (i < N):
        j = 0
        while (j < N):
            i_plus_1 = i+1
            i_minus_1 = i-1
            j_plus_1 = j+1
            j_minus_1 = j-1

            # Implement Periodic Boundary Conditions
            if (i == N-1):
                i_plus_1 = 0
            if (i == 0):
                i_minus_1 = N-1
            if (j == N-1):
                j_plus_1 = 0
            if (j == 0):
                j_minus_1 = N-1

            # Calculate central finite differences
            Du1 = (u[i_plus_1, j]-u[i_minus_1, j])/(2*dx)
            Du2 = (u[i, j_plus_1] - u[i, j_minus_1])/(2*dx)
            Dv1 = (v[i_plus_1, j]-v[i_minus_1, j])/(2*dx)
            Dv2 = (v[i, j_plus_1]-v[i, j_minus_1])/(2*dx)

            Duu1 = (u[i_plus_1, j]**2 - u[i_minus_1, j]**2)/(2*dx)
            Duv2 = (u[i, j_plus_1]*v[i, j_plus_1] -
                    u[i, j_minus_1]*v[i, j_minus_1])/(2*dx)
            Duv1 = (u[i_plus_1, j]*v[i_plus_1, j] -
                    u[i_minus_1, j]*v[i_minus_1, j])/(2*dx)
            Dvv2 = (v[i, j_plus_1]**2 - v[i, j_minus_1]**2)/(2*dx)

            S1 = 1/2*(u[i, j]*Du1 + v[i, j]*Du2) + 1/2*(Duu1 + Duv2)
            S2 = 1/2*(u[i, j]*Dv1 + v[i, j]*Dv2) + 1/2*(Duv1 + Dvv2)

            #print(S1, S1)

            w1[i, j] = u[i, j]
            w2[i, j] = v[i, j]

            # Calculate w1 and w2
            if (half == 0):
                w1[i, j] -= (dt/2)*S1 - (dt/(2*rho))*f1[i, j]
                w2[i, j] -= (dt/2)*S2 - (dt/(2*rho))*f2[i, j]
            elif (half == 1):
                Lu = (1/(dx**2))*(u[i_plus_1, j] + u[i_minus_1, j] - 4*u[i, j] + u[i, j_plus_1] + u[i, j_minus_1])
                Lv = (1/(dx**2))*(v[i_plus_1, j] + v[i_minus_1, j] - 4*v[i, j] + v[i, j_plus_1] + v[i, j_minus_1])

                w1[i, j] -= dt*S1 - (dt/rho)*f1[i, j] - (dt*mu/(2*rho))*Lu
                w2[i, j] -= dt*S2 - (dt/rho)*f2[i, j] - (dt*mu/(2*rho))*Lv
            j += 1
        i += 1

    return w1, w2


def calculate_step(u, v, f1, f2, rho, mu, dx, dt):
    M, N = u.shape
    if (M != N):
        print("Grid must be square.")

    # __________________________________________________________________________
    # Half step:
    # Calculate w1 and w2
    w1, w2 = calculate_w(u, v, f1, f2, rho, mu, dx, dt, 0)

    # Take Fourier transforms
    w_hat_1 = fft.fft2(w1)
    w_hat_2 = fft.fft2(w2)

    # Initialize velocity and pressure
    u_hat = 1j*np.zeros((N, N))
    v_hat = 1j*np.zeros((N, N))

    # Initilize matrices and transformed operators
    D_hat_1 = 1j*np.zeros((N, N))
    D_hat_2 = 1j*np.zeros((N, N))
    L_hat = 1j*np.zeros((N, N))

    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            D_hat_1[m1, m2] = (1j/dx)*np.sin(2*np.pi*m1*dx/N)
            D_hat_2[m1, m2] = (1j/dx)*np.sin(2*np.pi*m2*dx/N)
            L_hat[m1, m2] = (-4/(dx**2))*(np.sin(np.pi*m1*dx/N)
                                          ** 2 + np.sin(np.pi*m2*dx/N)**2)
            m2 += 1
        m1 += 1

    # Calculate transformed velocity
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            a11 = 0*1j
            a12 = 0*1j
            a21 = 0*1j
            a22 = 0*1j

            if (np.abs(D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2) < 0.0001):
                a11 = 1
                a12 = 0
                a21 = 0
                a22 = 1
            else:
                a11 = (1 - D_hat_1[m1, m2]**2 / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2))
                a12 = -D_hat_1[m1, m2]*D_hat_2[m1, m2] / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2)
                a21 = -D_hat_1[m1, m2]*D_hat_2[m1, m2] / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2)
                a22 = (1 - D_hat_2[m1, m2]**2 / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2))

                if (np.isnan(a11)):
                    print(m1, m2)
                    exit()

            u_hat[m1, m2] = (w_hat_1[m1, m2]*a11 + w_hat_2[m1, m2] * a12) / (1 - mu*dt/(2*rho)*L_hat[m1, m2])
            v_hat[m1, m2] = (w_hat_1[m1, m2]*a21 + w_hat_2[m1, m2] * a22) / (1 - mu*dt/(2*rho)*L_hat[m1, m2])

            m2 += 1
        m1 += 1

    # Transform velocity back

    u = np.real(fft.ifft2(u_hat))
    v = np.real(fft.ifft2(v_hat))

    # __________________________________________________________________________
    # Full step
    # Calculate w1 and w2
    w1, w2 = calculate_w(u, v, f1, f2, rho, mu, dx, dt, 1)

    # Take Fourier transforms
    w_hat_1 = fft.fft2(w1)
    w_hat_2 = fft.fft2(w2)

    # Initialize velocity and pressure
    u_hat = 1j*np.zeros((N, N))
    v_hat = 1j*np.zeros((N, N))

    # Calculate transformed velocity
    m1 = 0
    while (m1 < N):
        m2 = 0
        while (m2 < N):
            a11 = 0*1j
            a12 = 0*1j
            a21 = 0*1j
            a22 = 0*1j

            if (np.abs(D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2) < 0.0001):
                a11 = 1
                a12 = 0
                a21 = 0
                a22 = 1
            else:
                a11 = (1 - D_hat_1[m1, m2]**2 / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2))
                a12 = -D_hat_1[m1, m2]*D_hat_2[m1, m2] / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2)
                a21 = -D_hat_1[m1, m2]*D_hat_2[m1, m2] / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2)
                a22 = (1 - D_hat_2[m1, m2]**2 / (D_hat_1[m1, m2]**2 + D_hat_2[m1, m2]**2))

            u_hat[m1, m2] = (w_hat_1[m1, m2]*a11 + w_hat_2[m1, m2]
                             * a12) / (1 - mu*dt/(2*rho)*L_hat[m1, m2])
            v_hat[m1, m2] = (w_hat_1[m1, m2]*a21 + w_hat_2[m1, m2]
                             * a22) / (1 - mu*dt/(2*rho)*L_hat[m1, m2])

            m2 += 1
        m1 += 1

    # Transform velocity back

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
def solve_ns_const_force_1(rho, mu, L, dx, dt, Niter):
    N = int(L/dx)

    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.ones((N, N))
    f2 = np.zeros((N, N))
    
    err = np.zeros(Niter)

    ii = 0
    while (ii < Niter-1):
        print("iteration: ", ii)
        u[:, :, ii+1], v[:, :, ii +
                         1] = calculate_step(u[:, :, ii], v[:, :, ii], f1, f2, rho, mu, dx, dt)
        err[ii] = np.abs(np.average(v[:, :, ii+1]-np.zeros((N, N))))
        print(err[ii])
        ii += 1

    x = np.arange(0, Niter)
    y = np.zeros(len(err))

    for i in range(len(err)):
        if (err[i] == 0):
            y[i] = 0
        else:
            y[i] = np.log10(err[i])

    plt.plot(x, y)
    plt.show()
    np.savetxt("half_step_error.txt", y[10:630])

    args = (rho, mu, dx, dt, Niter)
    arg_list = ("rho: ", "mu: ", "dx: ", "dt: ", "Niter: ")
    params = []

    for i in range(len(args)):
        params.append(str(arg_list[i]) + str(args[i]))

    anim_u = plot_ns.animated_heatmap(u, "u", params)
    anim_v = plot_ns.animated_heatmap(v, "v", params)

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


def init_explosion(f1, f2, dx, icenter, jcenter, magx, magy, radius, tstart, tend, explosion_type):
    f1[icenter, jcenter, tstart:tend] = 0
    f2[icenter, jcenter, tstart:tend] = 0
    tcenter = (tstart + tend)/2
    
    M, N, T = f1.shape

    def gauss_3d(i, j, t):
        return magx*np.exp(-1*((i-icenter)**2 + (j-jcenter)**2 + (t-tstart)**2)), magy*np.exp(-1*((i-icenter)**2 + (j-jcenter)**2 + (t-tstart)**2))
    
    def delta_3d(i, j, t):
        def phi(r):
            if (r <= 2):
                return 0.25*(1+np.cos(np.pi*r/2))
            else:
                return 0
        
        
        return magx*(1/((4*dx)**3))*phi(i-icenter)*phi(j-jcenter)*phi(t-tcenter), magy*(1/((4*dx)**3))*phi(i-icenter)*phi(j-jcenter)*phi(t-tcenter)

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
                        f1[i, j, k], f2[i, j, k] = delta_3d(i, j, k)
                    else:
                        print("Invalid explosion type")
                        exit()
                k += 1
            j += 1
        i += 1
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

    # Initialize f1 and f2 for explosion

    f1, f2 = init_explosion(f1, f2, dx, icenter, jcenter,
                            magx, magy, radius, tstart, tend, 0)

    ii = 0
    while (ii < Niter-1):
        print(ii)
        u[:, :, ii+1], v[:, :, ii+1] = calculate_step(
            u[:, :, ii], v[:, :, ii], f1[:, :, ii], f2[:, :, ii], rho, mu, dx, dt)
        ii += 1

    args = (rho, mu, dx, dt, Niter, magx, magy, radius)
    arg_list = ("rho: ", "mu: ", "dx: ", "dt: ",
                "Niter: ", "magx: ", "magy: ", "radius: ")
    params = []

    for i in range(len(args)):
        params.append(str(arg_list[i]) + str(args[i]))

    # Display results with animated heat map
    anim_u = plot_ns.animated_heatmap(u, "u", params)
    anim_v = plot_ns.animated_heatmap(v, "v", params)

    return anim_u, anim_v


def main():
    rho = 
    mu = 
    dx = 
    L = 
    Niter = 
    magx = 
    magy = 
    radius = 8
    tstart = 
    tend = 
    N = int(L/dx)
    
    t0 = time.perf_counter()
    anim_u, anim_v = solve_ns_explosion(rho, mu, L, dx, dt, Niter, N//2, N//2, magx, magy, radius, tstart, tend, 0)
        
    tf = time.perf_counter()
    print("time to run: " + str(round(100*(tf-t0))/100.0) + " (s)")
    
    writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
    anim_u.save('u_plot.gif', writer=writer)
    anim_v.save('v_plot.gif', writer=writer)
    plt.show()
    plt.close()

main()
