#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import full_step_ns as fs
import half_step_ns as hs
import half_step_upwind_ns as hsu

import numpy as np
import matplotlib.pyplot as plt

def test1():
    L = 50
    dx = 2
    N = int(L/dx)
    Niter = 1000
    
    rho = 1
    mu = 100
    
    dt = 0.01
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.ones((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    
    pi = int(N*np.random.random())
    pj = int(N*np.random.random())
    fig1 = probe_point(fs, "Full Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("full_step_probe.png")
    fig2 = probe_point(hs, "Half Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_probe.png")
    fig3 = probe_point(hsu, "Half Step Upwind", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_upwind_probe.png")

def test2():
    L = 50
    dx = 2
    N = int(L/dx)
    Niter = 1000
    
    rho = 1
    mu = 100
    
    dt = 0.01
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.zeros((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    f1, f2 = fs.init_explosion(f1, f2, dx, N//2, N//2, 100, 100, 8, 2, 10, 0)
    
    pi = int(N*np.random.random())
    pj = int(N*np.random.random())
    fig1 = probe_point(fs, "Full Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("full_step_probe.png")
    fig2 = probe_point(hs, "Half Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_probe.png")
    fig3 = probe_point(hsu, "Half Step Upwind", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_upwind_probe.png")

def test3():
    L = 50
    dx = 2
    N = int(L/dx)
    Niter = 1000
    
    rho = 1
    mu = 100
    
    dt = 0.01
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.zeros((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    f1, f2 = fs.init_explosion(f1, f2, dx, N//2, N//2, 100, 100, 8, 2, 10, 0)
    
    points = []
    for _ in range(2):
        pi = int(N*np.random.random())
        pj = int(N*np.random.random())
        points.append((pi, pj))
    
    fig1 = probe_point(fs, "Full Step", u, v, f1, f2, rho, mu, dx, dt, Niter, points)
    plt.savefig("full_step_multiple_probe.png")
    fig2 = probe_point(hs, "Half Step", u, v, f1, f2, rho, mu, dx, dt, Niter, points)
    plt.savefig("half_step_multiple_probe.png")
    fig3 = probe_point(hsu, "Half Step Upwind", u, v, f1, f2, rho, mu, dx, dt, Niter, points)
    plt.savefig("half_step_upwind_multiple_probe.png")
    
def probe_point(solver, solver_name, u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj):
    ii = 0
    while (ii < Niter-1):
        u[:,:,ii+1], v[:,:,ii+1] = solver.calculate_step(u[:, :, ii], v[:, :, ii], f1[:, :, ii], f2[:, :, ii], rho, mu, dx, dt)
        ii += 1
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(np.arange(0, Niter), u[pi, pj, :], label='u')
    ax[1].plot(np.arange(0, Niter), v[pi, pj, :], label='v')
    
    # Added legend
    ax[0].legend()
    ax[1].legend()
    
    # Added axis labels and title
    ax[0].set_xlabel('Timesteps')
    ax[0].set_ylabel('Velocity (u)')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity (v)')
    fig.suptitle("Velocity at a Randomly Sampled Grid Point with " + solver_name + " Solver")
    
    # Adjusted layout
    plt.tight_layout()
    
    return fig

def probe_points(solver, solver_name, u, v, f1, f2, rho, mu, dx, dt, Niter, points):
    ii = 0
    while (ii < Niter-1):
        u[:,:,ii+1], v[:,:,ii+1] = solver.calculate_step(u[:, :, ii], v[:, :, ii], f1[:, :, ii], f2[:, :, ii], rho, mu, dx, dt)
        ii += 1
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    for j in range(len(points)):
        pi, pj = points[j]
        
        ax[0].plot(np.arange(0, Niter), u[pi, pj, :], label='u')
        ax[1].plot(np.arange(0, Niter), v[pi, pj, :], label='v')
    
    ax[0].legend()
    ax[1].legend()
    
    ax[0].set_xlabel('Timesteps')
    ax[0].set_ylabel('Velocity (u)')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity (v)')
    fig.suptitle("Velocity at a Randomly Sampled Grid Point with " + solver_name + " Solver")
    
    plt.tight_layout()

test1()
