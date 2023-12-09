#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import full_step_ns as fs
import half_step_ns as hs
import half_step_upwind_ns as hsu

import numpy as np
import matplotlib.pyplot as plt

import plot_ns

def test1():
    L = 50
    dx = 0.01
    N = int(L/dx)
    Niter = 1000
    
    rho = 1
    mu = 10
    
    dt = 0.01
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.ones((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    
    pi = int(N*np.random.random())
    pj = int(N*np.random.random())
    fig1 = probe_point(fs, "Full Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("full_step_probe_1.png")
    fig2 = probe_point(hs, "Half Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_probe_1.png")
    fig3 = probe_point(hsu, "Half Step Upwind", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_upwind_probe_1.png")

def test2():
    L = 0.5
    dx = 0.01
    N = int(L/dx)
    Niter = 10
    
    rho = 1
    mu = 10
    
    dt = 0.01
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.zeros((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    f1, f2 = fs.init_explosion(f1, f2, dx, N//2, N//2, 100, 100, 8, 2, 4, 0)
    
    pi = int(N*np.random.random())
    pj = int(N*np.random.random())
    fig1 = probe_point(fs, "Full Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("full_step_probe_2.png")
    fig2 = probe_point(hs, "Half Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_probe_2.png")
    fig3 = probe_point(hsu, "Half Step Upwind", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_upwind_probe_2.png")

def test3():
    L = 0.5
    dx = 0.01
    N = int(L/dx)
    Niter = 1000
    
    rho = 1
    mu = 10
    
    dt = 0.01
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    f1 = np.zeros((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    f1, f2 = fs.init_explosion(f1, f2, dx, N//2, N//2, 100, 100, 8, 2, 4, 1)
    
    pi = int(N*np.random.random())
    pj = int(N*np.random.random())
    
    fig1 = probe_point(fs, "Full Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("full_step_multiple_probe_3.png")
    fig2 = probe_point(hs, "Half Step", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_multiple_probe_3.png")
    fig3 = probe_point(hsu, "Half Step Upwind", u, v, f1, f2, rho, mu, dx, dt, Niter, pi, pj)
    plt.savefig("half_step_upwind_multiple_probe_3.png")

def basic_test(solver, dx, dt, hm):
    L = 50
    N = int(L/dx)
    print("N", N)
    Niter = int(2/dt)
    
    rho = 1
    mu = 1
    
    dt = dt
    
    print("dt", dt)
    
    #Define grid variables
    X = np.zeros((N,N))
    Y = np.zeros((N,N))
    
    i = 0
    while (i < N):
        j = 0
        while (j < N):
            X[i,j] = i*dx
            Y[i,j] = j*dx
            j+=1
        i+=1
    
    u = np.zeros((N, N, Niter))
    v = np.zeros((N, N, Niter))
    
    u[:,:,0] = -np.cos(X)*np.sin(Y)
    v[:,:,0] = np.sin(X)*np.cos(Y)
    
    f1 = np.zeros((N, N, Niter))
    f2 = np.zeros((N, N, Niter))
    
    #Calculate analytic solution
    u_anal = np.zeros((N, N, Niter))
    v_anal = np.zeros((N, N, Niter))
    
    t = 0
    while (t < Niter):
        u_anal[:,:,t] = -np.cos(X)*np.sin(Y)*np.exp(-2*t*dt)
        v_anal[:,:,t] = np.sin(X)*np.cos(Y)*np.exp(-2*t*dt)
        t+=1
    
    u_s, v_s = gen_sol(solver, u, v, f1, f2, rho, mu, dt, dx, L, Niter)
    
    args = (rho, mu, dx, dt, Niter)
    arg_list = ("rho: ", "mu: ", "dx: ", "dt: ", "Niter: ")
    params = []

    for i in range(len(args)):
        params.append(str(arg_list[i]) + str(args[i]))
    if (hm):
        anim_u = plot_ns.animated_heatmap(u_s, "u", params)
        anim_v = plot_ns.animated_heatmap(v_s, "v", params)
        err_u = plot_ns.animated_heatmap(np.abs(u_s[2:N-2,2:N-2,:] - u_anal[2:N-2,2:N-2,0:Niter-3]), "u_err", params)
        err_v = plot_ns.animated_heatmap(np.abs(v_s[2:N-2,2:N-2,:] - v_anal[2:N-2,2:N-2,0:Niter-3]), "v_err", params)
        plt.show()
    
    return np.average(np.abs(u_s[2:N-2,2:N-2,:] - u_anal[2:N-2,2:N-2,0:Niter-3])), np.average(np.abs(v_s[2:N-2,2:N-2,:] - v_anal[2:N-2,2:N-2,0:Niter-3])) 

def gen_sol(solver, u, v, f1, f2, rho, mu, dt, dx, L, Niter):
    ii = 0
    while (ii < Niter-1):
        u[:,:,ii+1], v[:,:,ii+1] = solver.calculate_step(u[:, :, ii], v[:, :, ii], f1[:, :, ii], f2[:, :, ii], rho, mu, dx, dt)
        ii+=1
    
    return u[:,:,0:Niter-3], v[:,:,0:Niter-3]
    
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

def plot_analytic_res(solver, name):
    dx_list = [1]
    dt_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    err_u = np.zeros(len(dt_list))
    err_v = np.zeros(len(dt_list))
    
    for i in range(len(dx_list)):
        for j in range(len(dt_list)):
            err_u[j], err_v[j] = basic_test(solver, dx_list[i], dt_list[j], False)
    
    # Choose colors for the plots
    color_u = 'blue'
    color_v = 'green'
    
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for err_u
    ax[0].plot(dt_list, err_u, marker='o', linestyle='-', color=color_u, label='Error (u)')
    
    # Plot for err_v
    ax[1].plot(dt_list, err_v, marker='o', linestyle='-', color=color_v, label='Error (v)')
    
    # Add grid lines
    for axis in ax:
        axis.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title for the first subplot
    ax[0].set_xlabel('Timestep (dt)')
    ax[0].set_ylabel('Error (u)')
    ax[0].set_title('Error (u) vs Timestep')
    
    # Add labels and title for the second subplot
    ax[1].set_xlabel('Timestep (dt)')
    ax[1].set_ylabel('Error (v)')
    ax[1].set_title('Error (v) vs Timestep')
    
    # Add legends
    ax[0].legend()
    ax[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(name + ".png")

def main():    
    plot_analytic_res(fs, "fs_converge")
    plot_analytic_res(hs, "hs_converge")
    plot_analytic_res(hsu, "hsu_converge")
    
main()
