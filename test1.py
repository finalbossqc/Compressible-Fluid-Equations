#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from full_step_ns import calculate_full_step
import numpy as np

def main():
    N = 20
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    f1 = np.ones((N, N))*3
    f2 = np.zeros((N, N))
    rho = 1
    mu = 1
    dx = 1/N
    dt = 0.2
    
    u_next, v_next = calculate_full_step(u, v, f1, f2, rho, mu, dx, dt)
    
main()