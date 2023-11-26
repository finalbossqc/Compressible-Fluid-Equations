#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

"""
Description: A plotting function that creates an animated heatmap
Arguments:
    u: NxN double
        - The velocity (either x or y velocity)
    variable: string
        - The name of the variable to be plotted.
    params: tuple 
        - A tuple of parameters to be put in the title
"""
def animated_heatmap(u, variable, params):
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