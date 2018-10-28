import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('ggplot')


def plot_traj(traj, dof, filename):
    """
    Plot a trajectory
    """
    # prepare figures
    fig = plt.figure(figsize=(12, 10))
    fig.tight_layout()
    plot_number = 0
    for i in range(dof):
        # plot x(t)
        plot_number += 1
        fig.add_subplot(dof+1,3,plot_number)
        for j in range(len(traj)):
            plt.plot(traj[j][:,-1],traj[j][:,i])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x_{}$'.format(i))
        # plot p(t)
        plot_number += 1
        p_pos = 2*i+1
        if (i==0):
            p_pos = dof
        fig.add_subplot(dof+1,3,plot_number)
        for j in range(len(traj)):
            plt.plot(traj[j][:,-1],traj[j][:,p_pos])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$p_{}$'.format(i))
        # plot p(x)
        plot_number += 1
        fig.add_subplot(dof+1,3,plot_number)
        for j in range(len(traj)):
            plt.plot(traj[j][:,i],traj[j][:,p_pos])
        plt.xlabel(r'$x_{}$'.format(i))
        plt.ylabel(r'$p_{}$'.format(i))

    plot_number += 1
    # get exponent of energy
    # ex = np.floor(np.log10(np.abs(traj[0,-2]))).astype(int)
    fig.add_subplot(dof+1,3,plot_number)
    # plt.plot(traj[:,-1],traj[:,-2]*10.0**(-np.floor(np.log10(np.abs(traj[0,-2]))).astype(int)),c='red')
    for j in range(len(traj)):
            plt.plot(traj[j][:,-1],traj[j][:,-2])
    plt.xlabel(r'$t$')
    # plt.ylabel(r'$E (x 10^{})$'.format(-np.abs(ex)))
    plt.ylabel(r'$E$')

    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.35)
    plt.show()
    #plt.show(block=False)
    
