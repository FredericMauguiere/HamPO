import matplotlib
#matplotlib.use('gg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')


def plot_traj(traj, filename):
    fig = plt.figure(figsize=(12, 10))
    fig.add_subplot(2,2,1)
    #plt.subplot(221)
    plt.plot(traj[:,0],traj[:,2],c='red')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$p_1$')

    fig.add_subplot(2,2,2)
    #plt.subplot(222)
    plt.plot(traj[:,1],traj[:,3],c='red')
    plt.xlabel(r'$x_2$')
    plt.ylabel(r'$p_2$')

    fig.add_subplot(2,2,3)
    #plt.subplot(223)
    plt.plot(traj[:,4],traj[:,0],c='red')
    plt.xlabel(r'time')
    plt.ylabel(r'$x_1$')

    fig.add_subplot(2,2,4)
    #plt.subplot(224)
    plt.plot(traj[:,4],traj[:,1],c='red')
    plt.xlabel(r'time')
    plt.ylabel(r'$x_2$')

    #cwd = os.getcwd()
    #outputfile = os.path.join(cwd,filename)
    #plt.savefig(outputfile,dpi=300)
    plt.show()
    #plt.show(block=False)
