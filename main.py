import os
from hamiltonian import Hamiltonian
from locator import Locator
from integrator import Integrator
import numpy as np
from symplectic import get_metric_mat
from eigen_system import analyse_equilibrium
from plot import plot_traj
from scipy.optimize import show_options

#matplotlib.style.use('ggplot')

def main():
    H = Hamiltonian()
    locator = Locator(H)
    integrator = Integrator(H)

    J = get_metric_mat(2)

    locate_po = True

    x = np.array([0.7,1.0,0.3,-0.3])

    # print ("Locate minimum by minimum search of Hamiltonian")
    # mini = locator.find_min(x,method='SLSQP',bounds=((0,2),(0,2),(-1,1),(-1,1)))
    # if mini.success:
    #     print("found minimum at : {}".format(mini.x))
    # else:
    #     print("couldn't find minimum : {}".format(mini.message))

    print ("Locate minimum by root finding of Hamiltonian gradient")
    mini = locator.find_saddle(x,root_method='hybr')
    if mini.success:
        print("found critical pt at : {}\n".format(mini.x))
        # analyse stability of critical pt
        eig, eigv, periods = analyse_equilibrium(H, 0.0, mini.x)
    else:
        print("couldn't find minimum : {}".format(mini.message))

    # just to get options for specific solver
    # show_options(solver='root', method='broyden1', disp=True)

    if locate_po and mini.success:
        print ("\n\n########################\nTrying to locate PO...")
        period = periods[0] + periods[0]*0.05
        dev = np.array([0.5,0,0.1,0])
        xini = mini.x + dev #0.1*eigv[:,0]
        
        # variational_0 = np.array(np.identity(2*H.dof)).flatten()
        # xstart = np.concatenate((xini, variational_0))
        # print("xstart:\n{}".format(xstart))
        # traj = integrator.integrate_variational_plot(x=xstart, tstart=0., tend=period, npts=50)
        # # traj = integrator.integrate_plot(x=xini, tstart=0., tend=period, npts=50)
        # print("trajectory:\n{}".format(traj[-1]))
        # plot_traj(traj,H.dof,'traj.pdf')
        
        PO_sol = locator.locatePO(xini,period,root_method='broyden1',integration_method="dop853")

        if PO_sol.success:
            print("success : {}".format(PO_sol.success))
            print("PO initial conditions {}".format(PO_sol.x))

            PO = integrator.integrate_plot(PO_sol.x,0,period,'dop853',100)
            print("PO:\n{}".format(PO))
            plot_traj(PO,H.dof,'PO.pdf')
            
        else:
            print("Couldn't find PO : {}".format(PO_sol.message))

if __name__ == "__main__":
    main()