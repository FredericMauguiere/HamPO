import os
from hamiltonian import Hamiltonian
from locator import Locator
from integrator import Integrator
import numpy as np
from symplectic import get_metric_mat
from eigen_system import analyse_equilibrium, compute_eigenval_mat
from plot import plot_traj
from scipy.optimize import show_options

def main():
    H = Hamiltonian()
    locator = Locator(H)
    integrator = Integrator(H)

    J = get_metric_mat(2)

    locate_min = True
    locate_po = True
    plot = True
    period = 3.16
    n = 2
    delta_T = 0.005

    x = np.array([0.5,0,0.1,0])

    if locate_min:
        print ("########################\nLocating minimum by root finding of Hamiltonian gradient")
        mini = locator.find_saddle(x,root_method='hybr')
        if mini.success:
            print("Found critical pt at : {}\n".format(mini.x))
            # analyse stability of critical pt
            eig, eigv, periods = analyse_equilibrium(H, 0.0, mini.x)
        else:
            print("Couldn't find minimum : {}".format(mini.message))

    if locate_po:
        PO_init = []
        PO = [] # list to accumulate found POs
        for i in range(n):
            if i==0:
                if locate_min and mini.success:
                    period = periods[0] + periods[0]*0.05
                    dev = np.array([0.5,0,0.1,0])
                    xini = mini.x + dev
                else:
                    period = period 
                    xini = x
            print ("\n\n########################\nTrying to locate PO {}\n starting at {}\nperiod: {}"
            .format(i,xini,period))
            PO_sol = locator.locatePO(xini,period,root_method='broyden1',integration_method="dop853")

            if PO_sol.success:
                print("Found PO at initial conditions {}".format(PO_sol.x))
                PO_init.append(PO_sol.x)
                xini = PO_sol.x
                period += delta_T

                # compute monodromy matrix
                variational_0 = np.array(np.identity(2*H.dof)).flatten()
                xstart = np.concatenate((PO_sol.x, variational_0))
                # print("xstart:\n{}".format(xstart))
                traj = integrator.integrate_variational(x=xstart, tstart=0., tend=period,method="dop853")
                # last = traj[-1]
                monod = np.reshape(traj[2*H.dof:], (2*H.dof, 2*H.dof))
                # print("monodromy matrix:\n{}".format(monod))
                eig, eigenvec = compute_eigenval_mat(monod)
                print("monodromy matrix eigenvalues:")
                for i in range(eig.size):
                    print("eigenvalue: {}".format(eig[i]))
                if plot:
                    PO.append(integrator.integrate_plot(PO_sol.x,0,period,'dop853',100))
                    # print("PO:\n{}".format(PO))
                    #plot_traj(PO,H.dof,'PO.pdf')
                
            else:
                print("Couldn't find PO : {}".format(PO_sol.message))
                period += delta_T
        if plot:
            plot_traj(PO,H.dof,'PO.pdf')


if __name__ == "__main__":
    main()