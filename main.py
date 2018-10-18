import os
from hamiltonian import Hamiltonian
from locator import Locator
from integrator import Integrator
import numpy as np
from symplectic import get_metric_mat
from eigen_system import compute_eigenval_mat
#from plot import plot_traj

#matplotlib.style.use('ggplot')

def main():
    H = Hamiltonian()
    locator = Locator(H)
    integrator = Integrator(H)

    J = get_metric_mat(2)

    locate_po = True

    x = np.array([0.7,1.0,0.3,-0.3])

    print ("Locate minimum by minimum search of Hamiltonian")
    mini = locator.find_min(x,method='SLSQP',bounds=((0,2),(0,2),(-1,1),(-1,1)))
    if mini.success:
        print("found minimum at : {}".format(mini.x))
    else:
        print("couldn't find minimum : {}".format(mini.message))

    print ("Locate minimum by root finding of Hamiltonian gradient")
    mini = locator.find_saddle(x,root_method='hybr')
    if mini.success:
        print("found minimum at : {}".format(mini.x))
        # compute hessian at minimum
        df = H.compute_hessian(mini.x)
        print("hessian at critical point:\n{}".format(df))
        eig, eigenvec = compute_eigenval_mat(df)
        scale = lambda x: 2.0*np.pi/x 
        periods = scale(eig)
        for i in range(eig.size):
            print("{}, {}\n{}".format(eig[i],periods[i],eigenvec[:,i]))
    else:
        print("couldn't find minimum : {}".format(mini.message))


    if locate_po:
        print ("Locate PO...")
        period = periods[0]+periods[0]*0.1
        xini = mini.x + 0.1*eigenvec[:,0]
        
        traj = integrator.integrate_plot(x=xini, tstart=0., tend=period,
        method='lsoda', npts=100)
        print("trajectory:\n{}".format(traj))
        #plot_traj(traj,'traj.pdf')
        
        # PO_sol = locator.locatePO(mini.x+disp,period,root_method='hybr')

        # if PO_sol.success:
        #     print("success : {}".format(PO_sol.success))
        #     #print("PO initial conditions {}".format(PO_init_cond))

        #     PO = integrator.integrate_plot(PO_sol.x,0,period,'lsoda',100)
        #     plot_traj(PO,'PO.pdf')
            
        # else:
        #     print("Couldn't find PO : {}".format(PO_sol.message))

if __name__ == "__main__":
    main()