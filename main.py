import os
from hamiltonian import Hamiltonian
from locator import Locator
from integrator import Integrator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

def main():
    H = Hamiltonian()
    locator = Locator(H)
    integrator = Integrator(H)

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
    else:
        print("couldn't find minimum : {}".format(mini.message))


    if locate_po:
        print ("Locate PO...")
        period = 2
        PO_sol = locator.locatePO(x,period,root_method='hybr')

        if PO_sol.success:
            print("success : {}".format(PO_sol.success))
            #print("PO initial conditions {}".format(PO_init_cond))

            PO = integrator.integrate_plot(PO_sol.x,0,period,'lsoda',10)
            print("trajectory : {}".format(PO))

            plt.subplot(221)
            plt.plot(PO[:,0],PO[:,2],c='red')
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$p_1$')

            plt.subplot(222)
            plt.plot(PO[:,1],PO[:,3],c='red')
            plt.xlabel(r'$x_2$')
            plt.ylabel(r'$p_2$')

            plt.subplot(223)
            plt.plot(PO[:,4],PO[:,0],c='red')
            plt.xlabel(r'time')
            plt.ylabel(r'$x_1$')

            plt.subplot(224)
            plt.plot(PO[:,4],PO[:,1],c='red')
            plt.xlabel(r'time')
            plt.ylabel(r'$x_2$')

            plt.show()
        else:
            print("Couldn't find PO : {}".format(PO_sol.message))

if __name__ == "__main__":
    main()