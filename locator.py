from scipy import optimize
from integrator import Integrator
from hamiltonian import Hamiltonian

class Locator:
    """
    Class for various location tools:
    1) locate periodic orbits
    2) locate minimum of a multivariate function
    3) locate saddle points of a multivariate function (find zeros of its derivatives)
    """
    def __init__(self,hamiltonian):
        self.hamiltonian = hamiltonian
        self.opt = {"maxiter":1000,
        #"ftol":1.e-5,
        #"fatol":1.e-5,
        "disp":True,
        #"xtol" : 1.e-8,
        #"xatol" : 1.e-8
        }
        self.tol = 1.e-8

    def locatePO(self,xinit,period,root_method='broyden1',integration_method='lsoda'):
        extra_args = period, integration_method
        sol = optimize.root(fun = self.func,
                            x0 = xinit,
                            args = extra_args,
                            method = root_method,
                            jac = None,
                            tol = self.tol,
                            callback = None,
                            options = None
                            )
        return sol

    def func(self,x,period,integration_method):
        integrator = Integrator(self.hamiltonian)
        tstart = 0
        tend = period
        xfinal = integrator.integrate(x,tstart,tend,integration_method)
        #print("func output : {}".format(x-xfinal))
        return x-xfinal

    def find_min(self,xinit,method='SLSQP',bounds=None,constraints=({})):
        res = optimize.minimize(self.hamiltonian.compute_energy,
                                xinit,
                                method=method,
                                jac=self.hamiltonian.get_grad,
                                bounds=bounds)
        
        return res

    def find_saddle(self,xinit,root_method='broyden1'):
        sol = optimize.root(fun = self.hamiltonian.get_grad,
                            x0 = xinit,
                            method = root_method,
                            jac = None,
                            tol = self.tol,
                            callback = None,
                            options = None
                            )
        return sol