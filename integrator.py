from hamiltonian import Hamiltonian
from scipy.integrate import ode
import numpy as np

class Integrator:
    def __init__(self,hamiltonian):
        self.hamiltonian = hamiltonian
        self.atol = 1.e-8
        self.rtol = 1.e-8
    
    def integrate_plot(self,x,tstart,tend,method="dop853",npts=100):
        """
        """

        # set up scipy integrator
        integ = ode(self.hamiltonian.get_vector_field).set_integrator(method,atol=self.atol,rtol=self.rtol)
        integ.set_initial_value(x,tstart)

        # do integration
        dt = (tend-tstart)/npts
        E = self.hamiltonian.compute_energy(x)
        res = np.transpose(np.concatenate((x,[E],[tstart]), axis=0))
        for i in range(npts):
            x = integ.integrate(integ.t+dt)
            E = self.hamiltonian.compute_energy(x)
            if integ.successful():
                ite = np.transpose(np.concatenate((x,[E],[integ.t]), axis=0))
                res = np.vstack((res,ite))
        
        return res

    def integrate_variational_plot(self,x,tstart,tend,method="dop853",npts=100):
        """
        integrate trajectory along with variational equations
        """
        # set up scipy integrator
        integ = ode(self.hamiltonian.vect_field_traj_variational)
        integ.set_integrator(method,atol=self.atol,rtol=self.rtol)
        integ.set_initial_value(x,tstart)

        # do integration
        dt = (tend-tstart)/npts
        E = self.hamiltonian.compute_energy(x)
        res = np.transpose(np.concatenate((x,[E],[tstart]), axis=0))
        for i in range(npts):
            x = integ.integrate(integ.t+dt)
            E = self.hamiltonian.compute_energy(x)
            if integ.successful():
                ite = np.transpose(np.concatenate((x,[E],[integ.t]), axis=0))
                res = np.vstack((res,ite))
        
        return res

    def integrate(self,x,tstart,tend,method):
        """
        """

        # set up scipy integrator
        integ = ode(self.hamiltonian.get_vector_field).set_integrator(method,atol=self.atol,rtol=self.rtol)
        integ.set_initial_value(x,tstart)
        # do integration
        #print("integration successful : {}".format(integ.successful()))
        x = integ.integrate(integ.t+tend)
        return x

    def integrate_variational(self,x,tstart,tend,method):
        """
        """

        # set up scipy integrator
        integ = ode(self.hamiltonian.vect_field_traj_variational).set_integrator(method,atol=self.atol,rtol=self.rtol)
        integ.set_initial_value(x,tstart)

        # do integration
        #print("integration successful : {}".format(integ.successful()))
        x = integ.integrate(integ.t+tend)   
        return x