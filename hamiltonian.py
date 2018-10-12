import numpy as np
from symplectic import get_metric_mat

# Hamiltonian class
class Hamiltonian:
    """
    class for Hamiltonian
    """
    def __init__(self):
        print("Hamiltonian instantiation...")
        # 2D Morse example
        # parameters
        self.dof = 2
        self.m1 = 1.0
        self.m2 = 2.0
        self.beta1 = 0.5
        self.beta2 = 1.0
        self.re1 = 0.5
        self.re2 = 0.7
        self.D0 = 1.0
    
    def compute_energy(self,x):
        """
        Compute Energy
        @param x input numpy vector, phase space coordinates
        @param t input scalar, time
        @return Energy scalar
        """
        # kinetic energy
        T1 = pow(x[2],2)/(2.0*self.m1)
        T2 = pow(x[3],2)/(2.0*self.m2)
        #T12 = -x[2]*x[3]/(m1*m2/(m1+m2))
        T = T1 + T2 #+ T12
	
        # potential energy
        Vm1 = self.D0 * pow((1.0-np.exp(-self.beta1*(x[0]-self.re1))),2)
        Vm2 = self.D0 * pow((1.0-np.exp(-self.beta2*(x[1]-self.re2))),2)
        V = Vm1 + Vm2

        # energy
        Energy = T+V

        return Energy

    def get_vector_field(self,t,x):
        """
        get vector field associated with the Hamiltonian
        @param x input vector, phase space coordinates
        @param t input scalar, time
        @return vector 
        """
        """ vf = np.zeros(2*self.dof)
        # x[0] dot dH/dx[2]
        vf[0] = x[2]/self.m1
        # x[1] dot dH/dx[3]
        vf[1] = x[3]/self.m2
        # x[2] dot = -dH/dx[0]
        vf[2] = -2.0*self.beta1*self.D0*np.exp(-self.beta1*(x[0]-self.re1))*(1.0-np.exp(-self.beta1*(x[0]-self.re1)))
        # x[3] dot = -dH/dx[1]
        vf[3] = -2.0*self.beta2*self.D0*np.exp(-self.beta2*(x[1]-self.re2))*(1.0-np.exp(-self.beta2*(x[1]-self.re2))) """

        J = get_metric_mat(self.dof)
        gradH = self.get_grad(x)
        vf = np.matmul(J,gradH)

        return vf

    def get_grad(self,x):
        grad = np.zeros(2*self.dof)
        
        # dH/dx[0]
        grad[0] = 2.0*self.beta1*self.D0*np.exp(-self.beta1*(x[0]-self.re1))*(1.0-np.exp(-self.beta1*(x[0]-self.re1)))
        # dH/dx[1]
        grad[1] = 2.0*self.beta2*self.D0*np.exp(-self.beta2*(x[1]-self.re2))*(1.0-np.exp(-self.beta2*(x[1]-self.re2)))
        # dH/dx[2]
        grad[2] = x[2]/self.m1
        # dH/dx[3]
        grad[3] = x[3]/self.m2
        
        return grad

    def compute_hessian(self,x):
        """
        Compute Hessian matrix (second derivatives of Hamiltonian)
        at point x
        """
        # get symplectic matrix
        J = get_metric_mat(self.dof)
        hes = np.zeros((2*self.dof,2*self.dof))
        t11 = np.exp(-self.beta1*(x[0]-self.re1))
        t12 = 1.0-np.exp(-self.beta1*(x[0]-self.re1))
        t21 = np.exp(-self.beta2*(x[1]-self.re2))
        t22 = 1.0-np.exp(-self.beta2*(x[1]-self.re2))

        # either we compute matrix elt analytically or we use numerical derivates
        hes[0,2] = 1.0/self.m1
        
        hes[1,3] = 1.0/self.m2

        hes[2,0] = 2.0 * (self.beta1)**2.0 * self.D0 * t11**2.0 + \
        2.0 * self.D0 * t12 * (self.beta1)**2.0 * t12
        
        hes[3,1] = 2.0 * (self.beta2)**2.0 * self.D0 * t21**2.0 + \
        2.0 * self.D0 * t22 * (self.beta2)**2.0 * t22
        return hes