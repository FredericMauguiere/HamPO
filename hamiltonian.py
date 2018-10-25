import numpy as np
from symplectic import get_metric_mat
from sympy import *

# Hamiltonian class
class Hamiltonian:
    """
    class for Hamiltonian
    """
    def __init__(self):
        #print("Hamiltonian instantiation...")
        # 2D Morse example
        # parameters
        self.dof = 2
        self.m1 = 1.0
        self.m2 = 3.0
        self.k1 = 2.0
        self.k2 = 3.0
        self.beta1 = 0.5
        self.beta2 = 1.0
        self.c1 = 1.0
        self.c2 = 1.0
        self.re1 = 0.5
        self.re2 = 0.7
        self.D0 = 1.0

        print("define phase space coordinates")
        self.x = []
        for i in range(2*self.dof):
            name = 'self.x' + str(i)
            print("coordinate {}: {}".format(i,name))
            print("command : %s = Symbol('x%d')" % (name,i))
            exec("%s = Symbol('x%d')" % (name,i))
            exec("self.x.append(%s)" % (name))
        print(self.x)

        # define Hamiltonian function
        self.h = self.x[2]**2/(2.0*self.m1) +\
                 self.x[3]**2/(2.0*self.m2) +\
                 self.k1 * self.x[0]**2 + self.c1 * self.x[0]**3 +\
                 self.k2 * self.x[1]**2 + self.c2 * self.x[1]**3
        print("Hamiltonian:\n{}".format(self.h))

    
    def compute_energy(self,x_):
        """
        Compute Energy
        @param x input numpy vector, phase space coordinates
        @param t input scalar, time
        @return Energy scalar
        """
        # sympy
        # for i in range
        # fvars = [] #these probably already exist, use: fvars = [x,y]
        Energy = self.h.subs(self.x, x_).evalf()
        # E = evalf(self.h, subs = dict(zip(self.x,x)))

        # # kinetic energy
        # T1 = x[2]**2/(2.0*self.m1)
        # T2 = x[3]**2/(2.0*self.m2)
        # #T12 = -x[2]*x[3]/(m1*m2/(m1+m2))
        # T = T1 + T2 #+ T12
	
        # # Morse potential energy
        # #V1 = self.D0 * pow((1.0-np.exp(-self.beta1*(x[0]-self.re1))),2)
        # #V2 = self.D0 * pow((1.0-np.exp(-self.beta2*(x[1]-self.re2))),2)
        
        # # anhamonic oscillator
        # V1 = self.k1 * x[0]**2 + self.c1 * x[0]**3
        # V2 = self.k2 * x[1]**2 + self.c2 * x[1]**3
        
        # V = V1 + V2

        # # energy
        # Energy = T+V

        return Energy

    def get_vector_field(self,t,x):
        """
        get vector field associated with the Hamiltonian
        @param x input vector, phase space coordinates
        @param t input scalar, time
        @return vector 
        """
        J = get_metric_mat(self.dof)
        gradH = self.get_grad(x)
        vf = np.matmul(J,gradH)
        # print("vect field type: {}".format(type(vf)))

        return vf

    def get_grad(self,x):
        grad = np.zeros(2*self.dof)
        
        # Morse case
        # dH/dx[0]
        #grad[0] = 2.0*self.beta1*self.D0*np.exp(-self.beta1*(x[0]-self.re1))*(1.0-np.exp(-self.beta1*(x[0]-self.re1)))
        # dH/dx[1]
        #grad[1] = 2.0*self.beta2*self.D0*np.exp(-self.beta2*(x[1]-self.re2))*(1.0-np.exp(-self.beta2*(x[1]-self.re2)))

        # anharmonic oscillator case
        grad[0] = 2.0*self.k1 * x[0] + 3.0*self.c1 * x[0]**2
        grad[1] = 2.0*self.k2 * x[1] + 3.0*self.c2 * x[1]**2
        
        # dH/dx[2]
        grad[2] = x[2]/self.m1
        # dH/dx[3]
        grad[3] = x[3]/self.m2
        
        return grad

    def compute_hessian(self,t,x):
        """
        Compute Hessian matrix (second derivatives of Hamiltonian)
        at point x
        """
        # get symplectic matrix
        hes = np.zeros((2*self.dof,2*self.dof))
        # t11 = np.exp(-self.beta1*(x[0]-self.re1))
        # t12 = 1.0-np.exp(-self.beta1*(x[0]-self.re1))
        # t21 = np.exp(-self.beta2*(x[1]-self.re2))
        # t22 = 1.0-np.exp(-self.beta2*(x[1]-self.re2))

        # either we compute matrix elt analytically or we use numerical derivates
        hes[0,2] = 1.0/self.m1
        
        hes[1,3] = 1.0/self.m2

        # hes[2,0] = 2.0 * (self.beta1)**2.0 * self.D0 * t11**2.0 + \
        # 2.0 * self.D0 * t12 * (self.beta1)**2.0 * t12
        
        # hes[3,1] = 2.0 * (self.beta2)**2.0 * self.D0 * t21**2.0 + \
        # 2.0 * self.D0 * t22 * (self.beta2)**2.0 * t22

        # anharmonic oscillator case
        hes[2,0] = -2.0*self.k1 - 6.0*self.c1*x[0]
        hes[3,1] = -2.0*self.k2 - 6.0*self.c2*x[1]
        return hes

    def vect_field_traj_variational(self,t,x):
        """
        vector containing vector field + derivativates for
        variational equations
        We concatenate everything into a 2n+4n^2 big vector
        (n = nbre of dof)
        """
        vect_field = self.get_vector_field(t,x)
        hess = self.compute_hessian(t,x)
        flow_sol_mat = np.reshape(x[2*self.dof:],(2*self.dof, 2*self.dof))
        der_flow = np.matmul(hess,flow_sol_mat)
        der_flow = der_flow.flatten()
        full_vect_field = np.concatenate((vect_field, der_flow))
        return full_vect_field