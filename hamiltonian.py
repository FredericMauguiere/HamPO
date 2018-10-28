import numpy as np
from symplectic import get_metric_mat
import sympy as sy

# Hamiltonian class
class Hamiltonian:
    """
    class for Hamiltonian
    """
    def __init__(self,verbose=False):
        """
        Initialise Hamiltonian \n
        Phase space coordinates are [x_{0},...,x_{2n}]\n
        x_{0},...,x_{n-1} are configuration space coordinates\n
        x_{n},...,x_{2n-1} are conjugate momenta\n

        Given a Hamiltonian, we compute first and second derivatives
        automatically with sympy.
        """
        if verbose:
            print("################################\nHamiltonian construction...")
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

        # define phase space coordinates
        exec("self.x = list(sy.symbols('x0:%d'))" % (2*self.dof))
        if verbose:
            print("Phase space coordinates: {}".format(self.x))

        # define Hamiltonian function
        if verbose:
            print("\nHamiltonian:")
        self.h = sy.Pow(self.x[2],2)/(2*self.m1) +\
            sy.Pow(self.x[3],2)/(2*self.m2) +\
            self.k1 * sy.Pow(self.x[0],2) + self.c1 * sy.Pow(self.x[0],3) +\
            self.k2 * sy.Pow(self.x[1],2) + self.c2 * sy.Pow(self.x[1],3)
        if verbose:
            print("H = {}".format(self.h))
        # lambdify
        self.h_lamb = sy.lambdify(tuple(self.x),self.h)

        # gradient
        if verbose:
            print("\nFirst derivatives:")
        self.grad = []
        self.grad_lamb = []
        for i in range(2*self.dof):
            self.grad.append(sy.diff(self.h, self.x[i]))
            # lambdify
            self.grad_lamb.append(sy.lambdify(tuple(self.x),self.grad[i]))
            if verbose:
                print("dH/dx{} = {}".format(i,self.grad[i]))
    
        # hessian
        if verbose:
            print("\nsecond derivatives:")
        self.hess = []
        self.hess_lamb = []
        for i in range(2*self.dof):
            row = []
            row_lamb = []
            for j in range(2*self.dof):
                row.append(sy.diff(self.grad[i],self.x[j]))
                # lambdify
                row_lamb.append(sy.lambdify(tuple(self.x),row[j]))
                if verbose:
                    print("d2H/dx{}dx{} = {}".format(i,j,row[j]))
            self.hess.append(row)
            self.hess_lamb.append(row_lamb)
        if verbose:
            print("################################")

    def compute_energy(self,x):
        """
        Compute Energy
        @param x input numpy vector, phase space coordinates
        @param t input scalar, time
        @return Energy scalar
        """
        # get coordinates as string 
        s = ','.join(str(e) for e in x)

        # evaluate hamiltonian at x
        Energy = eval("self.h_lamb(%s)" % s)

        # der = sy.diff(self.h, self.x[0])
        # print("der:\n{}".format(der))
        return Energy

    def get_vector_field(self,t,x):
        """
        get vector field associated with the Hamiltonian
        @param x input vector, phase space coordinates
        @param t input scalar, time
        @return numpy array 
        """
        J = get_metric_mat(self.dof)
        gradH = self.get_grad(x)
        vf = np.matmul(J,gradH)
        # print("vect field type: {}".format(type(vf)))

        return vf

    def get_grad(self,x):
        """
        Compute gradient of H at x
        @param x input vector, phase space coordinates
        @return numpy array
        """
        s = ','.join(str(e) for e in x)
        grad = np.zeros(2*self.dof)
        for i in range(2*self.dof):
            grad[i] = eval("self.grad_lamb[%d](%s)" % (i,s))
        return grad

    def get_hessian(self,t,x):
        """
        Get Hessian matrix (second derivatives of Hamiltonian)
        at point x
        """
        s = ','.join(str(e) for e in x)

        hes = np.zeros((2*self.dof,2*self.dof))
        for i in range(2*self.dof):
            for j in range(2*self.dof):
                hes[i,j] = eval("self.hess_lamb[%d][%d](%s)" % (i,j,s))

        # t11 = np.exp(-self.beta1*(x[0]-self.re1))
        # t12 = 1.0-np.exp(-self.beta1*(x[0]-self.re1))
        # t21 = np.exp(-self.beta2*(x[1]-self.re2))
        # t22 = 1.0-np.exp(-self.beta2*(x[1]-self.re2))

        # either we compute matrix elt analytically or we use numerical derivates
        # hes[0,2] = 1.0/self.m1
        
        # hes[1,3] = 1.0/self.m2

        # hes[2,0] = 2.0 * (self.beta1)**2.0 * self.D0 * t11**2.0 + \
        # 2.0 * self.D0 * t12 * (self.beta1)**2.0 * t12
        
        # hes[3,1] = 2.0 * (self.beta2)**2.0 * self.D0 * t21**2.0 + \
        # 2.0 * self.D0 * t22 * (self.beta2)**2.0 * t22

        # anharmonic oscillator case
        # hes[2,0] = -2.0*self.k1 - 6.0*self.c1*x[0]
        # hes[3,1] = -2.0*self.k2 - 6.0*self.c2*x[1]
        return hes

    def vect_field_traj_variational(self,t,x):
        """
        vector containing vector field + derivativates for
        variational equations
        We concatenate everything into a 2n+4n^2 big vector
        (n = nbre of dof)
        """
        vect_field = self.get_vector_field(t,x[:2*self.dof])
        hess = self.get_hessian(t,x[:2*self.dof])
        J = get_metric_mat(self.dof)
        hess = np.matmul(J,hess)
        flow_sol_mat = np.reshape(x[2*self.dof:],(2*self.dof, 2*self.dof))
        der_flow = np.matmul(hess,flow_sol_mat)
        der_flow = der_flow.flatten()
        full_vect_field = np.concatenate((vect_field, der_flow))
        return full_vect_field
        # return hess