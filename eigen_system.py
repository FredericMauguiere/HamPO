import numpy as np
from numpy import linalg as LA
from symplectic import get_metric_mat

def compute_eigenval_mat(M):
    """
    Compute eigenvalues and eigenvectors
    of symmetric matrix
    """
    return LA.eig(M)

def analyse_equilibrium(hamiltonian, t, x):
    """
    Analyses linear stability of equilibrium pt
    """
    dim = x.size
    # compute matrix of derivatives of vector field at equilibrium
    J = get_metric_mat(hamiltonian.dof)
    df =  np.matmul(J,hamiltonian.get_hessian(t, x))
    eig, eigenvec = compute_eigenval_mat(df)
    n_pure_imag = 0
    n_pure_real = 0
    n_pure_real_neg = 0
    n_pure_real_pos = 0
    n_non_zero = 0
    periods = []
    for i in range(eig.size):
        if (np.allclose([np.real(eig[i])],[0.0]) and not np.allclose([np.imag(eig[i])],[0.0])):
            n_pure_imag += 1
            periods.append(2.*np.pi/abs(np.imag(eig[i])))
        if (np.allclose([np.imag(eig[i])],[0.0]) and not np.allclose([np.real(eig[i])],[0.0])):
            n_pure_real += 1
            if (np.real(eig[i])>0.0):
                n_pure_real_pos += 1
            if (np.real(eig[i])<0.0):
                n_pure_real_neg += 1
        if (np.allclose([np.real(eig[i])],[0.0]) or np.allclose([np.imag(eig[i])],[0.0])):
            n_non_zero += 1
        print("Eigenvalue: ({:4e} + {:4e} i)\nEigenvector:".format(np.real(eig[i]),np.imag(eig[i])))
        for j in range(eigenvec[:,i].size):
            print("{}".format(eigenvec[j,i]))
        print("\n")
    
    print("number of non zero eigenvalues: {}".format(n_non_zero))
    print("number of pure imaginary eigenvalues: {}".format(n_pure_imag))
    print("number of real eigenvalues: {} ({} positives and {} negatives)\n".
    format(n_pure_real, n_pure_real_pos, n_pure_real_neg))
    
    if (n_pure_imag == 0 and n_non_zero==dim): # hyperbolic case
        print("critical point is hyperbolic")
    if (n_pure_imag > 0): # non hyperbolic case
        print("critical point is non hyperbolic")
        if (dim==n_pure_imag):
            print("critical point is a center")
            for p in periods:
                print("center harmonic period: {}".format(p))
    return eig, eigenvec, periods