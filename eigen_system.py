from numpy import linalg as LA

def compute_eigenval_mat(M):
    """
    Compute eigenvalues and eigenvectors
    of symmetric matrix
    """
    return LA.eig(M)