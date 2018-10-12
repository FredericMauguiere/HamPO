"""
Symplectic utility functions
"""

import numpy as np

def get_metric_mat(dof):
    """
    get symplectic metric matrix J for
    2*dof dimensional phase space
    """
    Id = np.identity(dof)
    nullblock = np.zeros((dof,dof))
    J = np.block([
        [nullblock, Id],
        [-Id, nullblock]
        ])
    #print("symplectic metric:\n{}".format(J))
    return J