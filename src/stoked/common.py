from stoked.forces import pairwise_central_force

def lennard_jones(rmin, eps):
    """Lennard-Jones pairwise interactions

    Arguments:
        rmin       distance where potential is a minimum 
        eps        depth of the potential well
    """

    def F(r):
        return 12*eps*(rmin**12/r**13 - rmin**6/r**7)

    return pairwise_central_force(F)
