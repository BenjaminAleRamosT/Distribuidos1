# My Utility : 

import numpy  as np   

# normalization of the data
def norm_data(x):
    """
    Normalizes the data in x
    :param x: input data to be normalized
    :return: normalized x
    """
    dfmax, dfmin = np.max(x), np.min(x)
    x = (x - dfmin)/(dfmax - dfmin)

    return x
# SVD of the data
def svd_data():
    ...    
    return()



#-----------------------------------------------------------------------
