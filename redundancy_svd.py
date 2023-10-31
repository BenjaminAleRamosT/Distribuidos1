# My Utility : 

import numpy  as np   

# normalization of the data
def norm_data(X):

    # dfmax, dfmin = np.max(x), np.min(x)
    # x = (x - dfmin)/(dfmax - dfmin)

    X = X/(np.sqrt(len(X.columns)-1))

    return X

# SVD of the data
def svd_data(X,Y,param):
    
    media_columnas = X.mean()
    X = X - media_columnas
    X = np.asarray(norm_data(X))
    
    U, S, Vt = np.linalg.svd(X)
    
    Vt = Vt[:,:int(param[2])]
    
    
    return Vt



#-----------------------------------------------------------------------
