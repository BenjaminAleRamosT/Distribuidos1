# My Utility : 

import numpy  as np   
import pandas as pd
# information gain
def inform_gain(Y,x):
    
    Y = pd.DataFrame(Y)
    d = Y.value_counts()
    I = inform_estimate(d/len(Y))
    return I - entropy_xy(x, Y) 
    
# estimation of information
def inform_estimate(p):
    
    # print(p)
    p = np.asarray(p)
    p = p[np.where(p != 0)]
    return -sum( p * np.log2(p))

def sumar(p_ij):
    
    suma=0
    for i in range(p_ij.shape[0]):
        suma += (sum(p_ij[i]) * inform_estimate(p_ij[i]))
    return suma


# Entropy of the variables  
def entropy_xy(x,y):
    
    N = len(x)
    B = int(np.floor(np.sqrt(N)))
    y = np.asarray(y)    
    x = np.asarray(x)
    Xmax ,Xmin = np.max(x), np.min(x)
    l = (Xmax - Xmin)/(B)
    
    if l == 0:
        return 0
    
    arraypos = np.floor((x-Xmin)/ l )
    
    condicion = arraypos == np.max(arraypos)
    arraypos[condicion] -= 1

    d_ij = np.zeros((N,3)) ############ACA EL @ CORRESPONDE AL NUMERO DE CLASES
    #### NO SE ME OCURRIO COMO SACARLO JIJI SIN DEJARLO HARCODEAO EN 3 
    ## Y DESPUES BORRANDO LAS CLASES QUE VALGAN 0
    
    for i in range(N):
        d_ij[int(arraypos[i])][int(y[i])] += 1
    
    return sumar(d_ij/N)
    

#-----------------------------------------------------------------------
