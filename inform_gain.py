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

def sumar_la_wa(p_ij, i=0, suma=0 ):
    
    suma += (sum(p_ij[i]) * inform_estimate(p_ij[i]))
    
    if i == p_ij.shape[0]-1:
        print(suma)
        input()
        return suma
    
    return sumar_la_wa(p_ij, i+1, suma)

def sumar_la_wa2(p_ij):
    
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
    
    d = []
    
    # min_ran = (np.arange(B)*l)+Xmin
    # max_ran = min_ran + l
    #######
    
    arraypos = np.floor((x-Xmin)/ l )
    
    # Identificar los elementos iguales al valor máximo
    condicion = arraypos == np.max(arraypos)

    # Restar 1 solo a los valores máximos
    arraypos[condicion] -= 1

    
    
    d_ij = np.zeros((N,2))
    
    for i in range(N):
        d_ij[int(arraypos[i])][int(y[i])] += 1
    
    return sumar_la_wa2(d_ij/N)
    
    
    
    # d_i = np.where((min_ran[i] <= x) & (x < max_ran[i]), y, None)
    #d_i = [y[j][0] for j , a in enumerate(x) if min_ran[i] <= a < max_ran[i]]
    # d_i = pd.DataFrame(d_i)
    # d_i = d_i.value_counts()
    
    if Xmax == Xmin:
        d_i = []
    if len(d_i) != 0:
        I = inform_estimate(d_i,N)
        d_i = sum(d_i)/N
        d.append(d_i*I)
    
    ######
    
    
    
    
    # for i in range(B):
    #     d_i = [y[j] for j , a in enumerate(x) if (i*l)+Xmin <= a < (i*l)+l+Xmin]
    #     d_i = pd.DataFrame(d_i)
    #     d_i = d_i.value_counts()
    #     if Xmax == Xmin:
    #         d_i = []
    #     if len(d_i) != 0:
    #         I = inform_estimate(d_i,N)
    #         d_i = sum(d_i)/N
    #         d.append(d_i*I)
    d = np.asarray(d)
    return sum(d)

#-----------------------------------------------------------------------
