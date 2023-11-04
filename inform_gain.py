# My Utility : 

import numpy  as np   
import pandas as pd
# information gain
def inform_gain(Y,x):
    return inform_estimate(Y) - entropy_xy(x, Y)
    
    

# estimation of information
def inform_estimate(y):
    y = pd.DataFrame(y)
    p = y.value_counts()/len(y)
    # print(p)
    return -sum( p * np.log2(p))

# Entropy of the variables  
def entropy_xy(x,y):
    
    N = len(x)
    B = int(np.floor(np.sqrt(N)))
    y = np.asarray(y)    
    x = np.asarray(x)
    Xmax ,Xmin = np.max(x), np.min(x)
    l = (Xmax - Xmin)/(B-1) 
    
    d = []
    for i in range(B):
        d_i = [y[j] for j , a in enumerate(x) if (i*l)+Xmin <= a < (i*l)+l+Xmin]
        d_i = pd.DataFrame(d_i)
        d_i = d_i.value_counts()
        d_i = sum(d_i)/N
        if d_i != 0:
            d.append(d_i)
    d = np.asarray(d)
    I = inform_estimate(d)
    
    return sum(d) * I

#-----------------------------------------------------------------------
