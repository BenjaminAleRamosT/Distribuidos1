# My Utility : 

import numpy  as np   

# information gain
def inform_gain(Y,x):
    return inform_estimate(Y) - entropy_xy(x, Y)
    
    

# estimation of information
def inform_estimate(y):
    
    p = np.unique(y)/len(y)

    return -sum( p * np.log2(p))

# Entropy of the variables  
def entropy_xy(x,y):
    
    N = len(x)
    B = int(np.floor(np.sqrt(N)))
    

    Xmax ,Xmin = np.max(x), np.min(x)
    l = (Xmax - Xmin)/(B-1) 
    
    d = []
    for i in range(B):
        d_i = len([a for a in x if (i*l)+Xmin <= a < (i*l)+l+Xmin])
        if d_i != 0:
            d.append(d_i)
    d = np.asarray(d)
    I = inform_estimate(d)
    p = sum(np.asarray(d)/N)
    
    
    return p * I



#-----------------------------------------------------------------------
