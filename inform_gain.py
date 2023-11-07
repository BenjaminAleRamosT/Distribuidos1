# My Utility : 

import numpy  as np   
import pandas as pd
# information gain
def inform_gain(Y,x):
    
    Y = pd.DataFrame(Y)
    d = Y.value_counts()
    I = inform_estimate(d,len(Y))
    return I - entropy_xy(x, Y) 
    
# estimation of information
def inform_estimate(d_i,N):
    p = d_i / N
    # print(p)
    return -sum( p * np.log2(p))

# Entropy of the variables  
def entropy_xy(x,y):
    
    N = len(x)
    B = int(np.floor(np.sqrt(N)))
    y = np.asarray(y)    
    x = np.asarray(x)
    Xmax ,Xmin = np.max(x), np.min(x)
    l = (Xmax - Xmin)/(B) 
    
    d = []
    
    min_ran = (np.arange(B)*l)+Xmin
    max_ran = min_ran + l
    #######
    for i in range(B):
        print((min_ran[i] <= x) & (x < max_ran[i]))
        # d_i = np.where((min_ran[i] <= x) & (x < max_ran[i]), y, None)
        d_i = [y[j][0] for j , a in enumerate(x) if min_ran[i] <= a < max_ran[i]]
        d_i = pd.DataFrame(d_i)
        d_i = d_i.value_counts()
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
