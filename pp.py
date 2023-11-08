# Pre-proceso: Selecting variables for IDS

import pandas as pd
import numpy          as np
import inform_gain    as ig
import redundancy_svd as rsvd

# Load Parameters
def load_config(dire = 'cnf_sv.csv'):
    
    # Número de Muestras (idx_samples.csv)
    # Top-K de Relevancia : 30
    # Número de Vectores Singulares : 20
    # Clase Normal (s/n) : 1
    # Clase DOS (s/n) : 0
    # Clase Probe (s/n) : 1 
    
    param = np.loadtxt(fname=dire)  
    
    return param

# Load data 
def load_data(param, dire = 'KDDTrain.txt'):
	
    c1 = ['normal']
    c2 = ['neptune', 'teardrop', 'smurf', 'pod', 'back','land', 'apache2', 'processtable', 'mailbomb','udpstorm']
    c3 = ['ipsweep', 'portsweep', 'nmap','satan', 'saint', 'mscan']
    
    data = pd.read_csv(dire, header = None)
    
    data = data.drop(42, axis=1)
    
    data[41].replace(c1, param[3], inplace=True)
    data[41].replace(c2, param[4], inplace=True)
    data[41].replace(c3, param[5], inplace=True)
    
    data[41] = pd.to_numeric(data[41], errors='coerce')
    
    for i in [1,2,3]:
        data[i], labels = pd.factorize(data[i])
        
   

    idx = np.genfromtxt('idx_samples.csv', dtype=int)
    
    #hacer el sampling
    idx = np.asarray(idx)-1
    data = data.iloc[idx[:int(param[0])]]
    

    data.dropna(subset=[41], inplace=True)


    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    data = (data - min_values) / (max_values - min_values + 1.0e-16) 
    

    return data

# selecting variables
def select_vars(X,param):
    
    X = X.sample(frac=1).reset_index(drop=True)
    
    Y = X[41]
    X = X.drop(columns=[41])
    
    idx = np.arange(41)
    gain = []
    for i in range(len(X.columns)):
        gain.append(ig.inform_gain(Y, X[i]))
        
    union = list(zip(gain, idx))
    combinadas_ordenadas = sorted(union, key=lambda x: x[0], reverse=True)[:int(param[1])]
    gain, idx = zip(*combinadas_ordenadas)
    idx = np.asarray(idx)
    
    print(idx)
    X = X[idx]
    
    V = rsvd.svd_data(X,Y,param)
    
    return gain, idx, V
#save results
def save_results(gain,idx,V):
    
    gain = pd.DataFrame(gain)
    gain.to_csv('gain_values.csv', index=False, header=None)
    
    idx = pd.DataFrame(idx)
    idx.to_csv('gain_idx.csv', index=False, header=None)
    
    V = pd.DataFrame(V)
    V.to_csv('filter_v.csv', index=False, header=None)
    
    return

#-------------------------------------------------------------------
# Beginning ...
def main():
    param        = load_config()            
    X            = load_data(param)
    # print(X)
    gain, idx, V = select_vars(X,param)                 
    save_results(gain,idx,V)
       
if __name__ == '__main__':   
	 main()

#-------------------------------------------------------------------
