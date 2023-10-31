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
    data.dropna(subset=[41], inplace=True)
    
    for i in [1,2,3]:
        data[i], labels = pd.factorize(data[i])
        
    # print(data)

    return data

# selecting variables
def select_vars(X,param):
    
    Y = X[41]
    X = X.drop(columns=[41])
    
    idx = np.arange(42)
    gain = []
    # for i in range(len(X.columns)):
    #     gain.append(ig.inform_gain(Y, X[i]))
        
    # union = list(zip(gain, idx))
    # combinadas_ordenadas = sorted(union, key=lambda x: x[0], reverse=True)[:int(param[1])]
    # gain, idx = zip(*combinadas_ordenadas)
    
    idx = [6, 17, 13, 21, 11, 14, 7, 1, 4, 10, 18, 3, 12, 15, 16, 9, 5, 27, 25, 30, 26, 36, 24, 38, 29, 2, 40, 39, 37, 0]
    X = X[idx] 
    
    V = rsvd.svd_data(X,Y,param)
    
    return gain, idx, V
#save results
def save_results(gain,idx,V):
    
    gain = pd.DataFrame(gain)
    gain.to_csv('gain_values.csv', index=False)
    
    idx = pd.DataFrame(idx)
    idx.to_csv('gain_idx.csv', index=False)
    
    V = pd.DataFrame(V)
    V.to_csv('filter_v.csv', index=False)
    
    return

#-------------------------------------------------------------------
# Beginning ...
def main():
    param        = load_config()            
    X            = load_data(param)   
    gain, idx, V = select_vars(X,param)                 
    save_results(gain,idx,V)
       
if __name__ == '__main__':   
	 main()

#-------------------------------------------------------------------
