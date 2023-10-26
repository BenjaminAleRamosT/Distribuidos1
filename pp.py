# Pre-proceso: Selecting variables for IDS

import pandas as pd
import numpy          as np
import inform_gain    as ig
import redundancy_svd as rsvd

# Load Parameters
def load_config(dire = 'cnf_sv.csv'):
    
    param = np.loadtxt(fname=dire)  
    
    return param

# Load data 
def load_data(dire = 'KDDTrain.txt'):
	
    c1 = ['normal']
    c2 = ['neptune', 'teardrop', 'smurf', 'pod', 'back','land', 'apache2', 'processtable', 'mailbomb','udpstorm']
    c3 = ['ipsweep', 'portsweep', 'nmap','satan', 'saint', 'mscan']
    
    data = pd.read_csv(dire, header = True)
    print(data)
    
    
    return data

# selecting variables
def select_vars(X,param):
	gain, idx, V = 0,0,0
	return gain, idx, V

#save results
def save_results(gain,idx,V):
    
    return

#-------------------------------------------------------------------
# Beginning ...
def main():
    param        = load_config()            
    X            = load_data()   
    gain, idx, V = select_vars(X,param)                 
    save_results(gain,idx,V)
       
if __name__ == '__main__':   
	 main()

#-------------------------------------------------------------------
