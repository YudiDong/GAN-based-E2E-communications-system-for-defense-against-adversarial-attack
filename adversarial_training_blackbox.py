
import matplotlib.pyplot as plt   
import numpy as np
import tensorflow as tf
import time
import copy
import os


# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



k = 4
n = 7
seed = 0


import pickle 
with open('UAP','rb') as load_data:
    UAP = pickle.load(load_data)
    
from classes.Autoencoder_Classes import AE_CNN

# Training the CNN autoencoder
train_EbNodB =  8.5 
val_EbNodB = train_EbNodB


training_params = [
    #batch_size, lr, ebnodb, iterations
    [1000    , 0.001,  train_EbNodB, 500],
    [1000    , 0.0001, train_EbNodB, 5000],
    [10000   , 0.0001, train_EbNodB, 5000],
    [10000   , 0.00001, train_EbNodB, 5000]
]
validation_params = [
    #batch_size, ebnodb, val_steps 
    [100000, val_EbNodB, 50],
    [100000, val_EbNodB, 500],
    [100000, val_EbNodB, 500],
    [100000, val_EbNodB, 500]
]

# Perturbations
p= UAP.reshape(1,2,n)
# Training
ae_CNN = AE_CNN(k,n,seed)
ae_CNN.train(True,0, np.zeros([1,2,n]),training_params, validation_params) # clean inputs
ae_CNN.train(True,0, p,training_params, validation_params) #  inputs with perturbations



PSR_dB= -6
batch_size=10000 # montcarlo
ebnodbs = np.linspace(0,14,15,dtype=int) 

from classes.hamming import hamming_74
num_blocks = 10000

# Performace of BPSK, HD system
BLER_HD, BLER_HD_adv, BLER_HD_jam = hamming_74(n, k, ebnodbs, num_blocks, UAP, PSR_dB)

# Performace of Autoencoder
BLER_no_attack_uap_fgm6, BLER_attack_rolled_uap_fgm6, BLER_jamming1_uap_fgm6 = ae_CNN.bler_sim_attack_AWGN(False, 0, UAP.reshape(1,2,n), PSR_dB, ebnodbs, batch_size, 300)



# plot
fig, ax = plt.subplots(dpi=1200)

ax.plot(ebnodbs, BLER_attack_rolled_uap_fgm6,'r^-',label='Black-Box Attack (Adversarial Training)')
ax.plot(ebnodbs,BLER_HD_adv,'g*--',label='Black-Box Attack (BPSK)')
ax.plot(ebnodbs,BLER_HD,'s--',label='No Attack (BPSK)')
ax.plot(ebnodbs,BLER_no_attack_uap_fgm6,'ko-',label='No Attack (Adversarial Training)')


ax.set_yscale('log')
plt.legend(loc='lower left', fontsize=11)
plt.xticks(ebnodbs,ebnodbs)
ax.set_xlabel('${E_b/N_0}$ (dB)',fontweight='bold')
ax.set_ylabel('Block-error rate (%)',fontweight='bold')
plt.xlim([0, 14]) 
ax.set_xticklabels(ax.get_xticks(),fontsize=12)
ax.grid(True)
plt.rcParams["font.weight"] = "bold"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.show()


# Save results
with open('black_attack_AEad-BPSK', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([BLER_attack_rolled_uap_fgm6, BLER_no_attack_uap_fgm6,BLER_HD_adv,BLER_HD], f)