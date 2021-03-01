

import matplotlib.pyplot as plt   
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
    
from classes.Autoencoder_Classes import AE_MLP

# Training the MLP autoencoder
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
ae_MLP = AE_MLP(k,n,seed)
ae_MLP.train(np.zeros([1,2,n]),training_params, validation_params) # clean inputs
ae_MLP.train(p,training_params, validation_params) #  inputs with perturbations




PSR_dB = -6
num_samples =  10
ebnodb = 0
universal_per_fgm_MLP = ae_MLP.UAPattack_fgm(ebnodb,num_samples,PSR_dB)


ebnodbs = np.linspace(0,14,15,dtype=int) 
batch_size = 10000

from classes.hamming import hamming_74
num_blocks = 10000

# Performace of BPSK, HD system
BLER_HD, BLER_HD_adv, BLER_HD_jam = hamming_74(n, k, ebnodbs, num_blocks, universal_per_fgm_MLP.reshape(2*n), PSR_dB)

# Performace of Autoencoder
BLER_No_Attack, BLER_Adv_Attack_minus6, BLER_Jamming_Attack_minus6 = ae_MLP.bler_sim_attack_AWGN(universal_per_fgm_MLP.reshape(1,2,n), PSR_dB, ebnodbs, batch_size, 300) 





fig, ax = plt.subplots(dpi=1200)

ax.plot(ebnodbs,BLER_Adv_Attack_minus6,'r^-',label='White-Box Attack (Adversarial Training)')
ax.plot(ebnodbs,BLER_HD_adv,'g*--',label='White-Box Attack (BPSK, Hamming)')
ax.plot(ebnodbs,BLER_HD,'s--',label='No Attack (BPSK, Hamming)')
ax.plot(ebnodbs,BLER_No_Attack,'ko-',label='No Attack (Adversarial Training)')


ax.set_yscale('log')
plt.legend(loc='lower left', fontsize=11)
plt.xticks(ebnodbs,ebnodbs)
plt.rcParams["font.weight"] = "bold"
ax.set_xlabel('${E_b/N_0}$ (dB)',fontweight='bold')
ax.set_ylabel('Block-error rate (%)',fontweight='bold')
plt.xlim([0, 14]) 
plt.ylim([10e-8, 0.5]) 
ax.set_xticklabels(ax.get_xticks())
ax.grid(True)
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

plt.show()

with open('white_attack_AEad-BPSK', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([BLER_Adv_Attack_minus6, BLER_No_Attack,BLER_HD_adv,BLER_HD], f)




