'''
Created on 22 avr. 2013

@author: Manu
'''
import os
import matplotlib.pyplot as plt
import numpy as np
from PyMP import Signal, mp
from PyMP.mdct.dico import Dico, LODico
from PyMP.mdct.atom import Atom

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.shadow'] = True
mpl.rcParams['image.interpolation'] = 'Nearest'
#mpl.rcParams['text.usetex'] = True

print "Running MP, OMP and local versions on synthetic k-sparse"
scales = [16, 64, 256]
dico = Dico(scales)
M = len(scales)
L = 256 * 4
k = 0.2*L

# create a k-sparse signal
sp_vec = np.zeros(M*L,)
from PyMP.tools import mdct
random_indexes = np.arange(M*L)
np.random.shuffle(random_indexes)
random_weights = np.random.randn(M*L)
sp_vec[random_indexes[0:k]] = random_weights[0:k]

sparse_data = np.zeros(L,)
for m in range(M):
    sparse_data += mdct.imdct(sp_vec[m*L:(m+1)*L], scales[m])

#plt.figure()
#plt.subplot(211)
#plt.plot(sparse_data)
#plt.subplot(212)
#plt.plot(sp_vec)
#plt.show()



signal_original = Signal(
    sparse_data, Fs=8000, mono=True, normalize=False)

signal_original.data += 0.01 * np.random.random(signal_original.length,)

n_atoms = k 

signal_original.pad(dico.get_pad())

app_2, dec2 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='locgp')
app_1, dec1 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='mp')
app_3, dec3 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='locomp')

#plt.figure(figsize=(16,6))
#plt.subplot(121)

plt.figure()
plt.plot(10.0 * np.log10(dec1 / dec1[0]))
plt.plot(10.0 * np.log10(dec2 / dec2[0]))
plt.plot(10.0 * np.log10(dec3 / dec3[0]))
plt.grid()
plt.ylabel('Residual Energy decay (dB)')
plt.xlabel('Iteration')
plt.legend(('MP', 'LocGP', 'LocOMP'))



#mp_sp_vec = app_1.to_array()[0]
#locgp_sp_vec = app_2.to_array()[0]
#locomp_sp_vec = app_3.to_array()[0]
#
#true_index = sp_vec.nonzero()[0]
#mp_index = mp_sp_vec.nonzero()[0]
#locgp_index = locgp_sp_vec.nonzero()[0]
#locomp_index = locomp_sp_vec.nonzero()[0]
#
#pad = dico.get_pad()
#N = mp_sp_vec.shape[0]/M
#
#depad_idx = lambda x:x - (2*x/N + 1)*pad 
#pad_idx = lambda x:x + ((2*x/L) + 1)*pad
#
#plt.subplot(122)
##plt.figure()
#plt.stem(map(pad_idx,true_index), sp_vec[true_index],'-bo')
#plt.stem(mp_index, mp_sp_vec[mp_index], '--rx')
#plt.stem( locgp_index, locgp_sp_vec[locgp_index], '-.ks')
#plt.stem( locomp_index, locomp_sp_vec[locomp_index], ':md')
#plt.legend(('Truth', 'MP', 'LocGP', 'LocOMP'))
#plt.xlabel('Atom index')

plt.show()
