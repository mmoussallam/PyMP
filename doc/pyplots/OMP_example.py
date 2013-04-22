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
dico = Dico([16, 64])
L = 16 * 100
k = 30
data = np.zeros(L,)
# create a k-sparse signal
sp_vec = np.zeros(2*L,)
from PyMP.tools import mdct
random_indexes = np.arange(2*L)
np.random.shuffle(random_indexes)
random_weights = np.random.randn(2*L)
sp_vec[random_indexes[0:k]] = random_weights[0:k]

sparse_data = mdct.imdct(sp_vec[0:L], 16)
sparse_data += mdct.imdct(sp_vec[L:], 64)

#plt.figure()
#plt.subplot(211)
#plt.plot(sparse_data)
#plt.subplot(212)
#plt.plot(sp_vec)
#plt.show()



signal_original = Signal(
    sparse_data, Fs=8000, mono=True, normalize=True)
#signal_original.data += 0.01 * np.random.random(L,)

n_atoms = k + 1

app_1, dec1 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=True, update='mp')
app_2, dec2 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='locgp')
app_3, dec3 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='locomp')

plt.figure()
plt.plot(10.0 * np.log10(dec1 / dec1[0]))
plt.plot(10.0 * np.log10(dec2 / dec2[0]))
plt.plot(10.0 * np.log10(dec3 / dec3[0]))
plt.grid()
plt.legend(('MP', 'LocGP', 'LocOMP'))

plt.figure()
plt.plot(sp_vec, 'o')
plt.plot(app_1.to_array()[0], 'rx')
plt.plot(app_2.to_array()[0], 'ks')
plt.plot(app_3.to_array()[0], 'md')

plt.show()
