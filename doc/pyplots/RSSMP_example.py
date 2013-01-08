'''
Created on 9 sept. 2012

@author: manumouss
'''

import os

import numpy as np
import matplotlib.pyplot as plt

from PyMP import signals, mp
from PyMP.mdct import Dico, LODico
from PyMP.mdct.rand import SequenceDico

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.shadow'] = True
mpl.rcParams['image.interpolation'] = 'Nearest'
#mpl.rcParams['text.usetex'] = True

# Load glockenspiel signal
abPath = os.path.abspath('../../data/')
sig = signals.Signal(abPath + '/glocs.wav', mono=True, normalize=True)

sig.crop(0, 3 * sig.fs)

scales = [128, 1024, 8192]
n_atoms = 500
srr = 30

mp_mdct_dico = Dico(scales)
lomp_mdct_dico = LODico(scales)
rssmp_mdct_dico = SequenceDico(scales)

mp_approx, mp_decay = mp.mp(sig, mp_mdct_dico, srr, n_atoms, pad=True)
lomp_approx, lomp_decay = mp.mp(sig, lomp_mdct_dico, srr, n_atoms, pad=True)
rssmp_approx, rssmp_decay = mp.mp(sig, rssmp_mdct_dico, srr, n_atoms, pad=False)


plt.figure()
plt.plot(10 * np.log10(mp_decay) - 10 * np.log10(mp_decay[0]))
plt.plot(10 * np.log10(lomp_decay) - 10 * np.log10(lomp_decay[0]), 'r:')
plt.plot(10 * np.log10(rssmp_decay) - 10 * np.log10(rssmp_decay[0]), 'k--')
plt.ylabel('Normalized reconstruction error (dB)')
plt.xlabel('Iteration')
plt.legend(('Standard MP', 'Locally Optimized MP', 'RSS MP'))
plt.grid()
plt.show()
