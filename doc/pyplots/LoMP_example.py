'''
Created on 9 sept. 2012

@author: manumouss
'''
import os
import matplotlib.pyplot as plt

from PyMP import Signal, mp
from PyMP.mdct import Dico, LODico

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.shadow'] = True
mpl.rcParams['image.interpolation'] = 'Nearest'
#mpl.rcParams['text.usetex'] = True

# Load glockenspiel signal
abPath = os.path.abspath('../../data/')
sig = Signal(abPath + '/glocs.wav', mono=True, normalize=True)

sig.crop(0, 3 * sig.fs)

scales = [128, 1024, 8192]
n_atoms = 500
srr = 30

mp_dico = Dico(scales)
lomp_dico = LODico(scales)

mp_approx, mp_decay = mp.mp(sig, mp_dico, srr, n_atoms, pad=True)
lomp_approx, lomp_decay = mp.mp(sig, lomp_dico, srr, n_atoms, pad=False)

plt.figure()
plt.subplot(211)
mp_approx.plot_tf()
plt.subplot(212)
lomp_approx.plot_tf()

# print mp_approx , lomp_approx

plt.show()
