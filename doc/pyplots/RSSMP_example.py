'''
Created on 9 sept. 2012

@author: manumouss
'''

import os

import numpy as np
import matplotlib.pyplot as plt

from PyMP import signals, mp
from PyMP.mdct import dico
from PyMP.mdct.rand import dico as random_dico

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.shadow'] = True
mpl.rcParams['image.interpolation'] = 'Nearest'
#mpl.rcParams['text.usetex'] = True

# Load glockenspiel signal
abPath = os.path.abspath('../../data/')
pySig = signals.Signal(abPath + '/glocs.wav', mono=True, normalize=True)

pySig.crop(0, 3 * pySig.fs)

scales = [128, 1024, 8192]
nbAtom = 500
srr = 30

mpDico = dico.Dico(scales)
lompDico = dico.LODico(scales)
rssMPDico = random_dico.SequenceDico(scales)

approxMP, decayMP = mp.mp(pySig, mpDico, srr, nbAtom, pad=True)
approxLoMP, decayLoMP = mp.mp(pySig, lompDico, srr, nbAtom, pad=True)
approxRSSMP, decayRSSMP = mp.mp(pySig, rssMPDico, srr, nbAtom, pad=False)


plt.figure()
plt.plot(10 * np.log10(decayMP) - 10 * np.log10(decayMP[0]))
plt.plot(10 * np.log10(decayLoMP) - 10 * np.log10(decayLoMP[0]), 'r:')
plt.plot(10 * np.log10(decayRSSMP) - 10 * np.log10(decayRSSMP[0]), 'k--')
plt.ylabel('Normalized reconstruction error (dB)')
plt.xlabel('Iteration')
plt.legend(('Standard MP', 'Locally Optimized MP', 'RSS MP'))
plt.grid()
plt.show()
