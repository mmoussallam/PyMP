'''
Created on Sep 3, 2012

@author: moussall
'''
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.shadow'] = True
mpl.rcParams['image.interpolation'] = 'Nearest'
#mpl.rcParams['text.usetex'] = True

from PyMP import Signal, mp
from PyMP.mdct import Dico


sizes = [128, 1024, 8192]
n_atoms = 1000


abPath = os.path.abspath('../../data/')
sig = Signal(abPath + '/glocs.wav', mono=True, normalize=True)

# taking only the first musical phrase (3.5 seconds approximately)
sig.crop(0, 3.5 * sig.fs)
sig.pad(8192)

# add some minor noise to avoid null areas
sig.data += 0.0001 * np.random.randn(sig.length)

# create MDCT multiscale dictionary
dico = Dico(sizes)

# run the MP routine
approx, decay = mp.mp(sig, dico, 50, n_atoms)

# plotting the results
timeVec = np.arange(0, float(sig.length)) / sig.fs

plt.figure(figsize=(10, 6))
axOrig = plt.axes([0.05, 0.55, .4, .4])
axOrig.plot(timeVec, sig.data)
axOrig.set_title('(a)')
axOrig.set_xticks([1, 2, 3, 4])
axOrig.set_ylim([-1.0, 1.0])

axApprox = plt.axes([0.05, 0.07, .4, .4])
axApprox.plot(timeVec, approx.recomposed_signal.data)
axApprox.set_title('(c)')
axApprox.set_xlabel('Temps (s)')
axApprox.set_xticks([1, 2, 3, 4])
axApprox.set_ylim([-1.0, 1.0])

axApproxTF = plt.axes([0.55, 0.07, .4, .4])
approx.plot_tf(fontsize=16.0, french=False)
axtf = plt.gca()
axtf.set_title('(d)')

axFFt1 = plt.axes([.55, .816, .4, .133])
axFFt1.specgram(
    sig.data, NFFT=128, noverlap=64, cmap=cm.copper_r, Fs=sig.fs)
axFFt1.set_yticks([0, sig.fs / 4])
axFFt1.set_xticks([])
axFFt1.set_title('(b)')

axFFt2 = plt.axes([.55, .683, .4, .133])
axFFt2.specgram(
    sig.data, NFFT=1024, noverlap=512, cmap=cm.copper_r, Fs=sig.fs)
axFFt2.set_yticks([0, sig.fs / 4])
axFFt2.set_ylabel('Frequence (Hz)')
axFFt2.set_xticks([])

axFFt3 = plt.axes([.55, .55, .4, .133])
axFFt3.specgram(sig.data, NFFT=4096, noverlap=0.75 * 4096.0,
                cmap=cm.copper_r, Fs=sig.fs)
axFFt3.set_yticks([0, sig.fs / 4])
axFFt3.set_xticks([1, 2, 3, 4])

try:
    plt.tight_layout()
except Exception, e:
    pass


plt.show()

