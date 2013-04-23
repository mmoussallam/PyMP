'''
doc.pyplots.Spectro_example  -  Created on Apr 23, 2013
@author: M. Moussallam
'''
import os.path as op
from PyMP import Signal, approx
import matplotlib.pyplot as plt

abPath = op.abspath('../../data/')
sig = Signal(op.join(abPath, 'glocs.wav'), normalize=True, mono=True)


import matplotlib.cm as cm
plt.figure()
sig.spectrogram(1024, 128, order=2, log=True, cmap=cm.hot, cbar=True)

plt.show()