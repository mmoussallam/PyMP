'''
Created on 4 sept. 2012

@author: M. Moussallam
'''
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.shadow'] = True
mpl.rcParams['image.interpolation'] = 'Nearest'
mpl.rcParams['text.usetex'] = True

from PyMP import approx
from PyMP.mdct import atom


atomShort = atom.Atom(32, 1, 1024, 4, 8000, 1)
atomMid = atom.Atom(256, 1, 256, 10, 8000, 1)
atomLong = atom.Atom(2048, 1, 0, 40, 8000, 1)

atomShort.synthesize()
atomMid.synthesize()
atomLong.synthesize()

# Initialize empty approximant
approx = approx.Approx(
    None, [], None, atomLong.length, atomLong.samplingFrequency)

# add atoms
approx.addAtom(atomShort)
approx.addAtom(atomMid)
approx.addAtom(atomLong)

timeVec = np.arange(approx.length) / float(approx.samplingFrequency)

# Plot the recomposed Signal, both in time domain and Time-Frequency
plt.figure()
plt.subplot(211)
plt.plot(timeVec, approx.recomposedSignal.data)
plt.xlim([0, float(approx.length) / float(approx.samplingFrequency)])
plt.grid()
plt.subplot(212)
approx.plotTF(french=False, patchColor=(0, 0, 0))
plt.grid()

# plt.savefig(figurePath+'Example_3atoms_mdct.pdf',dpi=300)
plt.show()
