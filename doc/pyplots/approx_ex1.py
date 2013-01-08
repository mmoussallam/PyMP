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
#mpl.rcParams['text.usetex'] = True

from PyMP import approx
from PyMP.mdct import atom


short_atom = atom.Atom(32, 1, 1024, 4, 8000, 1)
mid_atom = atom.Atom(256, 1, 256, 10, 8000, 1)
long_atom = atom.Atom(2048, 1, 0, 40, 8000, 1)

short_atom.synthesize()
mid_atom.synthesize()
long_atom.synthesize()

# Initialize empty approximant
approx = approx.Approx(
    None, [], None, long_atom.length, long_atom.fs)

print approx

# add atoms
approx.add(long_atom)
print approx
approx.add(short_atom)
approx.add(mid_atom)


timeVec = np.arange(approx.length) / float(approx.fs)

# Plot the recomposed Signal, both in time domain and Time-Frequency
plt.figure()
plt.subplot(211)
plt.plot(timeVec, approx.recomposed_signal.data)
plt.xlim([0, float(approx.length) / float(approx.fs)])
plt.grid()
plt.subplot(212)
approx.plot_tf(french=False, patchColor=(0, 0, 0))
plt.grid()

# plt.savefig(figurePath+'Example_3atoms_mdct.pdf',dpi=300)
plt.show()
