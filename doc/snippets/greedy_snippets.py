import numpy as np
from PyMP import Signal, mp
from PyMP.mdct.dico import Dico, LODico
from PyMP.mdct.atom import Atom
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
signal_original = Signal(sparse_data, Fs=8000, mono=True, normalize=False)
signal_original.data += 0.01 * np.random.random(signal_original.length,)
n_atoms = k 
signal_original.pad(dico.get_pad())
app_2, dec2 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='locgp')
app_1, dec1 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='mp')
app_3, dec3 = mp.greedy(signal_original, dico, 100,
                        n_atoms, debug=0, pad=False, update='locomp')