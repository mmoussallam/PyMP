'''
doc.pyplots.bach_2_plot  -  Created on Jun 14, 2013
@author: M. Moussallam
'''
import os
import os.path as op
import matplotlib.pyplot as plt
os.environ['PYMP_PATH'] = '/home/manu/workspace/PyMP/'
from PyMP.mdct import Dico, LODico
from PyMP.mdct.rand import SequenceDico
from PyMP import mp, Signal

# Decomposing and visualizing the sparse dec
signal = Signal(op.join(os.environ['PYMP_PATH'],'data/Bach_prelude_4s.wav'), mono=True)

sig_occ1 = signal[:signal.length/2]
sig_occ2 = signal[signal.length/2:]

dico = Dico([128,1024,8192])
target_srr = 5
max_atom_num = 200
app_1, _ = mp.mp(sig_occ1, dico, target_srr, max_atom_num)
app_2, _ = mp.mp(sig_occ2, dico, target_srr, max_atom_num)

plt.figure(figsize=(16,6))
plt.subplot(121)
app_1.plot_tf()
plt.subplot(122)
app_2.plot_tf()
plt.show()