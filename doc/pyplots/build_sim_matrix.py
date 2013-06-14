'''
doc.pyplots.build_sim_matrix  -  Created on Jun 14, 2013
@author: M. Moussallam
'''
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
os.environ['PYMP_PATH'] = '/home/manu/workspace/PyMP/'
from PyMP.mdct import Dico, LODico
from PyMP import mp, Signal
from PyMP.signals import LongSignal
from PyMP.mp_coder import joint_coding_distortion

dico = Dico([128,1024,8192])
target_srr = 5
max_atom_num = 200
max_rate = 1000
seg_size = 5*8192
long_signal = LongSignal(op.join(os.environ['PYMP_PATH'],'data/Bach_prelude_40s.wav'),
                         seg_size,
                         mono=True, Noverlap=0.5)

# limit to the first 64 segments
long_signal.n_seg = 32

# decomposing the long signal
apps, decays = mp.mp_long(long_signal,
               dico,
               target_srr, max_atom_num)


mp._initialize_fftw(apps[0].dico, max_thread_num=1)
dists = np.zeros((long_signal.n_seg, len(apps)))    

for idx in range(long_signal.n_seg):
#    print idx
    target_sig = long_signal.get_sub_signal(idx, 1, mono=True, pad=dico.get_pad()+1024,fast_create=True)
    for jdx in range(idx+1):
        # test all preceeding segments only                                                                            
        dists[idx,jdx] = joint_coding_distortion(target_sig, apps[jdx],max_rate,1024, debug=0, precut=15, initfftw=False)  
                                                                                                                
mp._clean_fftw()    


# remove everything that is under zero
cutscores = np.zeros((long_signal.n_seg, len(apps)))
# normalize by reference srr
cutscores[dists>0] = dists[dists>0]/ float(target_srr)

plt.figure()
plt.imshow(cutscores, origin='lower')
plt.colorbar()
plt.xlabel('Target Segment index')
plt.ylabel('Reference Segment index')
plt.show()