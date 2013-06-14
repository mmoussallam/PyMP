"""

Tutorial provided as part of PyMP
M. Moussallam

"""
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
os.environ['PYMP_PATH'] = '/home/manu/workspace/PyMP/'
from PyMP.mdct import Dico
from PyMP import mp, Signal
from PyMP.tools.Misc import euclid_dist, hamming_dist

# Decomposing and visualizing the sparse dec
signal = Signal(op.join(os.environ['PYMP_PATH'],'data/Bach_prelude_4s.wav'), mono=True)

sig_occ1 = signal[:signal.length/2]
sig_occ2 = signal[signal.length/2:]

dico = Dico([128,1024,8192])
target_srr = 5
max_atom_num = 200
app_1, _ = mp.mp(sig_occ1, dico, target_srr, max_atom_num)
app_2, _ = mp.mp(sig_occ2, dico, target_srr, max_atom_num)

#plt.figure(figsize=(16,6))
#plt.subplot(121)
#app_1.plot_tf()
#plt.subplot(122)
#app_2.plot_tf()
#plt.show()


sp_vec_1 = app_1.to_array()[0]
sp_vec_2 = app_2.to_array()[0]

print "%1.5f, %1.5f"%(euclid_dist(sp_vec_1,sp_vec_2), hamming_dist(sp_vec_1,sp_vec_2)) 

# Now the information distance
from PyMP.mp_coder import joint_coding_distortion

# Measure the distortion of joint coding using approx of first patter as the reference
max_rate = 1000 # maximum bitrate allowed (in bits)
search_width = 1024 # maximum time shift allowed in samples
info_dist = joint_coding_distortion(sig_occ2, app_1, max_rate, search_width)
info_dist_rev = joint_coding_distortion(sig_occ1, app_2, max_rate, search_width)

print "%1.5f  - %1.5f"%(info_dist/target_srr, info_dist_rev/target_srr)


# building the similarity matrix
# Now load the long version
from PyMP.signals import LongSignal
seg_size = 5*8192
long_signal = LongSignal(op.join(os.environ['PYMP_PATH'],'data/Bach_prelude_40s.wav'),
                         seg_size,
                         mono=True, Noverlap=0.5)

# decomposing the long signal
apps, decays = mp.mp_long(long_signal,
               dico,
               target_srr, max_atom_num)




dists = np.zeros((long_signal.n_seg, len(apps)))    
mp._initialize_fftw(apps[0].dico, max_thread_num=1)
for idx in range(long_signal.n_seg):
    for jdx in range(idx):
        # test all preceeding segments only
        target_sig = long_signal.get_sub_signal(idx, 1, mono=True, pad=dico.get_pad(),fast_create=True)                                                                        
        dists[idx,jdx] = joint_coding_distortion(target_sig, apps[jdx],max_rate,1024, initfftw=False)  
                                                                                                                   
                                                                                                                   
mp._clean_fftw()    

plt.figure()
plt.imshow(dists)
plt.colorbar()
plt.show()
