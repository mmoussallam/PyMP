
import matplotlib.pyplot as plt
import os

from PyMP import Signal, mp, mp_coder
from PyMP.mdct import Dico


abPath = os.path.abspath('../../data/')
sig = Signal(
    abPath + '/ClocheB.wav', mono=True)  # Load Signal
sig.crop(0, 4.0 * sig.fs)     # Keep only 4 seconds

# atom of scales 8, 64 and 512 ms
scales = [(s * sig.fs / 1000) for s in (8, 64, 512)]

# Dictionary for Standard MP
pyDico = Dico(scales)

# Launching decomposition, stops either at 20 dB of SRR or 2000 iterations
mpApprox, mpDecay = mp.mp(sig, pyDico, 20, 2000)

# mpApprox.atomNumber

SNR, bitrate, quantizedApprox = mp_coder.simple_mdct_encoding(
    mpApprox, 2000, Q=14)

quantizedApprox.plot_tf()
plt.show()
