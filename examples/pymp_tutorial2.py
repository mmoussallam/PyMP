"""

Tutorial provided as part of PyMP

M. Moussallam

"""

from PyMP.mdct import dico
from PyMP.mdct.random import dico as random_dico
from PyMP import mp, mp_coder, signals
myPympSignal = signals.Signal('../data/ClocheB.wav', mono=True)  # Load Signal
myPympSignal.crop(0, 4.0 * myPympSignal.fs)     # Keep only 4 seconds
# atom of scales 8, 64 and 512 ms
scales = [(s * myPympSignal.fs / 1000) for s in (8, 64, 512)]
myPympSignal.pad(scales[-1])
# Dictionary for Standard MP
pyDico = dico.Dico(scales)
# Launching decomposition, stops either at 20 dB of SRR or 2000 iterations
mpApprox, mpDecay = mp.mp(myPympSignal, pyDico, 20, 2000, pad=False)

mpApprox.atomNumber

SNR, bitrate, quantizedApprox = mp_coder.simple_mdct_encoding(
    mpApprox, 8000, Q=14)
print (SNR, bitrate)

print "With Q=5"
SNR, bitrate, quantizedApprox = mp_coder.simple_mdct_encoding(
    mpApprox, 8000, Q=5)
print (SNR, bitrate)


SNR, bitrate, quantizedApprox = mp_coder.simple_mdct_encoding(
    mpApprox, 2000, Q=14)
print (SNR, bitrate)

pyLODico = dico.LODico(scales)
lompApprox, lompDecay = mp.mp(
    myPympSignal, pyLODico, 20, 2000, pad=False)
SNRlo, bitratelo, quantizedApproxLO = mp_coder.simple_mdct_encoding(
    lompApprox, 2000, Q=14, TsPenalty=True)
print (SNRlo, bitratelo)

pyRSSDico = random_dico.RandomDico(scales)
rssApprox, rssDecay = mp.mp(myPympSignal, pyRSSDico, 20, 2000, pad=False)
SNRrss, bitraterss, quantizedApproxRSS = mp_coder.simple_mdct_encoding(
    rssApprox, 2000, Q=14)
print (SNRrss, bitraterss)

print (quantizedApprox.atom_number, quantizedApproxLO.atom_number,
       quantizedApproxRSS.atom_number)

# quantizedApprox.plotTF()
# plt.show()


print " now at a much larger level : a SNR of nearly 50 dB and around 64 Kbps"
mpApprox, mpDecay = mp.mp(myPympSignal, pyDico, 50, 20000, pad=False)
SNR, bitrate, quantizedApprox = mp_coder.simple_mdct_encoding(
    mpApprox, 64000, Q=16)
print (SNR, bitrate)

lompApprox, lompDecay = mp.mp(
    myPympSignal, pyLODico, 50, 20000, pad=False)
SNRlo, bitratelo, quantizedApproxLO = mp_coder.simple_mdct_encoding(
    lompApprox, 64000, Q=16, TsPenalty=True)
print (SNRlo, bitratelo)

rssApprox, rssDecay = mp.mp(
    myPympSignal, pyRSSDico, 50, 20000, pad=False)
SNRrss, bitraterss, quantizedApproxRSS = mp_coder.simple_mdct_encoding(
    rssApprox, 64000, Q=16)
print (SNRrss, bitraterss)
