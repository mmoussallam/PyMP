"""

Tutorial provided as part of PyMP

M. Moussallam

"""

from math import floor 
from mdct import dico
from mdct.random import dico as random_dico
import mp , mp_coder, signals
import matplotlib.pyplot as plt
myPympSignal =  signals.InitFromFile('data/ClocheB.wav',forceMono=True) # Load Signal
myPympSignal.crop(0, 4.0*myPympSignal.samplingFrequency)     # Keep only 4 seconds

# atom of scales 8, 64 and 512 ms
scales = [(s * myPympSignal.samplingFrequency / 1000) for s in (8,64,512)] 

myPympSignal.pad(scales[-1])

# Dictionary for Standard MP
pyDico = dico.Dico(scales);                

# Launching decomposition, stops either at 20 dB of SRR or 2000 iterations
mpApprox , mpDecay = mp.mp(myPympSignal, pyDico, 20, 2000,padSignal=False);  

#mpApprox.atomNumber
SNR, bitrate, quantizedApprox = mp_coder.SimpleMDCTEncoding(mpApprox, 8000, Q=14);
print (SNR, bitrate)

print "With Q=5"
SNR, bitrate, quantizedApprox = mp_coder.SimpleMDCTEncoding(mpApprox, 8000, Q=5);
print (SNR, bitrate)


SNR, bitrate, quantizedApprox = mp_coder.SimpleMDCTEncoding(mpApprox, 2000, Q=14);
print (SNR, bitrate)

pyLODico = dico.LODico(scales);
lompApprox , lompDecay = mp.mp(myPympSignal, pyLODico, 20, 2000,padSignal=False);  
SNRlo, bitratelo, quantizedApproxLO = mp_coder.SimpleMDCTEncoding(lompApprox, 2000, Q=14, TsPenalty=True);
print (SNRlo, bitratelo)

pyRSSDico = random_dico.RandomDico(scales);
rssApprox , rssDecay = mp.mp(myPympSignal, pyRSSDico, 20, 2000,padSignal=False);  
SNRrss, bitraterss, quantizedApproxRSS = mp_coder.SimpleMDCTEncoding(rssApprox, 2000, Q=14);
print (SNRrss, bitraterss)

print (quantizedApprox.atomNumber,  quantizedApproxLO.atomNumber , quantizedApproxRSS.atomNumber)

#quantizedApprox.plotTF()
#plt.show()


print " now at a much larger level : a SNR of nearly 50 dB and around 64 Kbps"
mpApprox , mpDecay = MP.MP(myPympSignal, pyDico, 50, 20000,padSignal=False); 
SNR, bitrate, quantizedApprox = MPCoder.SimpleMDCTEncoding(mpApprox, 64000, Q=16);
print (SNR, bitrate)

lompApprox , lompDecay = MP.MP(myPympSignal, pyLODico, 50, 20000,padSignal=False);  
SNRlo, bitratelo, quantizedApproxLO = MPCoder.SimpleMDCTEncoding(lompApprox, 64000, Q=16, TsPenalty=True);
print (SNRlo, bitratelo)

rssApprox , rssDecay = MP.MP(myPympSignal, pyRSSDico, 50, 20000,padSignal=False);  
SNRrss, bitraterss, quantizedApproxRSS = MPCoder.SimpleMDCTEncoding(rssApprox, 64000, Q=16);
print (SNRrss, bitraterss)
