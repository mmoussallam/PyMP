from math import floor 
from Classes import *
from Classes.mdct import pymp_MDCTDico
import MP , MPCoder
import matplotlib.pyplot as plt
import os

abPath = os.path.abspath('../../data/');
myPympSignal =  pymp_Signal.InitFromFile(abPath+'/ClocheB.wav',forceMono=True) # Load Signal
myPympSignal.crop(0, 4.0*myPympSignal.samplingFrequency)     # Keep only 4 seconds

# atom of scales 8, 64 and 512 ms
scales = [(s * myPympSignal.samplingFrequency / 1000) for s in (8,64,512)] 

# Dictionary for Standard MP
pyDico = pymp_MDCTDico.pymp_MDCTDico(scales);                

# Launching decomposition, stops either at 20 dB of SRR or 2000 iterations
mpApprox , mpDecay = MP.MP(myPympSignal, pyDico, 20, 2000);  

#mpApprox.atomNumber

SNR, bitrate, quantizedApprox = MPCoder.SimpleMDCTEncoding(mpApprox, 2000, Q=14);

quantizedApprox.plotTF()
plt.show()
