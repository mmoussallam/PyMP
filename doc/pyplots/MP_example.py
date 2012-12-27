'''
Created on Sep 3, 2012

@author: moussall
'''

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True;
mpl.rcParams['legend.shadow'] = True;
mpl.rcParams['image.interpolation'] = 'Nearest';
mpl.rcParams['text.usetex'] = True;

import sys
import os
import MP
from Classes import pymp_Signal, pymp_Approx;
from Classes.mdct import *
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from math import pi

sizes = [128,1024,8192];
Natom = 1000;


abPath = os.path.abspath('../../data/');
pySig = pymp_Signal.InitFromFile(abPath+'/glocs.wav',forceMono=True,doNormalize=True);

# taking only the first musical phrase (3.5 seconds approximately)
pySig.crop(0, 3.5*pySig.samplingFrequency);
pySig.pad(8192);

# add some minor noise to avoid null areas 
pySig.dataVec += 0.0001*np.random.randn(pySig.length);

# create MDCT multiscale dictionary 
dico= pymp_MDCTDico.pymp_MDCTDico(sizes);

# run the MP routine
approx, decay = MP.MP(pySig, dico, 50, Natom);

# plotting the results
timeVec = np.arange(0,float(pySig.length))/pySig.samplingFrequency;

plt.figure(figsize=(10,6))
axOrig = plt.axes([0.05, 0.55, .4, .4])
axOrig.plot(timeVec, pySig.dataVec)
axOrig.set_title('(a)')
axOrig.set_xticks([1,2,3,4])
axOrig.set_ylim([-1.0, 1.0])

axApprox = plt.axes([0.05, 0.07, .4, .4])
axApprox.plot(timeVec, approx.recomposedSignal.dataVec)
axApprox.set_title('(c)')
axApprox.set_xlabel('Temps (s)')
axApprox.set_xticks([1,2,3,4])
axApprox.set_ylim([-1.0, 1.0])

axApproxTF = plt.axes([0.55, 0.07, .4, .4])
approx.plotTF(fontsize=16.0,french=False)
axtf = plt.gca()
axtf.set_title('(d)')

axFFt1 = plt.axes([.55, .816, .4, .133])
axFFt1.specgram(pySig.dataVec ,  NFFT=128,noverlap=64,cmap=cm.copper_r,Fs=pySig.samplingFrequency)
axFFt1.set_yticks([0,pySig.samplingFrequency/4])
axFFt1.set_xticks([])
axFFt1.set_title('(b)')

axFFt2 = plt.axes([.55, .683, .4, .133])
axFFt2.specgram(pySig.dataVec ,  NFFT=1024,noverlap=512,cmap=cm.copper_r,Fs=pySig.samplingFrequency)
axFFt2.set_yticks([0,pySig.samplingFrequency/4])
axFFt2.set_ylabel('Frequence (Hz)')
axFFt2.set_xticks([])

axFFt3 = plt.axes([.55, .55, .4, .133])
axFFt3.specgram(pySig.dataVec ,  NFFT=4096,noverlap=0.75*4096.0,cmap=cm.copper_r,Fs=pySig.samplingFrequency)
axFFt3.set_yticks([0,pySig.samplingFrequency/4])
axFFt3.set_xticks([1,2,3,4])


plt.show()