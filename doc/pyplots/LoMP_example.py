'''
Created on 9 sept. 2012

@author: manumouss
'''

import os
import MP
from Classes import pymp_Signal, pymp_Approx;
from Classes.mdct import *
import numpy as np
import matplotlib.pyplot as plt



import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True;
mpl.rcParams['legend.shadow'] = True;
mpl.rcParams['image.interpolation'] = 'Nearest';
mpl.rcParams['text.usetex'] = True;

# Load glockenspiel signal
abPath = os.path.abspath('../../data/');
pySig = pymp_Signal.InitFromFile(abPath+'/glocs.wav',forceMono=True,doNormalize=True);

pySig.crop(0,3*pySig.samplingFrequency)

scales = [128,1024,8192];
nbAtom = 500;
srr = 30;

mpDico = pymp_MDCTDico.pymp_MDCTDico(scales)
lompDico = pymp_MDCTDico.pymp_LODico(scales)

approxMP, decayMP = MP.MP(pySig,mpDico,srr,nbAtom,padSignal=True);
approxLoMP, decayLoMP = MP.MP(pySig,lompDico,srr,nbAtom,padSignal=False);

plt.figure()
plt.subplot(211)
approxMP.plotTF()
plt.subplot(212)
approxLoMP.plotTF()

plt.show()