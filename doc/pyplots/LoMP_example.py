'''
Created on 9 sept. 2012

@author: manumouss
'''

import numpy as np
import matplotlib.pyplot as plt

from PyMP import signals, approx, mp
from PyMP.mdct import atom
from PyMP.mdct import dico




import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fancybox'] = True;
mpl.rcParams['legend.shadow'] = True;
mpl.rcParams['image.interpolation'] = 'Nearest';
mpl.rcParams['text.usetex'] = True;

# Load glockenspiel signal
abPath = os.path.abspath('../../data/');
pySig = signals.InitFromFile(abPath+'/glocs.wav',forceMono=True,doNormalize=True);

pySig.crop(0,3*pySig.samplingFrequency)

scales = [128,1024,8192];
nbAtom = 500;
srr = 30;

mpDico = dico.Dico(scales)
lompDico = dico.LODico(scales)

approxMP, decayMP = mp.mp(pySig,mpDico,srr,nbAtom,padSignal=True);
approxLoMP, decayLoMP = mp.mp(pySig,lompDico,srr,nbAtom,padSignal=False);

plt.figure()
plt.subplot(211)
approxMP.plotTF()
plt.subplot(212)
approxLoMP.plotTF()

plt.show()