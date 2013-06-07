'''
PyMP.similarity.similarity  -  Created on Jun 7, 2013
@author: M. Moussallam
'''

import matplotlib.pyplot as plt
import math
import numpy as np
import os.path as op
from PyMP import signals, approx, mp
from PyMP.mdct import LODico
from PyMP.tools import Xcorr
from numpy.fft import fft
from joblib import Parallel, delayed

def build_sim_matrix(song_path, frame_size=4*16384,
                     target_srr=5, max_it_num=100, debug=0, pad=True):
    """ Get a song, decompose it on overlapping segment to build 
        A similarity matrix """
    orig_longsignal = signals.LongSignal(song_path,
                                         frame_size,
                                         mono=True, Noverlap=0.5)
    
    dictionary = LODico([128,1024,8192])
    
    apps, decays = mp.mp_long(orig_longsignal,
               dictionary,
               target_srr, max_it_num, debug, pad, write=False)
    
    return apps, decays, orig_longsignal

def jointcodingdistortion(l_target , reference, idx, max_rate ,
                          halfSearchArea , threshold=0 , doSubtract=True, 
                          allowDiffAmp =False , debug=0 , discard = False,
                          precut=-1):
        """ compute the joint coding distortion e.g. given a rate what is the distorsion achieved
        if coding the target knowing the reference"""
        target = l_target.get_sub_signal(idx, 1, 
                                        mono=True,
                                        pad=8192)
        # initialize factored approx
        factorizedApprox = approx.Approx(reference.dico, 
                                           [], 
                                           target, 
                                           reference.length, 
                                           reference.fs)        
        timeShifts = np.zeros(reference.atom_number)
        atom_idx = 0;
        rate = 0    
        residual = target.data.copy()
        startSRR = factorizedApprox.compute_srr()
        if debug > 0:
            print "starting factorization of " , reference.atom_number , " atoms"        
        while (rate < max_rate) and (atom_idx < reference.atom_number):
            # Stop only when the target rate is achieved or all atoms have been used
            
            atom = reference[atom_idx]
            # make a copy of the atom
            newAtom = atom.copy()
                
            # find max correlation and remember the TimeShift
            sigFft = fft(residual[newAtom.time_position - halfSearchArea : newAtom.time_position + newAtom.length + halfSearchArea] , atom.length + 2*halfSearchArea)
            atomFft = fft(np.concatenate( (np.concatenate((np.zeros(halfSearchArea) , newAtom.waveform) ) , np.zeros(halfSearchArea) ) ) ,atom.length + 2*halfSearchArea)
    
            # compute xcorr through fft
            newts , score = Xcorr.GetMaxXCorr(atomFft , sigFft ,
                                              maxlag =halfSearchArea , debug =debug) 
            if debug>0:
                print "Score of " , score , " found"
            # handle MP atoms #
            if newAtom.time_shift is not None:
                newAtom.time_shift += newts
                newAtom.time_position += newts
            else:
                newAtom.time_shift = newts
                newAtom.time_position += newts
                atom.proj_score = atom.mdct_value
                            
            if debug>0:
                print "Factorizing with new offset: " , newts                

            if allowDiffAmp:
                print "WARNING : THIS IS NOT WORKING"
                newAtom.proj_score = score;
                newAtom.waveform *= (score / np.sqrt(np.sum(newAtom.waveform**2))) 
                rate += np.log2(abs(score))
            
            if score <0:                    
                newAtom.waveform = -newAtom.waveform

            factorizedApprox.add(newAtom)
            if debug > 0:
                print "SRR Achieved of : " , factorizedApprox.compute_srr()            
            timeShifts[atom_idx] = newAtom.time_shift
                
            if doSubtract:                
                residual[newAtom.time_position : newAtom.time_position + newAtom.length ] -= newAtom.waveform
                
            rate += np.log2(abs(newts))+1
            if debug:
                print "Atom %d - rate of %1.3f"%(atom_idx, rate)
            atom_idx +=1
            
            if atom_idx>precut and precut>0:
                curdisto = factorizedApprox.compute_srr() 
                if curdisto<0:
                    return curdisto
                    

        # calculate achieved SNR :
        return factorizedApprox.compute_srr()
    
############## TESTING #############
import os
os.environ['PyMP_PATH'] = '/home/manu/workspace/PyMP'
filePath = op.join(os.environ['PyMP_PATH'],'data')
fileName =  "Bach_prelude_40s.wav"

apps, decays, long_sigs = build_sim_matrix(op.join(filePath,fileName), target_srr=5)
# now try to evaluate the first segment compared to the rest

#jointcodingdistortion(long_sigs, apps[1], 10 , 300, 1024, debug=1)


import time

for max_rate in [50,100,300]:
    dists = np.zeros((long_sigs.n_seg, len(apps)))    
    t = time.time()
    for idx in range(long_sigs.n_seg):
        for jdx in range(len(apps)):
            dists[idx,jdx] = jointcodingdistortion(long_sigs,
                                           apps[jdx],
                                           idx, max_rate,
                                           1024)    
    print time.time() - t    
    plt.figure()
    plt.imshow(dists>0)
    plt.title("%d"%max_rate)
        
plt.show()

#dist = jointcodingdistortion(long_sigs.get_sub_signal(0, 1, mono=True,pad=8192) ,
#                              apps[1] , 300,
#                              1024 , threshold=0 , doSubtract=True, 
#                              allowDiffAmp =False , debug=0 , discard = False)


