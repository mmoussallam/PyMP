'''
PyMP.separation.Separate  -  Created on Jun 6, 2013
Audio Source Separation based on redundancy detection and joint sparse decomposition
@author: M. Moussallam
'''
from PyMP import mp, signals
from PyMP.mdct import LODico
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#from scipy.sparse import  bmat , speye
##from Similarity import py_similarity, py_factorizer
import math
from scipy.io import savemat, loadmat
import os.path as op

# interface: song.wav -> similarity matrix -> redundancy map -> separated sources
# song.wav -> build_sim_matrix() -> similarity matrix

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
    
    return apps, decays

    
######### TESTING ##########
filePath = op.join(os.environ['PyMP_PATH'],'data')
fileName =  "Bach_prelude_40s.wav"

apps, decays = build_sim_matrix(op.join(filePath,fileName))