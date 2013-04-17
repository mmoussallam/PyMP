'''
Created on Mar 20, 2013

@author: manu
'''

import numpy as np
from math import  floor

from .. import  log
#from ..tools import Xcorr  # , Misc
from ..base import BaseBlock
from .atom import WaveAtom
import pywt
_Logger = log.Log('Wavelet Block')
class WaveletBlock(BaseBlock):
    '''A wavelet block implements a Multilevel Wavelet Transform.
    
    attributes
    ----------
    nature : str
        type of wavelet (e.g. 'haar', 'db8', ...) see pywavelets doc
    level : int
        minimum level of precision. If n is the size of the signal
        then decomposition from scale 2**log(n)-1 to 2**level is performed
    '''

    def __init__(self, wav, mode ='per', debug_level=0):
        '''
        Constructor
        '''
        _Logger.set_level(debug_level)
        self.nature = wav[0]
        self.level = int(wav[1])
        self.mode = mode
        # for the projections we keep the list structure of pywavelets
        self.projections = None
        _Logger.info("Created new wavelet block : %s - %d levels "%(self.nature, self.level))
        self.max_value = None
        self.max_index = None
        self.max_level = None
        self.residual_signal = None
        self.scales = None
        
    def __repr__(self):
        
        mv_str = str(self.max_value) if self.max_value is not None else 'Not Set'
        mi_str = str(self.max_index) if self.max_index is not None else '-'  
        ml_str = str(self.max_level) if self.max_level is not None else '-'  
        return """ 
Block for %s wavelet transform with %d levels 
    Max Value : %s at level %s position %s"""%(self.nature, self.level,
                                               mv_str, ml_str, mi_str)
    
    def update(self, new_res_signal):
        ''' recompute the projections using the fast cascading wavelet transforms'''
        self.residual_signal = new_res_signal                

        self.compute_transform()

        self.find_max()
    
    def initialize(self, residual_signal):
        ''' initialize some stuff @TODO check signal length is power of two'''
        self.residual_signal = residual_signal
        Jmax = int(floor(np.log2(self.residual_signal.length)))
        
        self.scales = [2**j for j in range( Jmax-self.level, Jmax)]
        _Logger.info("Initialized with scales : %s"%str(self.scales))
    
    def compute_transform(self):
        ''' use pywavelets implementation'''
        if self.scales is None:
            self.initialize(self.residual_signal)
        
        
        self.projections = pywt.wavedec(self.residual_signal.data,
                                        self.nature,
                                        self.mode,
                                        self.level)
    
    def find_max(self):
        ''' find the level, index and value of the biggest coefficient '''
        if self.projections is None:
            raise ValueError("Projections have not been computed")
        self.max_value = 0
        for l_proj in range(len(self.projections)):
            b_s_level = np.max(np.abs(self.projections[l_proj]))
            if b_s_level > self.max_value:
                self.max_value = b_s_level
                self.max_level = l_proj
                self.max_index = np.argmax(np.abs(self.projections[l_proj]))
    
        
        _Logger.info("Max element found in level %d at pos %d for value %1.5f"%(self.max_level,
                                                                                self.max_index,
                                                                                self.max_value))
        
    def get_max_atom(self):
        """ retrieve the best atom """        

        # First evaluate the time position given the scale and the index
        if self.max_level == 0:
            resolution = self.residual_signal.length  / self.scales[0]
            wavelet_depth = len(self.scales)
            part = 'a'
            
        else:
            resolution = self.residual_signal.length  / self.scales[self.max_level-1]
            wavelet_depth = len(self.scales)-(self.max_level-1)
            part ='d'
                
        best_atom = WaveAtom(scale=0,
                             amp=self.max_value,
                             timePos=0,
                             Fs=self.residual_signal.fs,
                             nature=self.nature,
                             level=wavelet_depth)
        
        # We use the pywt library to build a complete signal that corresponds to the atom
        # We have to do this since if not padding is performed and a cyclic analysis
        # then atoms can be spread between beginning and end
        # plus we need to control whether the selected coefficient is an approximation
        # or detail one
        coeffs = []
        for l in range(len(self.projections)):
            coeffs.append( np.zeros_like(self.projections[l]))
            if l == self.max_level:
                print len(self.projections), self.max_level,self.projections[self.max_level].shape, self.max_index        
                coeffs[-1][self.max_index] = self.projections[self.max_level][self.max_index]
        
        best_atom.waveform = pywt.waverec(coeffs, self.nature, mode='per')
        
        best_atom.length = best_atom.waveform.shape[0]
        
#        best_atom.synthesize()
#        N = best_atom.waveform.shape[0]
#        best_atom.length = N
#
#        if N % 2 == 1:
#            best_atom.time_position = (self.max_index * resolution) - (N-1)/4 
#        else:
#            best_atom.time_position = (self.max_index * resolution) - N/4 
        _Logger.debug("Best atom selected : %s"%str(best_atom))
        return best_atom