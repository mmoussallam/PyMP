'''
Created on Mar 20, 2013

@author: manu
'''
import math
import numpy as np
from .. import log
from ..base import BaseDico, BaseAtom
from . import block

#global _Logger
_Logger = log.Log('WaveletDico', level=0)

class WaveletDico(BaseDico):
    '''
    A wavelet dictionary is a collection of Wavelet Blocks
    each implements a multilevel Wavelet Transform of the data
    '''

    sizes = []
    tolerances = []
    nature = 'Wavelet'
    blocks = None
    max_block_score = 0
    best_current_block = None
    starting_touched_index = 0
    ending_touched_index = -1
    

    def __init__(self, wavelets, pad=8192, debug_level=0):
        '''
        Dictionary stores a joint collection of
        wavelet types and corresponding levels.
        Also the zero padding attribute must be set at initialization        
        '''
        _Logger.set_level(debug_level)
        self.wavelets = wavelets
        self.pad = pad
    
    def get_pad(self):
        return self.pad
    
    def initialize(self, residual_signal):
        ''' Create the collection of blocks specified by the MDCT sizes '''
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        
        # Checking the signal length is a power of 2
        if not np.log2(residual_signal.length) == int(np.log2(residual_signal.length)):
            print "WARNING Wavelets decomposition need signal with powers of two in lengths "
            print "Adding zeroes at the end to reach 2**%d"%np.ceil(np.log2(residual_signal.length))
            missings = 2**np.ceil(np.log2(residual_signal.length)) - residual_signal.length
            if missings % 2==0:
                residual_signal.pad(missings/2)
            else:
                residual_signal.pad((missings+1)/2)
                residual_signal.crop(1,residual_signal.length)
        
        for wav in self.wavelets:
            _Logger.info("Adding %s level %d"%wav)               
            self.blocks.append(block.WaveletBlock(wav, debug_level=_Logger.debugLevel))
    
    def update(self, residual, it_number):
        ''' quite classical : update each block independantly and retrieve the best one
        Selection of an atom will impact all the decomposition levels so for now we reprocess 
        all the projections scores at each iteration'''
        self.max_block_score = 0
        self.best_current_block = None        
            
        for block in self.blocks:            
            block.update(residual)

            if abs(block.max_value) > self.max_block_score:
                self.max_block_score = abs(block.max_value)
                self.best_current_block = block
    
    def get_best_atom(self, debug): 
        if self.best_current_block is None:
            raise ValueError("No best block selected yet")
        return self.best_current_block.get_max_atom()
    
    def compute_touched_zone(self, best_atom):
        pass
    
