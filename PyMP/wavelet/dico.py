'''
Created on Mar 20, 2013

@author: manu
'''
import math
import numpy as np
from .. import log
from ..base import BaseDico, BaseAtom
from . import block
from dbus.bus import _logger

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
        _logger.setLevel(debug_level)
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
        for wav in self.wavelets:
            print wav               
            self.blocks.append(block.WaveletBlock(wav, debug_level=_logger.level))
    
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