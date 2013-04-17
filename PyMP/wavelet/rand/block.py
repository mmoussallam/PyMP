'''
PyMP.wavelet.rand.block  -  Created on Apr 17, 2013
@author: M. Moussallam
'''
import numpy as np
from math import floor

from ...tools import Misc
from ... import win_server
from ... import log
from ...baserand import AbstractSequenceBlock
from .. import block as wavelet_block
from .. import atom as wavelet_atom

# declare global win_server shared by all MDCT blocks instances
#global _PyServer
# _Logger
import pywt
_PyServer = win_server.get_server()
_Logger = log.Log('SSMPWaveletBlocks', level=0)


class SequenceBlock(AbstractSequenceBlock, wavelet_block.WaveletBlock):
    """ block implementing the Randomized Pursuit with wavelets

    Attributes
    ----------
    `sequence_type`: str
        The type of time-shift sequence, available choices are 
            *scale*
            *random*
            *gaussian*
            *binom*
            *dicho*
            *jump*
            *binary*
        default is *random* which use a uniform pseudo-random generator
    `shift_list`: array-like
        The actual sequence of subdictionary time-shifts
    `current_shift`: int
        The current time-shift
    `nbSim`: int
        Number of consecutive iterations with the same time-shift (default is 1)
    """
    def __init__(self, wav, mode ='per', debug_level=0,
                 L_shifts =1000,
                 x_shifts=512,
                 randomType='random', nbSim=1, seed=None):
        '''
        Constructor
        '''
        
        super(SequenceBlock,self).__init__(randomType=randomType, nbSim=nbSim,                  
                                            seed=seed)
        _Logger.set_level(debug_level)
        self.nature = wav[0]
        self.level = int(wav[1])
        self.mode = mode
        self.Lshifts = L_shifts
        self.Xshifts = x_shifts
        # for the projections we keep the list structure of pywavelets
        self.projections = None
        _Logger.info("Created new Sequence Wavelet Block : %s - %d levels "%(self.nature, self.level))
        self.max_value = None
        self.max_index = None
        self.max_level = None
        self.residual_signal = None
        self.scales = None
        
        self.sequence_type = randomType
        if self.sequence_type == 'scale':
            self.shift_list = range(self.Xshifts)
        elif self.sequence_type == 'random':
            self.shift_list = [floor(
                (self.Xshifts / 2) * (i - 0.5)) for i in np.random.random(L_shifts)]
        elif self.sequence_type == 'gaussian':
            self.shift_list = [floor(self.Xshifts / 8 * i)
                  for i in np.random.randn(self.Lshifts / 2)]
            for k in self.shift_list:
                k = min(k, self.Xshifts / 4)
                k = max(k, -self.Xshifts / 4)

        else:
            _Logger.error("Unrecognized sequence type")
            raise ValueError("Unrecognized sequence type")
            self.shift_list = np.zeros(self.Xshifts)
        
    
    def update(self, new_res_signal, iteration_number=0):
        ''' recompute the projections using the fast cascading wavelet transforms
        
        difference with superclass is that the signal is randomly 
        shifted according to pre-computed sequence
        '''
        if (self.nb_consec_sim > 0):
            if (iteration_number % self.nb_consec_sim == 0):
                self.current_shift = self.shift_list[(
                    iteration_number / self.nb_consec_sim) % len(self.shift_list)]
        
        self.residual_signal = new_res_signal.copy()                

        # shifting
        self.current_shift = int(self.current_shift)
        self.residual_signal.data = np.roll(self.residual_signal.data, self.current_shift, axis=0)
        
        self.compute_transform()

        self.find_max()
      
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
                
        best_atom = wavelet_atom.WaveAtom(scale=0,
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
#                print len(self.projections), self.max_level,self.projections[self.max_level].shape, self.max_index        
                coeffs[-1][self.max_index] = self.projections[self.max_level][self.max_index]
        
        best_atom.waveform = pywt.waverec(coeffs, self.nature, mode='per')
        
        best_atom.length = best_atom.waveform.shape[0]
        
        # CHANGES: before it is done : need to shift back the signal
        best_atom.waveform = np.roll(best_atom.waveform, -self.current_shift)
        

        _Logger.debug("Best atom selected : %s"%str(best_atom))
        return best_atom
    