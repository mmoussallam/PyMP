'''
PyMP.wavelet.rand.dico  -  Created on Apr 17, 2013
@author: M. Moussallam
'''
"""
Module PyMP.wavelet.rand.dico
=============================

This file handle dctionaries that are used in Randomized Matching Pursuits
see [1] for details.

[1] M. Moussallam, L. Daudet, et G. Richard,
"Matching Pursuits with Random Sequential Subdictionaries"
Signal Processing, vol. 92, pp. 2532-2544 2012.

"""

import math
import numpy as np

from ..dico import WaveletDico
from ...baserand import AbstractSequenceDico
from ... import log
from . import block

# global _Logger
_Logger = log.Log('SSMPWaveletDicos', level=0)


class SequenceDico(AbstractSequenceDico, WaveletDico):
    """ This dictionary implements a sequence of subdictionaries that are shifted in time at each iteration in a pre-defined manner
        the shifts are controlled by the different blocks.

        Attributes:
        -----------
        `sequence_type`: str
            The type of time-shift sequence,
                available choices are *scale*,*random*,*gaussian*
                default is *random* which use a uniform pseudo-random generator
        L_shifts : int , opt
            number of different shifts default is 1000
        x_shifts : int, opt
            range of the shifting in the time axis in samples. default is 512
        `nb_consec_sim`: int
            Number of consecutive iterations with the same time-shift (default is 1)

    """

    # properties
    sequence_type = 'none'  # type of sequence , Scale , Random or Dicho
    it_num = 0  # memorizes the position in the sequence
    nb_consec_sim = 1
    # number of consecutive similar position
    nature = 'RandomSequentialWavelets'

    # constructor
    def __init__(self, wavelets, pad=8192, debug_level=0,
                 seq_type='random',
                 nbSame=1,
                 seed=None,
                 L_shifts=1000,
                 x_shifts=512):
        # calling super metho to AbstractSequenceDico

        super(SequenceDico, self).__init__(seq_type=seq_type,
                                           nbSame=nbSame, seed=seed)
        # The rest is the same as WaveletDico
        _Logger.set_level(debug_level)
        _Logger.info("Creating %s object" % self.__class__.__name__)
        self.wavelets = wavelets
        # Monstruous Hack to be removed
        self.sizes = [-1]*len(wavelets)
        self.pad = pad
        self.L_shifts = L_shifts
        self.x_shifts = x_shifts

    def initialize(self, residual_signal):
        ''' Create the collection of blocks specified by the wavelets '''
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for wav in self.wavelets:
            _Logger.info("Adding %s level %d" % wav)
            self.blocks.append(
                block.SequenceBlock(wav, debug_level=_Logger.debugLevel,
                                    randomType=self.sequence_type,
                                    nbSim=self.nb_consec_sim,
                                    seed=self.seed,
                                    L_shifts=self.L_shifts,
                                    x_shifts=self.x_shifts))
