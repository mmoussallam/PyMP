
"""
Module PyMP.mdct.rand.dico
==========================

This file handle dctionaries that are used in Randomized Matching Pursuits
see [1] for details.

[1] M. Moussallam, L. Daudet, et G. Richard,
"Matching Pursuits with Random Sequential Subdictionaries"
Signal Processing, vol. 92, pp. 2532-2544 2012.

"""

import math
import numpy as np

from ..dico import Dico
from ... import log
from . import block


#global _Logger
_Logger = log.Log('SSMPDicos', level=0)


class SequenceDico(Dico):
    """ This dictionary implements a sequence of subdictionaries that are shifted in time at each iteration in a pre-defined manner
        the shifts are controlled by the different blocks.

        Attributes:
        -----------        
        `sequence_type`: The type of time-shift sequence, 
                available choices are *scale*,*random*,*gaussian*,*binom*,*dicho*,*jump*,*binary* 
                default is *random* which use a uniform pseudo-random generator

        `nb_consec_sim`: Number of consecutive iterations with the same time-shift (default is 1)

    """

    # properties
    sequence_type = 'none'  # type of sequence , Scale , Random or Dicho
    it_num = 0  # memorizes the position in the sequence
    nb_consec_sim = 1
    # number of consecutive similar position
    nature = 'RandomSequentialMDCT'

    # constructor
    def __init__(self, sizes=[], seq_type='random', nbSame=1,
                 windowType=None,seed=None):
        self.sequence_type = seq_type
        self.sizes = sizes
        self.nb_consec_sim = nbSame
        self.windowType = windowType
        self.seed = seed
        
        
    def initialize(self, residualSignal):
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1

        for mdctSize in self.sizes:
            # check whether this block should optimize time localization or not
            self.blocks.append(block.SequenceBlock(mdctSize, residualSignal, randomType=self.
                sequence_type, nbSim=self.nb_consec_sim, windowType=self.windowType, seed=self.seed))

    def compute_touched_zone(self, previousBestAtom):
        # if the current time shift is about to change: need to recompute all
        # the scores
        if (self.nb_consec_sim > 0):
            if ((self.it_num + 1) % self.nb_consec_sim == 0):
                self.starting_touched_index = 0
                self.ending_touched_index = -1
            else:                
                self.starting_touched_index = 0
                self.ending_touched_index = -1
#                self.starting_touched_index = previousBestAtom.time_position - 2* previousBestAtom.length
#                self.ending_touched_index = self.starting_touched_index + 2.5 * previousBestAtom.length
        else:
            self.starting_touched_index = previousBestAtom.time_position - previousBestAtom.length / 2
            self.ending_touched_index = self.starting_touched_index + 1.5 * previousBestAtom.length

    def update(self, residualSignal, iterationNumber=0, debug=0):
        self.max_block_score = 0
        self.best_current_block = None
        self.it_num = iterationNumber
        # BUGFIX STABILITY
#        self.endingTouchedIndex = -1
#        self.startingTouchedIndex = 0

        for block in self.blocks:
            startingTouchedFrame = int(
                math.floor(self.starting_touched_index / (block.scale / 2)))
            if self.ending_touched_index > 0:
                endingTouchedFrame = int(math.floor(self.
                    ending_touched_index / (block.scale / 2))) + 1
                # TODO check this
            else:
                endingTouchedFrame = -1

            block.update(residualSignal,
                 startingTouchedFrame, endingTouchedFrame, iterationNumber)

            if np.abs(block.max_value) > self.max_block_score:
#                self.maxBlockScore = block.getMaximum()
                self.max_block_score = np.abs(block.max_value)
                self.best_current_block = block

    def getSequences(self, length):
        sequences = []
        for block in self.blocks:
            sequences.append(block.shift_list[0:length])
        return sequences

class StochasticDico(Dico):
    """ This dictionary implements Stochastic MP as described
        in Elad et al 2009 (see also Ferrando et al 2000)
        
        All products are computed but the best atom is chosen
        probabilistically. Many parallel pursuits are then averaged."""
    
    # properties
    nature = 'RandomStochasticMDCT'

    # constructor
    def __init__(self, sizes=[], seq_type='random', nbSame=1,
                 windowType=None,seed=None, sigma=1.0):
        self.sequence_type = seq_type
        self.sizes = sizes
        self.nb_consec_sim = nbSame
        self.windowType = windowType
        self.seed = seed
        self.sigma = sigma
        
        
    def initialize(self, residualSignal):
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1

        for mdctSize in self.sizes:
            # check whether this block should optimize time localization or not
            self.blocks.append(block.StochasticBlock(mdctSize, residualSignal,
                                                     windowType=self.windowType,
                                                     seed=self.seed,
                                                     sigma = self.sigma))

    # DO I need to restate it?
    def update(self, residualSignal, iterationNumber=0, debug=0):
        self.max_block_score = 0
        self.best_current_block = None
        self.it_num = iterationNumber
        # BUGFIX STABILITY
#        self.endingTouchedIndex = -1
#        self.startingTouchedIndex = 0

        for block in self.blocks:
            startingTouchedFrame = int(
                math.floor(self.starting_touched_index / (block.scale / 2)))
            if self.ending_touched_index > 0:
                endingTouchedFrame = int(math.floor(self.
                    ending_touched_index / (block.scale / 2))) + 1
                # TODO check this
            else:
                endingTouchedFrame = -1

            block.update(residualSignal,
                 startingTouchedFrame, endingTouchedFrame)

            if np.abs(block.max_value) > self.max_block_score:
#                self.maxBlockScore = block.getMaximum()
                self.max_block_score = np.abs(block.max_value)
                self.best_current_block = block
