'''
Abstract classes for Randomized Pursuits
'''
import numpy as np
from base import BaseDico, BaseBlock

class AbstractSequenceDico(BaseDico):
    """ RSSMP dictionary interface.
            
        Attributes:
        -----------        
        `sequence_type`: str
            The type of time-shift sequence, 
            available choices are *scale*,*random*,*gaussian*,*binom*,*dicho*,*jump*,*binary* 
            default is *random* which use a uniform pseudo-random generator
        `nb_consec_sim`: int
            Number of consecutive iterations with the same time-shift (default is 1)

    """

    # properties
    sequence_type = 'none'  # type of sequence , Scale , Random or Dicho
    it_num = 0  # memorizes the position in the sequence
    nb_consec_sim = 1
    # number of consecutive similar position
    nature = 'AbstractSequenceDico'

    # constructor
    def __init__(self, seq_type='random', nbSame=1, seed=None):
        self.sequence_type = seq_type        
        self.nb_consec_sim = nbSame            
        self.seed = seed
        
    def getSequences(self, length):
        sequences = []
        for block in self.blocks:
            sequences.append(block.shift_list[0:length])
        return sequences
    
    
class AbstractSequenceBlock(BaseBlock):
    """ block implementing RSSMP

    Attributes
    ----------
    `sequence_type`: str
        The type of time-shift sequence 
    `shift_list`: array-like
        The actual sequence of subdictionary time-shifts
    `current_shift`: int
        The current time-shift
    `nbSim`: int
        Number of consecutive iterations with the same time-shift (default is 1)
    """

    # properties
    sequence_type = 'random'
    shift_list = []
    current_shift = 0
    currentSubF = 0 #DEPRECATED
    nb_consec_sim = 1
    
    def __init__(self, 
                 randomType='random', nbSim=1,                  
                 seed=None):

        self.seed = seed
        np.random.seed(self.seed)
        
        
        