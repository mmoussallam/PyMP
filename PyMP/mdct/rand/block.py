

import numpy as np
from math import floor

from ...tools import Misc
from ... import win_server
from ... import log
from ... import parallelProjections
from ...baserand import AbstractSequenceBlock
from .. import block as mdct_block
from .. import atom as mdct_atom

# declare global win_server shared by all MDCT blocks instances
#global _PyServer
# _Logger
_PyServer = win_server.get_server()
_Logger = log.Log('SSMPBlocks', level=0)

class SequenceBlock(AbstractSequenceBlock, mdct_block.Block):
    """ block implementing the Randomized Pursuit

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

    # properties
    sequence_type = 'random'
    shift_list = []
    current_shift = 0
    currentSubF = 0 #DEPRECATED
    nb_consec_sim = 1
    w_long = None

    # constructor - initialize residual signal and projection matrix
    def __init__(self, length=0, resSignal=None, frameLen=0, 
                 randomType='random', nbSim=1, 
                 windowType=None,
                 seed=None):
        self.scale = length
        self.residual_signal = resSignal
        
        super(SequenceBlock,self).__init__(randomType=randomType, nbSim=nbSim,                  
                                            seed=seed)

#            np.random.seed()

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residual_signal == None:
            raise ValueError("no signal given")

        self.framed_data_matrix = self.residual_signal.data
        self.frame_num = len(self.framed_data_matrix) / self.frame_len
        self.projs_matrix = np.zeros(len(self.framed_data_matrix))

        self.sequence_type = randomType
        if self.sequence_type == 'scale':
            self.shift_list = range(self.scale / 2)
        elif self.sequence_type == 'random':
            self.shift_list = [floor(
                (self.scale / 2) * (i - 0.5)) for i in np.random.random(self.scale)]
        elif self.sequence_type == 'gaussian':
            self.shift_list = [floor(self.scale / 8 * i)
                  for i in np.random.randn(self.scale / 2)]
            for k in self.shift_list:
                k = min(k, self.scale / 4)
                k = max(k, -self.scale / 4)

        elif self.sequence_type == 'binom':
            self.shift_list = Misc.binom(range(self.scale / 2))
        elif self.sequence_type == 'dicho':
            self.shift_list = Misc.dicho([], range(self.scale / 2))
        elif self.sequence_type == 'jump':
            self.shift_list = Misc.jump(range(self.scale / 2))
        elif self.sequence_type == 'sine':
            self.shift_list = Misc.sine(range(self.scale / 2))
#        elif self.randomType == 'triangle':
#            self.shift_list = Misc.triangle(range(self.scale/2))
        elif self.sequence_type == 'binary':
            self.shift_list = Misc.binary(range(self.scale / 2))

        else:
            self.shift_list = np.zeros(self.scale / 2)

        self.nb_consec_sim = nbSim
        self.windowType = windowType

    # The update method is nearly the same as CCBlock
    def update(self, new_res_signal, startFrameIdx=0, stopFrameIdx=-1, iteration_number=0):
        """ Same as superclass except that at each update, one need to pick a time shift from the sequence """

        # change the current Time-Shift if enough iterations have been done
        if (self.nb_consec_sim > 0):
            if (iteration_number % self.nb_consec_sim == 0):
                self.current_shift = self.shift_list[(
                    iteration_number / self.nb_consec_sim) % len(self.shift_list)]

        self.residual_signal = new_res_signal

        if stopFrameIdx < 0:
            endFrameIdx = self.frame_num - 1
        else:
            endFrameIdx = stopFrameIdx
        L = self.scale

        # update residual signal
        self.framed_data_matrix[startFrameIdx * L / 2: endFrameIdx * L / 2 + L] = self.residual_signal.data[startFrameIdx * self.frame_len: endFrameIdx * self.frame_len + 2 * self.frame_len]

        # TODO changes here
        self.compute_transform(startFrameIdx, stopFrameIdx)

        # TODO changes here
        self.find_max()

    # inner product computation through MCLT with all possible time shifts
    def compute_transform(self,   startingFrame=1, endFrame=-1):
        if self.w_long is None:
            self.initialize()

        if endFrame < 0:
            endFrame = self.frame_num - 2

        if startingFrame < 2:
            startingFrame = 2
#        L = self.scale
#        K = L/2;

        ############" changes  ##############
#        T = K/2 + self.current_shift
        # new version with C calculus
        parallelProjections.project(self.framed_data_matrix, self.best_score_tree,
                                                 self.projs_matrix,
                                                 self.locCoeff,
                                                 self.post_twid_vec,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale,
                                                 int(self.current_shift))

    def find_max(self):
        treeMaxIdx = self.best_score_tree.argmax()
        maxIdx = np.abs(self.projs_matrix[treeMaxIdx * self.scale /
            2: (treeMaxIdx + 1) * self.scale / 2]).argmax()

        self.maxIdx = maxIdx + treeMaxIdx * self.scale / 2
        self.max_value = self.projs_matrix[self.maxIdx]

    # construct the atom
    def get_max_atom(self):
        self.max_frame_idx = floor(self.maxIdx / (0.5 * self.scale))
        self.max_bin_idx = self.maxIdx - self.max_frame_idx * (0.5 * self.scale)
        Atom = mdct_atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residual_signal.fs)
        Atom.frame = self.max_frame_idx
        Atom.mdct_value = self.max_value

        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom()

        if self.windowType == 'half1':
            Atom.waveform[0:self.scale / 2] = 0

        Atom.time_shift = self.current_shift
        Atom.time_position -= Atom.time_shift

        return Atom


class StochasticBlock(mdct_block.Block):
    """ Block implementing the stochastic MP:
        atom selection is probabilistic """
        
    def __init__(self, length=0, resSignal=None, frameLen=0, 
                 randomType='proba',
                 windowType=None,
                 seed=None,
                 sigma = 1,
                 debug_level=0):
        _Logger.set_level(debug_level)
        
        self.scale = length
        self.residual_signal = resSignal
        self.seed = seed

        np.random.seed(self.seed)

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residual_signal == None:
            raise ValueError("no signal given")

        self.framed_data_matrix = self.residual_signal.data
        self.frame_num = len(self.framed_data_matrix) / self.frame_len
        self.projs_matrix = np.zeros(len(self.framed_data_matrix))

        self.randomType = randomType
        
        self.windowType = windowType
    
        self.sigma= sigma
    
    def find_max(self):
        """ the selection step is probabilistic: the max
            index is taken at random with a probability proportionnal to
            the projections 
            
            values are chosen with probability linearly proportionnal 
            to exp(sigma * projection)
            """
        
        # draw a new index vector whose weights are poduct of
        # a random variable and the projection matrix
        probabilities = np.exp(self.sigma * np.abs(self.projs_matrix)) - np.exp(0)
        # normalize so it adds to 1                
        
        probabilities /= np.sum(probabilities)        
        # create the bins
        bins = np.add.accumulate(probabilities)
        
        # now draw an index at random: use it as the selected atom
        self.maxIdx = np.digitize(np.random.random_sample(1), bins)[0]
        
#        print np.max(np.abs(self.projs_matrix))
        # deduce the value of the selected atom
        self.max_value = self.projs_matrix[self.maxIdx]
        _Logger.debug("Chose index %d at random with value %1.5f"%(self.maxIdx, self.max_value))
        
#        print self.max_value

    def get_max_atom(self):
        self.max_frame_idx = floor(self.maxIdx / (0.5 * self.scale))
        self.max_bin_idx = self.maxIdx - self.max_frame_idx * (0.5 * self.scale)
        Atom = mdct_atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residual_signal.fs)
        Atom.frame = self.max_frame_idx
        Atom.mdct_value = self.max_value

        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom()


        return Atom
    