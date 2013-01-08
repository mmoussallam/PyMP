

import numpy as np
from math import floor

from ...tools import Misc
from ... import win_server
from ... import log
from ... import parallelProjections
from .. import block as mdct_block
from .. import atom as mdct_atom


# declare global win_server shared by all MDCT blocks instances
global _PyServer, _Logger
_PyServer = win_server.PyServer()
_Logger = log.Log('RandomMDCTBlock', level=0)


class SequenceBlock(mdct_block.Block):
    """ block implementing the Randomized Pursuit

    Attributes:

        `sequence_type`: The type of time-shift sequence, available choices are 

                *scale*
                *random*
                *gaussian*
                *binom*
                *dicho*
                *jump*
                *binary*

            default is *random* which use a uniform pseudo-random generator

        `shift_list`: The actual sequence of subdictionary time-shifts

        `current_shift`: The current time-shift

        `nbSim`: Number of consecutive iterations with the same time-shift (default is 1)
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
        self.seed = seed
        if self.seed is not None:
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
    def update(self, new_res_signal, startFrameIdx=0, stopFrameIdx=-1, iterationNumber=0):
        """ Same as superclass except that at each update, one need to pick a time shift from the sequence """

        # change the current Time-Shift if enough iterations have been done
        if (self.nb_consec_sim > 0):
            if (iterationNumber % self.nb_consec_sim == 0):
                self.current_shift = self.shift_list[(
                    iterationNumber / self.nb_consec_sim) % len(self.shift_list)]

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
#        computeMCLT.project(self.enframedDataMatrix, self.bestScoreTree,
#                                                 self.projectionMatrix ,
#                                                 self.locCoeff ,
#                                                 self.post_twidVec ,
#                                                 startingFrame,
#                                                 endFrame,
#                                                 self.scale ,
#                                                 int(self.current_shift))

#        normaCoeffs = sqrt(2/float(K))
##        locCoeff = self.wLong * self.pre_twidVec
#        for i in range(startingFrame , endFrame):
#
#
#            x = self.enframedDataMatrix[i*K - T: i*K + L - T]
#            if len(x) !=L:
#                x =zeros(L , complex);
#
# compute windowing and pre-twiddle simultaneously : suppose first and last
# frame are always zeroes
#            x = x * self.locCoeff
#
#            # compute fft
#            self.fftMat[: , i] = fft(x , L)
#
#            # post-twiddle
#            y = self.fftMat[0:K , i] * self.post_twidVec
#
#            try:
#                self.projectionMatrix[i*K : (i+1)*K] = normaCoeffs*y.real;
#            except:
#                print "oups here"
#            # store new max score in tree
# self.bestScoreTree[i] = abs(self.projectionMatrix[i*K : (i+1)*K]).max()
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
