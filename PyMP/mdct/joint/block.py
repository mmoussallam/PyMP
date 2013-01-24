#
#                       mdct.joint.block
#
#
#
# M. Moussallam                             Created on Dec 27, 2012
# -----------------------------------------------------------------------
#
#
"""
Module mdct.joint.block
=======================

Please refer to superclass for documentation

This file handle blocks that are used in Joint Matching Pursuis

* documentation coming soon *

"""
import numpy as np
from math import floor

from ... import win_server, log
from ... import parallelProjections
from ...mdct import block as mdct_block
from ...mdct import atom as mdct_atom

# declare global win_server shared by all MDCT blocks instances
#global _PyServer
# _Logger
_PyServer = win_server.get_server()
_Logger = log.Log('JointBlocks', level=0)


class SetBlock(mdct_block.Block):
    """ Classic MDCT block useful for handling Sets Only the update routine is changed: no need to
        look for the max since it will be done later in the dictionary

        NO ATOM Local Optimization!!
        """

    # parameters
    frame_len = 0
    frame_num = 0

    max_index = 0
    max_value = 0
    max_bin_idx = 0
    max_frame_idx = 0

    # MDCT static window and twiddle parameters
    w_long = None
    enframedDataMatrixList = []

    startingFrameList = None
    endingFrameList = None

    # constructor - initialize residual signals and projection matrix for each
    # of the signals
    def __init__(
        self, length=0, resSignalList=None, frameLen=0, useC=True,
            debug_level=None, nature='sum', tolerance=None):
        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.scale = length
        self.residualSignalList = resSignalList

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residualSignalList == None:
            raise ValueError("no signal given")

        # reference length
        self.length = max(
            [resSignalList[i].length for i in range(len(resSignalList))])

        # number of signals to handle
        self.sigNumber = len(self.residualSignalList)

        # initialize data matrix
        self.enframedDataMatrixList = []
        for i in range(self.sigNumber):
            self.enframedDataMatrixList.append(np.zeros((self.length, 1)))

        for sigIdx in range(self.sigNumber):
            # Important point : assert that all signals have same length!
            if self.residualSignalList[sigIdx].length < self.length:
                print ValueError('Signal ' + str(sigIdx) + ' is too short!' + str(self.residualSignalList[sigIdx].length) + " instead of " + str(self.length))

                # TODO :  just forbid the creation of atom in this zone
                pad = self.length - self.residualSignalList[sigIdx].length
                self.residualSignalList[sigIdx].data = np.concatenate(
                    (self.residualSignalList[sigIdx].data, np.zeros(pad)))
                self.residualSignalList[sigIdx].length += pad

            self.enframedDataMatrixList[
                sigIdx] = self.residualSignalList[sigIdx].data

        self.frame_num = len(self.enframedDataMatrixList[0]) / self.frame_len

        # The projection matrix is unidimensionnal since only one atom will be chosen eventually
# self.projectionMatrix = zeros(len(self.enframedDataMatrixList[0])
# ,complex)
        self.projs_matrix = np.zeros(
            (len(self.enframedDataMatrixList[0]), self.sigNumber), float)

        self.use_c_optim = useC

        if nature == 'sum':
            self.nature = 0

        elif nature == 'median':
            self.nature = 1
#        elif nature == 'maximin':
#            self.nature = 2

        else:
            raise ValueError('Unrecognized Criterion for selection')

        if tolerance is not None:
            self.tolerance = tolerance
        else:
            self.tolerance = 1

        _Logger.info('new MDCT Setblock constructed size : ' + str(
            self.scale) + ' tolerance of :' + str(self.tolerance))

    def initialize(self):

        # Windowing
        L = self.scale

        self.w_long = np.array(
            [np.sin(float(l + 0.5) * (np.pi / L)) for l in range(L)])

        # twidlle coefficients
        self.pre_twid_vec = np.array(
            [np.exp(n * (-1j) * np.pi / L) for n in range(L)])
        self.post_twid_vec = np.array([np.exp(
            (float(n) + 0.5) * -1j * np.pi * (L / 2 + 1) / L) for n in range(L / 2)])

        # score tree - first version simplified
        self.best_score_tree = np.zeros(self.frame_num)

        # OPTIM -> do pre-twid directly in the windows
        self.locCoeff = self.w_long * self.pre_twid_vec

    def compute_transform(self, startingFrameList=None, endFrameList=None):
        if self.w_long is None:
            self.initialize()

        # due to later time-shift optimizations , need to ensure nothing is
        # selected too close to the borders!!
        if startingFrameList is None:
            startingFrameList = [2] * self.sigNumber

        if endFrameList is None:
            endFrameList = [self.frame_num - 3] * self.sigNumber

        for i in range(self.sigNumber):
            if endFrameList[i] < 0 or endFrameList[i] > self.frame_num - 3:
                endFrameList[i] = self.frame_num - 3

            if startingFrameList[i] < 2:
                startingFrameList[i] = 2

# REFACTORED VIOLENT 21/11

        # ALL SIGNALS PROJECTED AND WE KEEP ALL PROJECTIONS, NO OPTIMIZATIONS !!
        for sigIdx in range(0, self.sigNumber):
            localProj = np.array(self.projs_matrix[:, sigIdx])
#            print localProj.shape
            localProj = localProj.reshape((self.projs_matrix.shape[0],))
            parallelProjections.project(self.enframedDataMatrixList[sigIdx],
                                        self.best_score_tree,
                                        localProj,
                                        self.locCoeff,
                                        self.post_twid_vec,
                                        startingFrameList[sigIdx],
                                        endFrameList[sigIdx],
                                        self.scale, 0)

        # WE NEED TO RECOMPUTE THE GLOBAL SUM SCORE NOW
            self.projs_matrix[:, sigIdx] = np.array(localProj.copy())
# print self.scale , startingFrameList[sigIdx] , endFrameList[sigIdx]
# ,self.frameNumber * self.scale/2
        if self.nature == 0:
            sumOfProjections = np.sum(self.projs_matrix ** 2, 1)
        elif self.nature == 1:
            sumOfProjections = np.median(self.projs_matrix ** 2, 1)
#        plt.figure()
#        plt.plot(self.projectionMatrix[:,sigIdx])
#        plt.show()
#        print sumOfProjections.shape
        self.maxIdx = sumOfProjections.argmax()
#        print
#        print "Found max index: ",self.maxIdx
#            plt.figure()
#            plt.subplot(212)
#            plt.plot((self.projectionMatrix))
##        plt.subplot(212)
##        plt.plot(self.bestScoreTree)
#        plt.show()
#

    def update(self, newResidualList, startFrameList=None, stopFrameList=None):
# print "block update called : " , self.scale  , self.frameNumber , " type
# : " , self.nature
        self.residualSignalList = newResidualList

        if startFrameList is None:
            startFrameList = [2] * self.sigNumber

        if stopFrameList is None:
            stopFrameList = [self.frame_num - 3] * self.sigNumber
#
    # MODIF: each signal is updated according to its own limits since
    # sometimes no atoms have been subtracted
        L = self.scale
        for sigIdx in range(self.sigNumber):

            if stopFrameList[sigIdx] < 0:
                stopFrameList[sigIdx] = self.frame_num - 2
            else:
                stopFrameList[sigIdx] = min((stopFrameList[
                                            sigIdx], self.frame_num - 2))
# print "block update called : " , startFrameList[sigIdx]  ,
# stopFrameList[sigIdx] , " type : " , self.scale
            self.enframedDataMatrixList[sigIdx][startFrameList[sigIdx] * L / 2: stopFrameList[sigIdx] * L / 2 + L] = self.residualSignalList[sigIdx].data[startFrameList[sigIdx] * self.frame_len: stopFrameList[sigIdx] * self.frame_len + 2 * self.frame_len]

        self.compute_transform(startFrameList, stopFrameList)

        self.find_max()

    def find_max(self):

#        treeMaxIdx = self.bestScoreTree.argmax()
#        print "Tree max Idx :",treeMaxIdx
# maxIdx = abs(self.projectionMatrix[treeMaxIdx*self.scale/2 :
# (treeMaxIdx+1)*self.scale/2]).argmax()

#        self.maxIdx = maxIdx + treeMaxIdx*self.scale/2
        self.max_value = np.sum(self.projs_matrix[self.maxIdx, :] ** 2)

#        self.maxFrameIdx = treeMaxIdx
        self.max_frame_idx = self.maxIdx / (self.scale / 2)

        self.max_bin_idx = self.maxIdx - (self.max_frame_idx * self.scale / 2)

# print "ppBlock : line 1796, index, frame , value , bin ",self.maxIdx
# ,self.maxFrameIdx , self.maxValue , self.maxBinIdx

    # TODO use subclass : MDCTAtom but later
    def synthesize_atom(self, value=None):
        ###################" new version ############"
        global _PyServer
#        print len(_PyServer.Waveforms)
        if value is None:
            return self.max_value * _PyServer.get_waveform(self.scale, self.max_bin_idx)
        else:
            return value * _PyServer.get_waveform(self.scale, self.max_bin_idx)

    def get_optimized_best_atoms(self, debug=0, noAdapt=True):
        _Logger.warning("No adaptation is allowed in this mode")
        return self.get_not_optimized_best_atoms()

    def get_not_optimized_best_atoms(self):
        # hack here : We just compute the value by projecting the waveform onto
        # the signal
        AtomList = []
        for sigIdx in range(self.sigNumber):
            offset = self.scale / 4
            Atom = mdct_atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - offset, 0), self.max_bin_idx, self.residualSignalList[0].fs)
            Atom.frame = self.max_frame_idx
            Atom.synthesize_ifft(1)
#            Atom.waveform /= sum(Atom.waveform**2)

            Atom.proj_score = self.projs_matrix[self.maxIdx, sigIdx]
            Atom.mdct_value = Atom.proj_score
            Atom.waveform *= Atom.proj_score
#            print Atom.projectionScore

            AtomList.append(Atom)

        return AtomList


class SetLOBlock(mdct_block.Block):
    """ Classic MDCT block useful for handling Sets Only the update routine is changed: no need to
        look for the max since it will be done later in the dictionary

        USE ONLY WHEN LOCAL ADAPTATION IS EXPECTED

        """

    # parameters
    frame_len = 0
    frame_num = 0

    max_index = 0
    max_value = 0
    max_bin_idx = 0
    max_frame_idx = 0

    # MDCT static window and twiddle parameters
    w_long = None
    enframedDataMatrixList = []

    startingFrameList = None
    endingFrameList = None

    # constructor - initialize residual signals and projection matrix for each
    # of the signals
    def __init__(
        self, length=0, resSignalList=None, frameLen=0, useC=True,
            debug_level=None, nature='sum', tolerance=None):
        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.scale = length
        self.residualSignalList = resSignalList

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residualSignalList == None:
            raise ValueError("no signal given")

        # reference length
        self.length = max(
            [resSignalList[i].length for i in range(len(resSignalList))])

        # number of signals to handle
        self.sigNumber = len(self.residualSignalList)

        # initialize data matrix
        self.enframedDataMatrixList = []
        for i in range(self.sigNumber):
            self.enframedDataMatrixList.append(np.zeros((self.length, 1)))

        for sigIdx in range(self.sigNumber):
            # Important point : assert that all signals have same length!
            if self.residualSignalList[sigIdx].length < self.length:
                print ValueError('Signal ' + str(sigIdx) + ' is too short!' + str(self.residualSignalList[sigIdx].length) + " instead of " + str(self.length))

                # TODO :  just forbid the creation of atom in this zone
                pad = self.length - self.residualSignalList[sigIdx].length
                self.residualSignalList[sigIdx].data = np.concatenate(
                    (self.residualSignalList[sigIdx].data, np.zeros(pad)))
                self.residualSignalList[sigIdx].length += pad

            self.enframedDataMatrixList[
                sigIdx] = self.residualSignalList[sigIdx].data

        self.frame_num = len(self.enframedDataMatrixList[0]) / self.frame_len

        # The projection matrix is unidimensionnal since only one atom will be chosen eventually
# self.projectionMatrix = zeros(len(self.enframedDataMatrixList[0])
# ,complex)
        self.projs_matrix = np.zeros(
            len(self.enframedDataMatrixList[0]), float)

        self.use_c_optim = useC

        if nature == 'sum':
            self.nature = 0

        elif nature == 'prod':
            self.nature = 1
        elif nature == 'maximin':
            self.nature = 2

        else:
            raise ValueError('Unrecognized Criterion for selection')

        if tolerance is not None:
            self.tolerance = tolerance
        else:
            self.tolerance = 2

        _Logger.info('new MDCT Setblock constructed size : ' + str(
            self.scale) + ' tolerance of :' + str(self.tolerance))

    def initialize(self):

        # Windowing
        L = self.scale

        self.w_long = np.array(
            [np.sin(float(l + 0.5) * (np.pi / L)) for l in range(L)])

        # twidlle coefficients
        self.pre_twid_vec = np.array(
            [np.exp(n * (-1j) * np.pi / L) for n in range(L)])
        self.post_twid_vec = np.array([np.exp(
            (float(n) + 0.5) * -1j * np.pi * (L / 2 + 1) / L) for n in range(L / 2)])

        # score tree - first version simplified
        self.best_score_tree = np.zeros(self.frame_num)

        # OPTIM -> do pre-twid directly in the windows
        self.locCoeff = self.w_long * self.pre_twid_vec

    def compute_transform(self, startingFrameList=None, endFrameList=None):
        if self.w_long is None:
            self.initialize()

        # due to later time-shift optimizations , need to ensure nothing is
        # selected too close to the borders!!
        if startingFrameList is None:
            startingFrameList = [2] * self.sigNumber

        if endFrameList is None:
            endFrameList = [self.frame_num - 3] * self.sigNumber

        for i in range(self.sigNumber):
            if endFrameList[i] < 0 or endFrameList[i] > self.frame_num - 3:
                endFrameList[i] = self.frame_num - 3

            if startingFrameList[i] < 2:
                startingFrameList[i] = 2

        # For each signal: update the projection matrix in the given boundaries with the appropriate method
        # TODO REFACTORING: do it in one pass in the C code
        # first signal:
#        print "Update projection 1"
# print "block update called : " , startingFrameList[0]  , endFrameList[0]
# , " type : " , self.scale
        parallelProjections.project_mclt(
            self.enframedDataMatrixList[0], self.best_score_tree,
            self.projs_matrix,
            self.locCoeff,
            self.post_twid_vec,
            startingFrameList[0],
            endFrameList[0],
            self.scale)
#        plt.figure()
#        plt.subplot(211)
#        plt.plot(abs(self.projectionMatrix))
#        plt.subplot(212)
#        plt.plot(self.bestScoreTree)
#        plt.show()
        # WORKAROUND DEBUG
        self.projs_matrix[startingFrameList[0] * self.frame_len: endFrameList[0] * self.frame_len] = abs(self.projs_matrix[startingFrameList[0] * self.frame_len: endFrameList[0] * self.frame_len])

        # next signals: update depend on strategies!
        for sigIdx in range(1, self.sigNumber):
#            print "Update projection "+str(sigIdx+1)
#            print startingFrameList[sigIdx] , endFrameList[sigIdx]
            parallelProjections.project_mclt_set(
                self.enframedDataMatrixList[sigIdx],
                self.best_score_tree,
                self.projs_matrix,
                self.locCoeff,
                self.post_twid_vec,
                startingFrameList[sigIdx],
                endFrameList[sigIdx],
                self.scale,
                self.nature)
#            plt.figure()
#            plt.subplot(212)
#            plt.plot((self.projectionMatrix))
##        plt.subplot(212)
##        plt.plot(self.bestScoreTree)
#        plt.show()
#

    def update(self, newResidualList, startFrameList=None, stopFrameList=None):
# print "block update called : " , self.scale  , self.frameNumber , " type
# : " , self.nature
        self.residualSignalList = newResidualList

        if startFrameList is None:
            startFrameList = [2] * self.sigNumber

        if stopFrameList is None:
            stopFrameList = [self.frame_num - 3] * self.sigNumber
#
    # MODIF: each signal is updated according to its own limits since
    # sometimes no atoms have been subtracted
        L = self.scale
        for sigIdx in range(self.sigNumber):

            if stopFrameList[sigIdx] < 0:
                stopFrameList[sigIdx] = self.frame_num - 2
            else:
                stopFrameList[sigIdx] = min((stopFrameList[
                                            sigIdx], self.frame_num - 2))
# print "block update called : " , startFrameList[sigIdx]  ,
# stopFrameList[sigIdx] , " type : " , self.scale
            self.enframedDataMatrixList[sigIdx][startFrameList[sigIdx] * L / 2: stopFrameList[sigIdx] * L / 2 + L] = self.residualSignalList[sigIdx].data[startFrameList[sigIdx] * self.frame_len: stopFrameList[sigIdx] * self.frame_len + 2 * self.frame_len]

        self.compute_transform(startFrameList, stopFrameList)

        self.find_max()

    def find_max(self):
        treeMaxIdx = self.best_score_tree.argmax()
#        print "Tree max Idx :",treeMaxIdx
        maxIdx = abs(self.projs_matrix[treeMaxIdx * self.scale / 2: (
            treeMaxIdx + 1) * self.scale / 2]).argmax()

        self.maxIdx = maxIdx + treeMaxIdx * self.scale / 2
        self.max_value = self.projs_matrix[self.maxIdx]

        self.max_frame_idx = treeMaxIdx
        self.max_bin_idx = maxIdx
#        print treeMaxIdx , maxIdx , (self.maxValue)

    # TODO use subclass : MDCTAtom but later
    def synthesize_atom(self, value=None):
        ###################" new version ############"
        global _PyServer
#        print len(_PyServer.Waveforms)
        if value is None:
            return self.max_value * _PyServer.get_waveform(self.scale, self.max_bin_idx)
        else:
            return value * _PyServer.get_waveform(self.scale, self.max_bin_idx)

    def get_optimized_best_atoms(self, debug=0, noAdapt=False):
        """ Here the index of the best atom is chosen one level up in the dictionary set
            because it depends on all the signals. Now we just try to locally adapt this
            atom in the best possible way """

        if noAdapt:
            print "No adaptation of the atom requested!"
            return self.get_not_optimized_best_atoms()
#        self.maxFrameIdx = floor(self.maxIdx / (0.5*self.scale))
#        self.maxBinIdx = self.maxIdx - self.maxFrameIdx * (0.5*self.scale)

        # initialize atom fft
        fftVec = np.zeros(self.scale * self.tolerance, complex)

# print "Searching optimization of atom " , self.maxFrameIdx ,
# self.maxBinIdx
        bestAtoms = []

        # call subfunction that adapt the prototyped atom to each of the signals
        for sigIdx in range(self.sigNumber):
#            print "Starting ", sigIdx

            bestAtoms.append(self.getAdaptedAtom(sigIdx, debug, fftVec))
#            print "Ending ", sigIdx
        return bestAtoms

    def getAdaptedAtom(self, sigIdx, debug, fftVec):
        # hack here : let us project the atom waveform on the neighbouring signal in the FFt domain,
        # so that we can find the maximum correlation and best adapt the time-shift
        # Construct prototype atom

        Atom = mdct_atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residualSignalList[0].fs)
        Atom.frame = self.max_frame_idx

        Atom.mdct_value = 1.0
        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom(value=1)
        Atom.time_shift = 0
        Atom.proj_score = 0.0

        # width of tolerance : equals to the width of the area scanned on left
        # and right to find optimal position
        HalfWidth = (self.tolerance - 1) * self.scale / 2

        # do not forget to compensate for the quarter window offsetted
        offset = self.scale / 4
        startSample = (
            (self.max_frame_idx) * self.scale / 2) - HalfWidth - offset
        stopSample = (
            (self.max_frame_idx + 2) * self.scale / 2) + HalfWidth - offset

#        startSample = ((self.maxFrameIdx -1.5) * self.scale/2)
#        stopSample = ((self.maxFrameIdx +2.5) * self.scale/2)

#        print "BLOCK line 1302" ,HalfWidth , startSample , stopSample , stopSample - startSample
        # retrieve the corresponding data
        input1 = self.enframedDataMatrixList[sigIdx][startSample: stopSample]

        if HalfWidth > 0:
        # surround the canonical atom waveform by zeroes
            input2 = np.concatenate((np.concatenate((np.zeros(
                HalfWidth), Atom.waveform)), np.zeros(HalfWidth)))

        else:
            # no time shift allowed !!!
            _Logger.warning("No time shift with this tolerance value!!")

            # retrieve newly projected waveform
#            print input1 , input1.shape
            Atom.proj_score = (np.sum(np.array(
                [Atom.waveform[i] * input1[i] for i in range(self.scale)])))

#            if Atom.projectionScore < 0.000000000001:
#                _Logger.debug("neglecting score")
#                Atom.projectionScore = 0
#
#            if Atom.projectionScore < 0:
#                Atom.projectionScore = - sqrt(-Atom.projectionScore)
#            else:
#                Atom.projectionScore = sqrt(Atom.projectionScore)

            # WORKAROUND
#            Atom.projectionScore = self.maxValue
#            print Atom.projectionScore , self.maxValue
#            print sum(Atom.waveform**2) , sum(input1**2)
            _Logger.debug("new score found of : " + str(Atom.proj_score))
            Atom.mdct_value = Atom.proj_score
            Atom.waveform *= Atom.proj_score
            return Atom

        if len(input1) != len(input2):
            print self.max_frame_idx, self.maxIdx, self.frame_num
            print len(input1), len(input2)
            if debug > 0:
                print "atom in the borders , no timeShift calculated"
            return Atom

        # retrieve optimal timeShift
        if self.use_c_optim:
            scoreVec = np.array([0.0])
#            Atom.timeShift = computeMCLT.project_atom(input1,input2 , scoreVec )
#            print "Is it here?"
            # Atom.timeShift = parallelProjections.project_atom(input1,input2 ,
            # scoreVec , self.scale)
            Atom.time_shift = parallelProjections.project_atom_set(
                input1, input2, fftVec, scoreVec, self.scale, sigIdx)

#            print "Found " ,Atom.timeShift
#            if abs(Atom.timeShift) > ((self.tolerance-1) * Atom.length)/2:
#                print "out of limits: found time shift of" , Atom.timeShift
#                Atom.timeShift = 0
#                return Atom

            self.maxTimeShift = Atom.time_shift
            Atom.time_position += Atom.time_shift

            # retrieve newly projected waveform
            Atom.proj_score = scoreVec[0]
            Atom.mdct_value = Atom.proj_score
            Atom.waveform *= Atom.proj_score

        else:
            print "Block 1345 : Not Implemented !"
            return None

#        print " Reaching here"
        return Atom

    def get_not_optimized_best_atoms(self):
        # hack here : We just compute the value by projecting the waveform onto
        # the signal
        AtomList = []
        for sigIdx in range(self.sigNumber):
            offset = self.scale / 4
            Atom = mdct_atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - offset, 0), self.max_bin_idx, self.residualSignalList[0].fs)
            Atom.frame = self.max_frame_idx
            Atom.synthesize_ifft(1)
            Atom.waveform /= np.sum(Atom.waveform ** 2)

            startSample = ((self.max_frame_idx) * self.scale / 2) - offset
            stopSample = ((self.max_frame_idx + 2) * self.scale / 2) - offset

            locsig = (
                self.enframedDataMatrixList[sigIdx][startSample: stopSample])

            Atom.proj_score = np.sum(np.multiply(Atom.waveform, locsig))
            Atom.mdct_value = Atom.proj_score
            Atom.waveform *= Atom.proj_score
#            print Atom.projectionScore

            AtomList.append(Atom)

        return AtomList


class RandomSetBlock(SetBlock):
    """ Classic MDCT block useful for handling Sets Only the update routine is changed: no need to
        look for the max since it will be done later in the dictionary

        NO ATOM Local Optimization!! But RSSMP
        """

    # parameters
    frame_len = 0
    frame_num = 0

    max_index = 0
    max_value = 0
    max_bin_idx = 0
    max_frame_idx = 0

    # MDCT static window and twiddle parameters
    w_long = None
    enframedDataMatrixList = []

    startingFrameList = None
    endingFrameList = None

    # constructor - initialize residual signals and projection matrix for each
    # of the signals
    def __init__(
        self, length=0, resSignalList=None, frameLen=0, useC=True,
            debug_level=None, nature='sum', tolerance=None):
        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.scale = length
        self.residualSignalList = resSignalList

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residualSignalList == None:
            raise ValueError("no signal given")

        # reference length
        self.length = max(
            [resSignalList[i].length for i in range(len(resSignalList))])

        # number of signals to handle
        self.sigNumber = len(self.residualSignalList)

        # initialize data matrix
        self.enframedDataMatrixList = []
        for i in range(self.sigNumber):
            self.enframedDataMatrixList.append(np.zeros((self.length, 1)))

        for sigIdx in range(self.sigNumber):
            # Important point : assert that all signals have same length!
            if self.residualSignalList[sigIdx].length < self.length:
                print ValueError('Signal ' + str(sigIdx) + ' is too short!' + str(self.residualSignalList[sigIdx].length) + " instead of " + str(self.length))

                # TODO :  just forbid the creation of atom in this zone
                pad = self.length - self.residualSignalList[sigIdx].length
                self.residualSignalList[sigIdx].data = np.concatenate(
                    (self.residualSignalList[sigIdx].data, np.zeros(pad)))
                self.residualSignalList[sigIdx].length += pad

            self.enframedDataMatrixList[
                sigIdx] = self.residualSignalList[sigIdx].data

        self.frame_num = len(self.enframedDataMatrixList[0]) / self.frame_len

        # The projection matrix is unidimensionnal since only one atom will be chosen eventually
# self.projectionMatrix = zeros(len(self.enframedDataMatrixList[0])
# ,complex)
        self.projs_matrix = np.zeros(
            (len(self.enframedDataMatrixList[0]), self.sigNumber), float)

        self.use_c_optim = useC

        if nature == 'sum':
            self.nature = 0
        elif nature == 'median':
            self.nature = 1
        else:
            raise ValueError(' Unrecognized Criterion for selection')

        self.shift_list = [floor(
            (self.scale / 2) * (i - 0.5)) for i in np.random.random(self.length)]

        if tolerance is not None:
            self.tolerance = tolerance
        else:
            self.tolerance = 1

        _Logger.info('new MDCT Setblock constructed size : ' + str(
            self.scale) + ' tolerance of :' + str(self.tolerance))

    def initialize(self):

        # Windowing
        L = self.scale

        self.w_long = np.array(
            [np.sin(float(l + 0.5) * (np.pi / L)) for l in range(L)])

        # twidlle coefficients
        self.pre_twid_vec = np.array(
            [np.exp(n * (-1j) * np.pi / L) for n in range(L)])
        self.post_twid_vec = np.array([np.exp(
            (float(n) + 0.5) * -1j * np.pi * (L / 2 + 1) / L) for n in range(L / 2)])

        # score tree - first version simplified
        self.best_score_tree = np.zeros(self.frame_num)

        # OPTIM -> do pre-twid directly in the windows
        self.locCoeff = self.w_long * self.pre_twid_vec

    def update(self, newResidualList, startFrameList=None, stopFrameList=None, iterationNumber=0):
# print "block update called : " , self.scale  , self.frameNumber , " type
# : " , self.nature
        self.residualSignalList = newResidualList

        self.current_shift = self.shift_list[iterationNumber]

        if startFrameList is None:
            startFrameList = [2] * self.sigNumber

        if stopFrameList is None:
            stopFrameList = [self.frame_num - 3] * self.sigNumber
#
    # MODIF: each signal is updated according to its own limits since
    # sometimes no atoms have been subtracted
        L = self.scale
        for sigIdx in range(self.sigNumber):

            if stopFrameList[sigIdx] < 0:
                stopFrameList[sigIdx] = self.frame_num - 2
            else:
                stopFrameList[sigIdx] = min((stopFrameList[
                                            sigIdx], self.frame_num - 2))
# print "block update called : " , startFrameList[sigIdx]  ,
# stopFrameList[sigIdx] , " type : " , self.scale
            self.enframedDataMatrixList[sigIdx][startFrameList[sigIdx] * L / 2: stopFrameList[sigIdx] * L / 2 + L] = self.residualSignalList[sigIdx].data[startFrameList[sigIdx] * self.frame_len: stopFrameList[sigIdx] * self.frame_len + 2 * self.frame_len]

        self.compute_transform(startFrameList, stopFrameList)

        self.find_max()

    def compute_transform(self, startingFrameList=None, endFrameList=None):
        if self.w_long is None:
            self.initialize()

        # due to later time-shift optimizations , need to ensure nothing is
        # selected too close to the borders!!
        if startingFrameList is None:
            startingFrameList = [2] * self.sigNumber

        if endFrameList is None:
            endFrameList = [self.frame_num - 3] * self.sigNumber

        for i in range(self.sigNumber):
            if endFrameList[i] < 0 or endFrameList[i] > self.frame_num - 3:
                endFrameList[i] = self.frame_num - 3

            if startingFrameList[i] < 2:
                startingFrameList[i] = 2

        # REFACTORED VIOLENT 21/11

        # ALL SIGNALS PROJECTED AND WE KEEP ALL PROJECTIONS, NO OPTIMIZATIONS !!
        for sigIdx in range(0, self.sigNumber):
            localProj = np.array(self.projs_matrix[:, sigIdx])
#            print localProj.shape
            localProj = localProj.reshape((self.projs_matrix.shape[0],))
            parallelProjections.project(self.enframedDataMatrixList[sigIdx],
                                        self.best_score_tree,
                                        localProj,
                                        self.locCoeff,
                                        self.post_twid_vec,
                                        startingFrameList[sigIdx],
                                        endFrameList[sigIdx],
                                        self.scale,
                                        int(self.current_shift))

        # WE NEED TO RECOMPUTE THE GLOBAL SUM SCORE NOW

            self.projs_matrix[:, sigIdx] = np.array(localProj.copy())

# print self.scale , startingFrameList[sigIdx] , endFrameList[sigIdx]
# ,self.frameNumber * self.scale/2
        if self.nature == 0:
            sumOfProjections = np.sum(self.projs_matrix ** 2, 1)
        elif self.nature == 1:
            sumOfProjections = np.median(self.projs_matrix ** 2, 1)
        self.maxIdx = sumOfProjections.argmax()

    def get_not_optimized_best_atoms(self):
        # hack here : We just compute the value by projecting the waveform onto
        # the signal
        AtomList = []
        for sigIdx in range(self.sigNumber):
            offset = self.scale / 4
            Atom = mdct_atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - offset, 0), self.max_bin_idx, self.residualSignalList[0].fs)
            Atom.frame = self.max_frame_idx
            Atom.synthesize_ifft(1)
#            Atom.waveform /= sum(Atom.waveform**2)

            Atom.proj_score = self.projs_matrix[self.maxIdx, sigIdx]
            Atom.mdct_value = Atom.proj_score
            Atom.waveform *= Atom.proj_score
#            print Atom.projectionScore

            Atom.time_shift = self.current_shift
            Atom.time_position -= Atom.time_shift

            AtomList.append(Atom)

        return AtomList


class SetNLLOBlock(SetLOBlock):
    """ Same as a CC block except that atom selection is based on Non Linear Criterias such as median or K-order element
    This implies some structural changes since all the signals projections must be computed before the selection while
    it was performed on the fly in the linear case. Mainly the compute_transform method is different"""

    # constructor - initialize residual signals and projection matrix for each
    # of the signals
    def __init__(
        self, length=0, resSignalList=None, frameLen=0, useC=True,
            debug_level=None, nature='median', tolerance=None, lambd=1):
        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.scale = length
        self.residualSignalList = resSignalList

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residualSignalList == None:
            raise ValueError("no signal given")

        # reference length
        self.length = min(
            [resSignalList[i].length for i in range(len(resSignalList))])

        # number of signals to handle
        self.sigNumber = len(self.residualSignalList)

#        # initialize data matrix
#        self.enframedDataMatrixList = []
#        for i in range(self.sigNumber):
#            self.enframedDataMatrixList.append(zeros( (self.length, 1)))
#
        self.framed_data_matrix = np.zeros(
            (self.sigNumber, self.length), float)

        for sigIdx in range(self.sigNumber):
            # Important point : assert that all signals have same length!
            if self.residualSignalList[sigIdx].length > self.length:
                print ValueError('Signal ' + str(sigIdx) + ' is too long: ' + str(self.residualSignalList[sigIdx].length) + " instead of " + str(self.length))

                self.residualSignalList[sigIdx].data = self.residualSignalList[
                    sigIdx].data[0:self.length]
                # now repad!!
                self.residualSignalList[sigIdx].data[-8192:] = 0
#                # TODO :  just forbid the creation of atom in this zone
#                pad = self.length - self.residualSignalList[sigIdx].length
#                self.residualSignalList[sigIdx].dataVec = concatenate((self.residualSignalList[sigIdx].dataVec , zeros(pad)))
#                self.residualSignalList[sigIdx].length += pad

# self.enframedDataMatrixList[sigIdx] =
# self.residualSignalList[sigIdx].dataVec
            self.framed_data_matrix[sigIdx, :
                                    ] = self.residualSignalList[sigIdx].data

# self.frameNumber = len(self.enframedDataMatrixList[0]) /
# self.frameLength
        self.frame_num = self.framed_data_matrix.shape[1] / self.frame_len
        # The projection matrix is Multidimensional: we need to compute all projections before selecting the right atom
#        self.intermediateProjectionList = []
##
#        for sigIdx in range(self.sigNumber):
#            self.intermediateProjectionList.append(zeros( (self.length, 1)))
        self.intermediateProjection = (
            np.zeros((self.sigNumber, self.length), float))
#
#        self.bestScoreTreeList = zeros((self.frameNumber,self.sigNumber))

#        print self.intermediateProjectionList

        # At the end of the computation the scores should be stored in this
        # matrix
        self.projs_matrix = np.zeros((self.length, 1), float)

        self.use_c_optim = useC

        if nature == 'median':
            self.nature = 0  # the final score
        elif nature == 'penalized':
            self.nature = 1
            if lambd is None:
                lambd = 1
            self.lambd = lambd
        elif nature == 'weighted':
            self.nature = 2
            if lambd is None:
                lambd = 1
            self.lambd = lambd
#        elif nature == 'maximin':
#            self.nature = 2

        else:
            raise ValueError('Unrecognized Criterion for selection')

        if tolerance is not None:
            self.tolerance = tolerance

        _Logger.info('new MDCT Non Linear Setblock constructed size : ' +
                     str(self.scale) + ' tolerance of :' + str(self.tolerance))

    def update(self, newResidualList, startFrameList=None, stopFrameList=None):
# print "block update called : " , self.scale  , self.frameNumber , " type
# : " , self.nature
        self.residualSignalList = newResidualList

        if startFrameList is None:
            startFrameList = [2] * self.sigNumber

        if stopFrameList is None:
            stopFrameList = [self.frame_num - 3] * self.sigNumber
#
    # MODIF: each signal is updated according to its own limits since
    # sometimes no atoms have been subtracted
        L = self.scale
        for sigIdx in range(self.sigNumber):

            if stopFrameList[sigIdx] < 0:
                stopFrameList[sigIdx] = self.frame_num - 2
            else:
                stopFrameList[sigIdx] = min((stopFrameList[
                                            sigIdx], self.frame_num - 2))
# print "block update called : " , startFrameList[sigIdx]  ,
# stopFrameList[sigIdx] , " type : " , self.scale
            self.framed_data_matrix[sigIdx, startFrameList[sigIdx] * L / 2: stopFrameList[sigIdx] * L / 2 + L] = self.residualSignalList[sigIdx].data[startFrameList[sigIdx] * self.frame_len: stopFrameList[sigIdx] * self.frame_len + 2 * self.frame_len]

        self.compute_transform(startFrameList, stopFrameList)

        self.find_max()

    def getAdaptedAtom(self, sigIdx, debug, fftVec):
        # hack here : let us project the atom waveform on the neighbouring signal in the FFt domain,
        # so that we can find the maximum correlation and best adapt the time-shift
        # Construct prototype atom

        Atom = mdct_atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residualSignalList[0].fs)
        Atom.frame = self.max_frame_idx

        Atom.mdct_value = 1.0
        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom(value=1)
        Atom.time_shift = 0
        Atom.proj_score = 0.0

        # width of tolerance : equals to the width of the area scanned on left
        # and right to find optimal position
        HalfWidth = (self.tolerance - 1) * self.scale / 2

        # do not forget to compensate for the quarter window offsetted
        offset = self.scale / 4
        startSample = (
            (self.max_frame_idx) * self.scale / 2) - HalfWidth - offset
        stopSample = (
            (self.max_frame_idx + 2) * self.scale / 2) + HalfWidth - offset

#        startSample = ((self.maxFrameIdx -1.5) * self.scale/2)
#        stopSample = ((self.maxFrameIdx +2.5) * self.scale/2)

#        print "BLOCK line 1302" ,HalfWidth , startSample , stopSample , stopSample - startSample
        # retrieve the corresponding data
        input1 = self.framed_data_matrix[sigIdx, startSample: stopSample]

        # surround the canonical atom waveform by zeroes
        input2 = np.concatenate((np.concatenate(
            (np.zeros(HalfWidth), Atom.waveform)), np.zeros(HalfWidth)))

        if len(input1) != len(input2):
            print self.max_frame_idx, self.maxIdx, self.frame_num
            print len(input1), len(input2)
            # if debug>0:
            print "atom in the borders , no timeShift calculated"
            return Atom

        # retrieve optimal timeShift
        if self.use_c_optim:
            scoreVec = np.array([0.0])
#            Atom.timeShift = computeMCLT.project_atom(input1,input2 , scoreVec )
#            print "Is it here?"
            # Atom.timeShift = parallelProjections.project_atom(input1,input2 ,
            # scoreVec , self.scale)
            Atom.time_shift = parallelProjections.project_atom_set(
                input1, input2, fftVec, scoreVec, self.scale, sigIdx)

#            print "Found " ,Atom.timeShift
#            if abs(Atom.timeShift) > ((self.tolerance-1) * Atom.length)/2:
#                print "out of limits: found time shift of" , Atom.timeShift
#                Atom.timeShift = 0
#                return Atom

            self.maxTimeShift = Atom.time_shift
            Atom.time_position += Atom.time_shift

            # retrieve newly projected waveform
            Atom.proj_score = scoreVec[0]
            Atom.mdct_value = Atom.proj_score
            Atom.waveform *= Atom.proj_score

        else:
            print "Block 1722 : Not Implemented !"
            return None

#        print " Reaching here"
        return Atom

    def compute_transform(self, startingFrameList=None, endFrameList=None):
        if self.w_long is None:
            self.initialize()

        # due to later time-shift optimizations , need to ensure nothing is
        # selected too close to the borders!!
        if startingFrameList is None:
            startingFrameList = [2] * self.sigNumber

        if endFrameList is None:
            endFrameList = [self.frame_num - 3] * self.sigNumber

        for i in range(self.sigNumber):
            if endFrameList[i] < 0 or endFrameList[i] > self.frame_num - 3:
                endFrameList[i] = self.frame_num - 3

            if startingFrameList[i] < 2:
                startingFrameList[i] = 2

        # For each signal: update the projection matrix in the given boundaries
        # with the appropriate method
        startFrame = min(startingFrameList)
        endFrame = max(endFrameList)
#        for sigIdx in range(self.sigNumber):
        # Refactored : everything is handled in C code now
        parallelProjections.project_mclt_NLset(self.framed_data_matrix,
                                               self.best_score_tree,
                                               self.intermediateProjection,
                                               self.projs_matrix,
                                               self.locCoeff,
                                               self.post_twid_vec,
                                               startFrame,
                                               endFrame,
                                               self.scale,
                                               self.nature)

# print self.scale ,"Blocks 1575:"
# ,self.intermediateProjectionList[sigIdx]

        # TODO passer tout ca en C !!!
        # Now in a second phase : compute the projection matrix according to the
        # chosen method:


#        if self.nature == 0:
#            A = concatenate(self.intermediateProjectionList, axis = 1)
##            A = self.intermediateProjectionList
#            self.projectionMatrix[startFrame*self.frameLength : endFrame * self.frameLength] = median(abs(A[startFrame*self.frameLength : endFrame * self.frameLength,:]) , axis=1)
#
#
#
#        elif self.nature ==1:
#            A = concatenate(self.intermediateProjectionList, axis = 1)
##            A = self.intermediateProjectionList
#            B = abs(A[startFrame*self.frameLength : endFrame * self.frameLength,:])
#            self.projectionMatrix[startFrame*self.frameLength : endFrame * self.frameLength] = sum(B , axis=1)
#
#            # add the penalty : lambda times the sum of the differences
#            for i in range(self.sigNumber):
#                for j in range(i+1,self.sigNumber):
##                    print i,j
#                    diff = self.lambd * abs(B[:,i] - B[:,j])
##                    print diff
#                    self.projectionMatrix[startFrame*self.frameLength : endFrame * self.frameLength] += diff
#        # case weighted: multiply the sum by the flatness measure
#        elif self.nature ==2:
#            A = concatenate(self.intermediateProjectionList, axis = 1)
##            A = self.intermediateProjectionList
#            B = abs(A[startFrame*self.frameLength : endFrame * self.frameLength,:])
#
#            flatnessMes = gmean(B,axis=1)/mean(B,axis=1)
#            # replace NaNs with 0
#            self.projectionMatrix[startFrame*self.frameLength : endFrame * self.frameLength] = multiply(nan_to_num(flatnessMes),sum(B , axis=1))
#
#
#        for i in range(self.frameNumber-3):
#            self.bestScoreTree[i] = (self.projectionMatrix[i * self.frameLength : (i+1) * self.frameLength]).max()
#
#            print "blocks 1606:",self.bestScoreTree[i] , self.scale
