#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

#
#                       PyMP.mdct.block
#
#
#
# M. Moussallam                             Created on Nov 12, 2012
# -----------------------------------------------------------------------


"""

Module mdct.block
=====================

This class inherits from :class:`.BaseBlock` and is used to represent and manipulate MDCT atoms.
:class:`mdct.Atom` Objects can either be constructed or recovered from Xml using :func:`mdct.Atom.fromXml`


MDCT Blocks are the objects encapsulating the actual atomic projections and the selection step of MP

This module describes 3 kind of blocks:
    - :class:`Block`  is a block based on the standard MDCT transform. Which means atoms localizations
                 are constrained by the scale of the transform
                 The atom selected is simple the one that maximizes the correlation with the residual
                 it is directly indexes by the max absolute value of the MDCT bin
                 No further optimization of the selected atom is performed

    - :class:`LOBlock`   is a block that performs a local optimization of the time localization of the selected atom

    - :class:`FullBlock`  this block simulates a dictionary where atoms at all time localizations are available
                  This kind of dictionary can be thought of as a Toeplitz matrix
                  BEWARE: memory consumption and CPU load will be very high! only use with very small signals
                  (e.g. 1024 samples or so) Fairly intractable at higher dimensions
"""

import sys

#from numpy.fft import fft
import numpy as np

from math import  floor
from cmath import exp
import matplotlib.pyplot as plt

from .. import win_server
from .. import  log
#from ..tools import Xcorr  # , Misc
from .. import parallelProjections
from ..base import BaseBlock
from . import atom



# declare global PyWinServer shared by all MDCT blocks instances
#global _PyServer
#, _Logger
_PyServer = win_server.get_server()
_Logger = log.Log('MDCTBlock', level=0)


class Block(BaseBlock):
    """ This class is a basic MDCT block. It handles a set of routines that allows greedy
        decomposition algorithms (MP, OMP , GP ...) to iteratively project the ongoing residual
        onto a specified MDCT base and to retrieve the maximum inner product

         IMPORTANT: the computation of the inner products relies on a C-written module parallelProjections
                 that needs being compiled first on whatever your plateform.
                 Run the test file: parallelProjections/install_and_test.py in order to compile it
                 and perform some basic execution tests

         """

    #members
    nature = 'MDCT'
    framed_data_matrix = None
    projs_matrix = None
    frame_len = 0
    frame_num = 0

    max_index = 0
    max_value = 0
    max_bin_idx = 0
    max_frame_idx = 0

    # MDCT static window and twiddle parameters
    w_long = None
    w_ledge = None
    w_redge = None

    pre_twid_vec = None
    post_twid_vec = None

    # store fft matrix for later Xcorr purposes
    fftMat = None
    fft = None
    # Score tree
    best_score_tree = None

    # optim?
    use_c_optim = True
    
    # DEPRECATED
    HF = False
    HFlimit = 0.1

    windowType = None

    # constructor - initialize residual signal and projection matrix
    def __init__(self, length=0, resSignal=None, frameLen=0, useC=True, forceHF=False, debug_level=None):
        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.scale = length
        self.residual_signal = resSignal

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residual_signal == None:
            raise ValueError("no signal given")

        self.framed_data_matrix = self.residual_signal.data
        self.frame_num = len(self.framed_data_matrix) / self.frame_len
        self.projs_matrix = np.zeros(len(self.framed_data_matrix))
        self.use_c_optim = useC
        self.HF = forceHF
        _Logger.info('new MDCT block constructed size : ' + str(self.scale))
#        self.projs_matrix = zeros((self.frameNumber , self.frameLength))

    # compute mdct of the residual and instantiate various windows and twiddle
    # coefficients
    def initialize(self):
        """ Compute mdct of the residual and instantiate various windows and twiddle coefficients"""
        #Windowing
        L = self.scale

        self.w_long = np.array([np.sin(float(l + 0.5) * (np.pi / L)) for l in range(L)])

        # twidlle coefficients
        self.pre_twid_vec = np.array([exp(n * (-1j) * np.pi / L) for n in range(L)])
        self.post_twid_vec = np.array(
            [exp((float(n) + 0.5) * -1j * np.pi * (L / 2 + 1) / L) for n in range(L / 2)])

        if self.windowType == 'half1':
            self.w_long[0:L / 2] = 0
            # twidlle coefficients
            self.pre_twid_vec[0:L / 2] = 0
#        self.fftMat = zeros((self.scale , self.frameNumber) , complex)
#        self.normaCoeffs = sqrt(1/float(L))

        # score tree - first version simplified
        self.best_score_tree = np.zeros(self.frame_num)

        if self.HF:
            self.bestScoreHFTree = np.zeros(self.frame_num)

        # OPTIM -> do pre-twid directly in the windows
        self.locCoeff = self.w_long * self.pre_twid_vec

    # Search among the inner products the one that maximizes correlation
    # the best candidate for each frame is already stored in the best_score_tree
    def find_max(self):
        """Search among the inner products the one that maximizes correlation
        the best candidate for each frame is already stored in the best_score_tree """
        treeMaxIdx = self.best_score_tree.argmax()

        maxIdx = abs(self.projs_matrix[treeMaxIdx * self.scale /
            2: (treeMaxIdx + 1) * self.scale / 2]).argmax()
        self.maxIdx = maxIdx + treeMaxIdx * self.scale / 2
        self.max_value = self.projs_matrix[self.maxIdx]

#        print "block find_max called : " , self.maxIdx , self.max_value

        if self.HF:
            treemaxHFidx = self.bestScoreHFTree.argmax()
            maxHFidx = abs(self.projs_matrix[(treemaxHFidx + self.HFlimit)
                * self.scale / 2: (treemaxHFidx + 1) * self.scale / 2]).argmax()

            self.maxHFIdx = maxHFidx + (
                treemaxHFidx + self.HFlimit) * self.scale / 2
            self.maxHFValue = self.projs_matrix[self.maxHFIdx]

    # construct the atom that best correlates with the signal
    def get_max_atom(self, HF=False):
        """ construct the atom that best correlates with the signal"""

        if not HF:
            self.max_frame_idx = floor(self.maxIdx / (0.5 * self.scale))
            self.max_bin_idx = self.maxIdx - self.max_frame_idx * (
                0.5 * self.scale)
        else:
            self.max_frame_idx = floor(self.maxHFIdx / (0.5 * self.scale))
            self.max_bin_idx = self.maxHFIdx - self.max_frame_idx * (
                0.5 * self.scale)
        Atom = atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residual_signal.fs)
        Atom.frame = self.max_frame_idx
#        print self.max_bin_idx, Atom.reducedFrequency
        if not HF:
            Atom.mdct_value = self.max_value
        else:
            Atom.mdct_value = self.maxHFValue

        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom()

        if HF:
            Atom.waveform = Atom.waveform * (self.maxHFValue / self.max_value)
        return Atom

    #
    def update(self, new_res_signal, startFrameIdx=0, stopFrameIdx=-1):
        """ update the part of the residual that has been changed and update inner products    """
# print "block update called : " , self.scale , new_res_signal.length ,
# self.frameNumber
        self.residual_signal = new_res_signal

        if stopFrameIdx < 0:
            endFrameIdx = self.frame_num - 1
        else:
            endFrameIdx = stopFrameIdx

        L = self.scale

        self.framed_data_matrix[startFrameIdx * L / 2: endFrameIdx * L / 2 + L] = self.residual_signal.data[startFrameIdx * self.frame_len: endFrameIdx * self.frame_len + 2 * self.frame_len]

        self.compute_transform(startFrameIdx, stopFrameIdx)

        self.find_max()

    # inner product computation through MDCT
    def compute_transform(self, startingFrame=1, endFrame=-1):
        """ inner product computation through MDCT """
        if self.w_long is None:
            self.initialize()

        if endFrame < 0:
            endFrame = self.frame_num - 1

        # debug -> changed from 1: be sure signal is properly zero -padded
        if startingFrame < 1:
            startingFrame = 1

        # Wrapping C code call for fast implementation
#        if self.use_c_optim:
        try:
            parallelProjections.project(self.framed_data_matrix, self.best_score_tree,
                                             self.projs_matrix,
                                             self.locCoeff,
                                             self.post_twid_vec,
                                             startingFrame,
                                             endFrame,
                                             self.scale, 0)

        except SystemError:
            print sys.exc_info()[0]
            print sys.exc_info()[1]
            raise
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
#        else:
#            try:
#                import fftw3
#            except ImportError:
#                print '''Impossible to load fftw3, you need to use the C extension library or have
#                        a local installation of the python fftw3 wrapper '''
#                return
#        # create a forward fft plan, since fftw is not faster for computing
#        # DCT's no need to use it for that
#            L = self.scale
#            K = L / 2
#            T = K / 2
#            normaCoeffs = sqrt(2 / float(K))
#            if self.fft is None:
#                self.inputa = self.framed_data_matrix[K -
#                     T: K + L - T].astype(complex)
#                self.outputa = None
#                self.fft = fftw3.Plan(self.inputa, self.
#                    outputa, direction='forward', flags=['estimate'])
#
#            self.project(startingFrame, endFrame, L, K, T, normaCoeffs)

        if self.HF:
            for i in range(startingFrame, endFrame):
                self.bestScoreHFTree[i] = abs(self.projs_matrix[
                    (i + self.HFlimit) * self.scale / 2: (i + 1) * self.scale / 2]).max()

    # UPDATE: need to keep this slow version for non C99 compliant systems
    # (windows)
#    def project(self, startingFrame, endFrame, L,  K, T, normaCoeffs):
#
#        for i in range(startingFrame, endFrame):
#            # initialize input data
#            self.inputa[:] = 0
#            self.inputa += self.framed_data_matrix[i * K - T: i *
#                K + L - T] * self.locCoeff
#
#            #compute fft
#            self.fft()
#
#            # post-twiddle and store for max search
#            self.projs_matrix[i * K: (i + 1) *
#                K] = normaCoeffs * (self.inputa[0:K] * self.post_twid_vec).real
## self.projs_matrix[i , :] = normaCoeffs*(self.inputa[0:K]*
## self.post_twid_vec).real
#
##            # store new max score in tree
#            self.best_score_tree[i] = abs(self.projs_matrix[
#                i * K: (i + 1) * K]).max()

    # synthesizes the best atom through ifft computation (much faster than
    # closed form)
    def synthesize_atom(self, value=None):
        """ synthesizes the best atom through ifft computation (much faster than closed form)
            New version uses the PyWinServer to serve waveforms"""
        ###################" new version ############"
#        global _PyServer
#        print len(_PyServer.Waveforms)
        if value is None:
            return self.max_value * _PyServer.get_waveform(self.scale, self.max_bin_idx)
        else:
            return value * _PyServer.get_waveform(self.scale, self.max_bin_idx)
        ###################  old version ############
#        temp = zeros(2*self.scale)
#        temp[self.scale/2 + self.max_bin_idx] = self.max_value
#        waveform = zeros(2*self.scale)
# Number of frames : only 4 we need zeroes on the border before overlap-adding
##        P = 4
#        L = self.scale
#        K = L/2
#        y = zeros(L, complex)
#        x = zeros(L, complex)
#        # we're note in Matlab anymore, loops are more straight than indexing
#        for i in range(1,3):
#            y[0:K] = temp[i*K : (i+1)*K]
#
#            # do the pre-twiddle
#            y = y *  self.pre_i_twidVec
#
#            # compute ifft
#            x = ifft(y)
#
#            # do the post-twiddle
#            x = x * self.post_i_twidVec
#
#            x = 2*sqrt(1/float(L))*L*x.real*self.w_long
#
#            # overlapp - add
#            waveform[i*K : i*K +L] = waveform[i*K : i*K +L] +  x
#
#        # scrap zeroes on the borders
#        return waveform[L/2:-L/2]

    def plot_proj_matrix(self):
        plt.figure()
#        plt.subplot(211)
#        plt.plot(self.best_score_tree)
#        plt.subplot(212)
        plt.plot(self.projs_matrix)
        plt.title("Block-" + str(self.scale) + " best Score of" + str(self.max_value)
                  + " p :" + str(self.max_frame_idx) + " , k: " + str(self.max_bin_idx))


class LOBlock(Block):
    """ Class that inherit classic MDCT block class and deals with local optimization
        This is the main class for differentiating LOMP from MP """

    # Typically all attributes are the same than mother class:
    maxTimeShift = 0
    adjustTimePos = True
    fftplan = None
    # constructor - initialize residual signal and projection matrix

    def __init__(self, length=0, resSignal=None, frameLen=0, tinvOptim=True, useC=True, forceHF=False):
        self.scale = length
        self.residual_signal = resSignal
        self.adjustTimePos = tinvOptim
        self.use_c_optim = useC
        self.HF = forceHF

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residual_signal == None:
            raise ValueError("no signal given")

        self.framed_data_matrix = self.residual_signal.data
        self.frame_num = len(self.framed_data_matrix) / self.frame_len

        # only difference , here, we keep the complex values for the cross
        # correlation
#        self.projs_matrix = zeros(len(self.framed_data_matrix) , complex)
        self.projs_matrix = np.zeros(len(self.framed_data_matrix), float)

# only the update method is interesting for us : we're hacking it to experiment
#    def update(self , newResidual , startFrameIdx=0 , stopFrameIdx=-1):
#        self.residual_signal = newResidual
#
#        if stopFrameIdx <0:
#            endFrameIdx = self.frameNumber -1
#        else:
#            endFrameIdx = stopFrameIdx
#        L = self.scale
#
#        # update residual signal
# self.framed_data_matrix[startFrameIdx*L/2 : endFrameIdx*L/2 + L] =
# self.residual_signal.dataVec[startFrameIdx*self.frameLength :
# endFrameIdx*self.frameLength + 2*self.frameLength]
#
#        # TODO changes here
##        self.computeMDCT(startFrameIdx , stopFrameIdx)
#        self.computeMCLT(startFrameIdx , stopFrameIdx)
#
#        # TODO changes here
#        self.find_max()
    # inner product computation through MDCT
    def compute_transform(self, startingFrame=1, endFrame=-1):
        if self.w_long is None:
            self.initialize()

        # due to later time-shift optimizations , need to ensure nothing is
        # selected too close to the borders!!

        if endFrame < 0 or endFrame > self.frame_num - 3:
            endFrame = self.frame_num - 3

        if startingFrame < 2:
            startingFrame = 2
        # new version: C binding
#        if self.use_c_optim:
        parallelProjections.project_mclt(self.framed_data_matrix, self.best_score_tree,
                                             self.projs_matrix,
                                             self.locCoeff,
                                             self.post_twid_vec,
                                             startingFrame,
                                             endFrame,
                                             self.scale)
#        else:
#            L = self.scale
#            K = L / 2
#            T = K / 2
#            normaCoeffs = sqrt(2 / float(K))
#
#            locenframedDataMat = self.framed_data_matrix
##            locfftMat = self.fftMat
##            locprojMat = self.projs_matrix
#
#            preTwidCoeff = self.locCoeff
#            postTwidCoeff = self.post_twid_vec
#
#            # Bottleneck here !! need to fasten this loop : do it by Matrix
#            # technique? or bind to a C++ file?
#            for i in range(startingFrame, endFrame):
#                x = locenframedDataMat[i * K - T: i * K + L - T]
#                if len(x) != L:
#                    x = np.zeros(L, complex)
#
#                # do the pre-twiddle
#                x = x * preTwidCoeff
#
#                # compute fft
##                locfftMat[: , i] = fft(x , L)
#
#                # post-twiddle
##                y = locfftMat[0:K , i] * postTwidCoeff
#                y = (fft(x, L)[0:K]) * postTwidCoeff
#    #            y = self.doPretwid(locfftMat[0:K , i], postTwidCoeff)
#
#                # we work with MCLT now
#                self.projs_matrix[i * K: (i + 1) * K] = normaCoeffs * y
#
#                # store new max score in tree
#                self.best_score_tree[i] = abs(self.
#                    projs_matrix[i * K: (i + 1) * K]).max()

#        if self.HF:
#            for i in range(startingFrame , endFrame):
# self.bestScoreHFTree[i] =
# abs(self.projs_matrix[(i+self.HFlimit)*self.scale/2 :
# (i+1)*self.scale/2]).max()
#

    # construct the atom that best correlates with the signal
    def get_max_atom(self, debug=0):

        self.max_frame_idx = floor(self.maxIdx / (0.5 * self.scale))
        self.max_bin_idx = self.maxIdx - self.max_frame_idx * (0.5 * self.scale)

        # hack here : let us project the atom waveform on the neighbouring
        # signal in the FFt domain,
        # so that we can find the maximum correlation and best adapt the time-
        # shift
        Atom = atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residual_signal.fs)
        Atom.frame = self.max_frame_idx

        # re-compute the atom amplitude for IMDCT
#        if self.max_value.real < 0:
#            self.max_value = -abs(self.max_value)
#        else:
#            self.max_value = abs(self.max_value)

        Atom.mdct_value = self.max_value
        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom(value=1)
        Atom.time_shift = 0
        Atom.proj_score = 0.0

        input1 = self.framed_data_matrix[(self.max_frame_idx - 1.5) *
             self.scale / 2: (self.max_frame_idx + 2.5) * self.scale / 2]
        input2 = np.concatenate((np.concatenate(
            (np.zeros(self.scale / 2), Atom.waveform)), np.zeros(self.scale / 2)))

        # debug cases: sometimes when lot of energy on the borders , pre-echo
        # artifacts tends
        # to appears on the border and can lead to seg fault if not monitored
        # therefore we need to prevent any time shift leading to positions
        # outside of original signals
        # ie in the first and last frames.
#        if debug>0:
#            print self.max_frame_idx , self.maxIdx , self.frameNumber
#            if self.max_frame_idx ==1:
#                plt.figure()
#                plt.plot(Atom.waveform)
# plt.plot(self.framed_data_matrix[0 : Atom.timePosition + Atom.length],'r')
#                plt.show()
#
# print (self.max_frame_idx-1.5)  * self.scale/2 , (self.max_frame_idx+2.5)  *
# self.scale/2
#            print len(input1) , len(input2)
        if len(input1) != len(input2):
            print self.max_frame_idx, self.maxIdx, self.frame_num
            print len(input1), len(input2)
            #if debug>0:
            print "atom in the borders , no timeShift calculated"
            return Atom

        # retrieve additional timeShift
#        if self.use_c_optim:
        scoreVec = np.array([0.0])
        Atom.time_shift = parallelProjections.project_atom(
            input1, input2, scoreVec, self.scale)

        if abs(Atom.time_shift) > Atom.length / 2:
            print "out of limits: found time shift of", Atom.time_shift
            Atom.time_shift = 0
            return Atom

        self.maxTimeShift = Atom.time_shift
        Atom.time_position += Atom.time_shift

        # retrieve newly projected waveform
        Atom.proj_score = scoreVec[0]
        Atom.waveform *= Atom.proj_score
#            Atom.waveform = input2[self.scale/2:-self.scale/2]
#        else:
#            sigFft = fft(self.framed_data_matrix[(self.max_frame_idx - 1.5) * self.scale /
#                2: (self.max_frame_idx + 2.5) * self.scale / 2], 2 * self.scale)
#            atomFft = fft(np.concatenate((np.concatenate((np.zeros(self.scale / 2),
#                 Atom.waveform)), np.zeros(self.scale / 2))), 2 * self.scale)
#
#            Atom.time_shift, score = Xcorr.GetMaxXCorr(
#                atomFft, sigFft, maxlag=self.scale / 2)
#            self.maxTimeShift = Atom.time_shift
#            Atom.proj_score = score
#    # print "found correlation max of " ,
#    # float(score)/sqrt(2/float(self.scale))
#
#            # CAses That might happen: time shift result in choosing another
#            # atom instead
#            if abs(Atom.time_shift) > Atom.length / 2:
#                print "out of limits: found time shift of", Atom.time_shift
#                Atom.time_shift = 0
#                return Atom
#
#            Atom.time_position += Atom.time_shift
#
#            # now let us re-project the atom on the signal to adjust it's
#            # energy: Only if no pathological case
#
#            # TODO optimization : pre-compute energy (better: find closed form)
#            if score < 0:
#                Atom.amplitude = -sqrt(-score)
#                Atom.waveform = (
#                    -sqrt(-score / sum(Atom.waveform ** 2))) * Atom.waveform
#            else:
#                Atom.amplitude = sqrt(score)
#                Atom.waveform = (
#                    sqrt(score / sum(Atom.waveform ** 2))) * Atom.waveform
# projOrtho = sum(Atom.waveform * self.residual_signal.dataVec[Atom.timePosition
# : Atom.timePosition + Atom.length])

#        if score <0:
#            Atom.amplitude = -1
#            Atom.waveform = -Atom.waveform

        return Atom

    def plot_proj_matrix(self):
        maxFrameIdx = floor(self.maxIdx / (0.5 * self.scale))
        maxBinIdx = self.maxIdx - maxFrameIdx * (0.5 * self.scale)
        plt.figure()
#        plt.subplot(211)
#        plt.plot(self.best_score_tree)
#        plt.subplot(212)
        plt.plot(abs(self.projs_matrix))
        plt.title("Block-" + str(self.scale) + " best Score of" + str(abs(self.max_value)) + " at " + str(self.maxIdx)
                  + " p :" + str(maxFrameIdx)
                  + " , k: " + str(maxBinIdx)
                  + " , l: " + str(self.maxTimeShift))


class FullBlock(Block):
    """ Class that inherit classic MDCT block class and but contains all time localizations """
    # parameters
    maxKidx = 0
    maxLidx = 0

    # constructor - initialize residual signal and projection matrix
    def __init__(self, length=0, resSignal=None, frameLen=0):
        self.scale = length
        self.residual_signal = resSignal

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residual_signal == None:
            raise ValueError("no signal given")

        self.framed_data_matrix = self.residual_signal.data
        self.frame_num = len(self.framed_data_matrix) / self.frame_len

        # ok here the mdct will be computed for every possible time shift
        # so allocate enough memory for all the transforms

        # here the projection matrix is actually a list of K by K matrices
        self.projs_matrix = dict()
        self.best_score_tree = dict()

        # initialize an empty matrix of size len(data) for each shift
        for i in range(self.scale / 2):
#            self.projs_matrix[i] = zeros((self.scale/2 , self.scale/2))
            self.projs_matrix[i] = np.zeros(len(self.framed_data_matrix))
            self.best_score_tree[i] = np.zeros(self.frame_num)

    def initialize(self):

        #Windowing
        L = self.scale

        self.w_long = np.array([np.sin(float(l + 0.5) * (np.pi / L)) for l in range(L)])

        # twidlle coefficients
        self.pre_twid_vec = np.array([exp(n * (-1j) * np.pi / L) for n in range(L)])
        self.post_twid_vec = np.array(
            [exp((float(n) + 0.5) * -1j * np.pi * (L / 2 + 1) / L) for n in range(L / 2)])

        if self.windowType == 'half1':
            self.w_long[0:L / 2] = 0
            # twidlle coefficients
            self.pre_twid_vec[0:L / 2] = 0

        # OPTIM -> do pre-twid directly in the windows
        self.locCoeff = self.w_long * self.pre_twid_vec

    # The update method is nearly the same as CCBlock
    def update(self, new_res_signal, startFrameIdx=0, stopFrameIdx=-1):
        self.residual_signal = new_res_signal

        if stopFrameIdx < 0:
            endFrameIdx = self.frame_num - 1
        else:
            endFrameIdx = stopFrameIdx

        # debug : recompute absolutely all the products all the time

        L = self.scale

# print "Updating Block " , L , " from frame " , startFrameIdx , " to " ,
# endFrameIdx

        # update residual signal
        self.framed_data_matrix[startFrameIdx * L / 2: endFrameIdx * L / 2 + L] = self.residual_signal.data[startFrameIdx * self.frame_len: endFrameIdx * self.frame_len + 2 * self.frame_len]

        # TODO changes here
        self.compute_transform(startFrameIdx, stopFrameIdx)

        # TODO changes here
        self.find_max()

    # inner product computation through MCLT with all possible time shifts
    def compute_transform(self, startingFrame=1, endFrame=-1):
        if self.w_long is None:
            self.initialize()

        if endFrame < 0:
            endFrame = self.frame_num - 1

        if startingFrame < 1:
            startingFrame = 1

        startingFrame = 1
        endFrame = self.frame_num - 1

        L = self.scale
        K = L / 2
#        T = K/2
#        normaCoeffs = sqrt(2/float(K))
#        print startingFrame , endFrame
        for l in range(-K / 2, K / 2, 1):
            parallelProjections.project(self.framed_data_matrix,
                                                 self.best_score_tree[l + K / 2],
                                                 self.projs_matrix[
                                                     l + K / 2],
                                                 self.locCoeff,
                                                 self.post_twid_vec,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale, l)

    def find_max(self):
        K = self.scale / 2
        bestL = 0
        treeMaxIdx = 0
        bestSCore = 0
        for l in range(-K / 2, K / 2, 1):
            if self.best_score_tree[l + K / 2].max() > bestSCore:
                bestSCore = self.best_score_tree[l + K / 2].max()
                treeMaxIdx = self.best_score_tree[l + K / 2].argmax()
                bestL = l

        maxIdx = abs(self.projs_matrix[bestL + K / 2]).argmax()

        self.maxLidx = bestL

        self.maxIdx = maxIdx
        self.max_frame_idx = treeMaxIdx
        self.max_value = self.projs_matrix[bestL + K / 2][maxIdx]

#        print "Max Atom : " , self.maxIdx , self.maxLidx , self.max_value
    # construct the atom that best correlates with the signal
    def get_max_atom(self):
        self.max_bin_idx = self.maxIdx - self.max_frame_idx * (0.5 * self.scale)

        Atom = atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residual_signal.fs)
        Atom.frame = self.max_frame_idx

        # re-compute the atom amplitude for IMDCT
        Atom.mdct_value = self.max_value

        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom()

        Atom.time_shift = -self.maxLidx  # + self.scale/4
        self.maxTimeShift = Atom.time_shift

        Atom.time_position += Atom.time_shift

        return Atom

    def plot_proj_matrix(self):
        ''' For debug purposes only... '''

        plt.figure()
#        plt.subplot(211)
        plt.plot(abs(self.projs_matrix[self.maxLidx + self.scale / 4]))
#        plt.title("reference l=0 , best k of "
#                  + str(bestK)
# + " max score of : " +
# str(abs(self.projs_matrix[self.maxIdx][bestK,:]).max())
# + " at l: " + str(abs(self.projs_matrix[self.maxIdx][bestK,:]).argmax() -
# self.scale/4) )
#        plt.subplot(212)
#        plt.imshow(abs(self.projs_matrix[self.maxIdx]) ,
#                   aspect='auto', cmap = cm.get_cmap('greys') ,
#                   extent=(-self.scale/4 , self.scale/4, self.scale/2, 0) ,
#                   interpolation = 'nearest')
        plt.title("Block-" + str(self.scale) + " best Score of" + str(self.max_value)
                  + " p :" + str(self.max_frame_idx)
#                  +" , k: "+ str(self.maxKidx)
                  + " , l: " + str(self.maxLidx))


class SpreadBlock(Block):
    ''' Spread MP is a technique in which you penalize the selection of atoms near existing ones
        in a  predefined number of iterations, or you specify a perceptual TF masking to enforce the selection of
        different features. The aim is to have maybe a loss in compressibility in the first iteration but
        to have the most characteristics / discriminant atoms be chosen in the first iterations '''

    # parameters
    distance = None
    mask = None
    maskSize = None
    penalty = None

    def __init__(self, length=0, resSignal=None, frameLen=0, useC=True, forceHF=False,
                 debug_level=None, penalty=0.5, maskSize=1):

        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.scale = length
        self.residual_signal = resSignal

        if frameLen == 0:
            self.frame_len = length / 2
        else:
            self.frame_len = frameLen
        if self.residual_signal == None:
            raise ValueError("no signal given")

        self.framed_data_matrix = self.residual_signal.data
        self.frame_num = len(self.framed_data_matrix) / self.frame_len
        self.projs_matrix = np.zeros(len(self.framed_data_matrix))
        self.use_c_optim = useC
        self.HF = forceHF
        _Logger.info('new MDCT block constructed size : ' + str(self.scale))

        self.penalty = penalty
        # initialize the mask: so far no penalty
        self.mask = np.ones(len(self.framed_data_matrix))
        self.maskSize = maskSize

    def find_max(self, it=-1):
        ''' Apply the mask to the projection before choosing the maximum '''

        # cannot use the tree indexing ant more... too bad
#        treeMaxIdx = self.best_score_tree.argmax()
        self.projs_matrix *= self.mask
#        print self.framed_data_matrix.shape, self.mask.shape

#        plt.figure()
# plt.imshow(reshape(self.mask,(self.frameNumber,self.scale/2)),interpolation='
# nearest',aspect='auto')
        self.maxIdx = np.argmax(abs(self.projs_matrix))

#        print self.maxIdx

# maxIdx = abs(self.projs_matrix[treeMaxIdx*self.scale/2 :
# (treeMaxIdx+1)*self.scale/2]).argmax()
#        self.maxIdx = maxIdx + treeMaxIdx*self.scale/2
        self.max_value = self.projs_matrix[self.maxIdx]

    def get_max_atom(self, HF=False):
        self.max_frame_idx = floor(self.maxIdx / (0.5 * self.scale))
        self.max_bin_idx = self.maxIdx - self.max_frame_idx * (0.5 * self.scale)

        # update the mask : penalize the choice of an atom overlapping in time
        # and or frequency
        for i in range(-self.maskSize, self.maskSize + 1, 1):
            self.mask[((self.max_frame_idx + i) * self.scale / 2) + (self.max_bin_idx - self.maskSize): ((self.max_frame_idx + i) * self.scale / 2) + (self.max_bin_idx + self.maskSize)] = self.penalty

        # proceed as usual
        Atom = atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residual_signal.fs)
        Atom.frame = self.max_frame_idx

        Atom.mdct_value = self.max_value

        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom()

        return Atom
