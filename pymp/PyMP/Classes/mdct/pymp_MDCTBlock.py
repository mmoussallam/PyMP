'''
#                                                                            
#                       Classes.mdct.pymp_MDCTBlock                                   
#                                                                            
#                                                
#                                                                            
# M. Moussallam                             Created on Nov 12, 2012  
# -----------------------------------------------------------------------
#                                                                            
#                                                                            
#  This program is free software; you can redistribute it and/or             
#  modify it under the terms of the GNU General Public License               
#  as published by the Free Software Foundation; either version 2            
#  of the License, or (at your option) any later version.                    
#                                                                            
#  This program is distributed in the hope that it will be useful,          
#  but WITHOUT ANY WARRANTY; without even the implied warranty of            
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             
#  GNU General Public License for more details.                              
#                                                                            
#  You should have received a copy of the GNU General Public License         
#  along with this program; if not, write to the Free Software               
#  Foundation, Inc., 59 Temple Place - Suite 330,                            
#  Boston, MA  02111-1307, USA.                                              
# 

MDCT Blocks are the objects encapsulating the actual atomic projections and the selection step of MP 
                                                                           
This module describes 3 kind of blocks:
- `pymp_MDCTBlock`  is a block based on the standard MDCT transform. Which means atoms localizations
                 are constrained by the scale of the transform
                 The atom selected is simple the one that maximizes the correlation with the residual
                 it is directly indexes by the max absolute value of the MDCT bin
                 No further optimization of the selected atom is performed

- `pymp_LOBlock`   is a block that performs a local optimization of the time localization of the selected atom

- `pymp_FullBlock`  this block simulates a dictionary where atoms at all time localizations are available
                  This kind of dictionary can be thought of as a Toeplitz matrix
                  BEWARE: memory consumption and CPU load will be very high! only use with very small signals 
                  (e.g. 1024 samples or so) Fairly intractable at higher dimensions
'''

# imports
from Classes import  PyWinServer , pymp_Log
from Classes.pymp_Block import pymp_Block 
from Classes.mdct import pymp_MDCTAtom
from numpy.fft import fft
from numpy import array , zeros , ones ,concatenate , sin , pi , random , abs , argmax ,max, sum,multiply,nan_to_num
from scipy.stats import gmean 
from math import sqrt , floor
from cmath import exp
import matplotlib.pyplot as plt
from Tools import  Xcorr #, Misc 
import parallelProjections, sys


# declare global PyWinServer shared by all MDCT blocks instances
global _PyServer , _Logger
_PyServer = PyWinServer.PyServer()
_Logger = pymp_Log.pymp_Log('MDCTBlock', level = 0)

class pymp_MDCTBlock(pymp_Block):
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
    enframedDataMatrix = None;
    projectionMatrix = None;    
    frameLength = 0;
    frameNumber = 0; 
    
    maxIndex = 0;
    maxValue = 0;
    maxBinIdx = 0;
    maxFrameIdx = 0;
    
    # MDCT static window and twiddle parameters
    wLong = None;
    wEdgeL = None;
    wEdgeR = None;
    
    pre_twidVec = None
    post_twidVec = None;
        
    # store fft matrix for later Xcorr purposes
    fftMat = None
    fft = None
    # Score tree
    bestScoreTree = None
    
    # optim?
    useC = True
    HF = False;
    HFlimit = 0.1;
    
    windowType =None;
    
    # constructor - initialize residual signal and projection matrix
    def __init__(self , length = 0 , resSignal = None , frameLen = 0 , useC =True , forceHF=False , debugLevel = None):
        if debugLevel is not None:
            _Logger.setLevel(debugLevel)
        
        self.scale = length;
        self.residualSignal = resSignal;
        
        if frameLen==0:
            self.frameLength = length/2;
        else:
            self.frameLength = frameLen;        
        if self.residualSignal ==None:
            raise ValueError("no signal given")

        self.enframedDataMatrix = self.residualSignal.dataVec;
        self.frameNumber = len(self.enframedDataMatrix) / self.frameLength
        self.projectionMatrix = zeros(len(self.enframedDataMatrix));
        self.useC = useC;
        self.HF = forceHF;
        _Logger.info('new MDCT block constructed size : ' + str(self.scale));
#        self.projectionMatrix = zeros((self.frameNumber , self.frameLength));
        
        
    # compute mdct of the residual and instantiate various windows and twiddle coefficients
    def initialize(self ):        
        
        #Windowing  
        L = self.scale;
        
        self.wLong = array([ sin(float(l + 0.5) *(pi/L)) for l in range(L)] )

        # twidlle coefficients
        self.pre_twidVec = array([exp(n*(-1j)*pi/L) for n in range(L)]);
        self.post_twidVec = array([exp((float(n) + 0.5) * -1j*pi*(L/2 +1)/L) for n in range(L/2)]) ;    
        
        if self.windowType == 'half1':
            self.wLong[0:L/2] = 0;
            # twidlle coefficients
            self.pre_twidVec[0:L/2] = 0;
#        self.fftMat = zeros((self.scale , self.frameNumber) , complex);
#        self.normaCoeffs = sqrt(1/float(L));
        
        # score tree - first version simplified
        self.bestScoreTree = zeros(self.frameNumber);
        
        if self.HF:
            self.bestScoreHFTree = zeros(self.frameNumber);
        
        # OPTIM -> do pre-twid directly in the windows
        self.locCoeff = self.wLong * self.pre_twidVec

        
    # Search among the inner products the one that maximizes correlation
    # the best candidate for each frame is already stored in the bestScoreTree
    def getMaximum(self):
        treeMaxIdx = self.bestScoreTree.argmax();        
        
        maxIdx = abs(self.projectionMatrix[treeMaxIdx*self.scale/2 : (treeMaxIdx+1)*self.scale/2]).argmax()                
        self.maxIdx = maxIdx + treeMaxIdx*self.scale/2;
        self.maxValue = self.projectionMatrix[self.maxIdx]
        
#        print "block getMaximum called : " , self.maxIdx , self.maxValue 
        
        if self.HF:
            treemaxHFidx = self.bestScoreHFTree.argmax();  
            maxHFidx = abs(self.projectionMatrix[(treemaxHFidx+self.HFlimit)*self.scale/2 : (treemaxHFidx+1)*self.scale/2]).argmax()                
            
            self.maxHFIdx = maxHFidx + (treemaxHFidx+self.HFlimit)*self.scale/2;
            self.maxHFValue = self.projectionMatrix[self.maxHFIdx]



    
    # construct the atom that best correlates with the signal
    def getMaxAtom(self , HF = False):    
        if not HF:
            self.maxFrameIdx = floor(self.maxIdx / (0.5*self.scale));
            self.maxBinIdx = self.maxIdx - self.maxFrameIdx * (0.5*self.scale)         
        else:
            self.maxFrameIdx = floor(self.maxHFIdx / (0.5*self.scale));
            self.maxBinIdx = self.maxHFIdx - self.maxFrameIdx * (0.5*self.scale) 
        Atom = pymp_MDCTAtom.pymp_MDCTAtom(self.scale , 1 , max((self.maxFrameIdx  * self.scale/2) - self.scale/4 , 0)  , self.maxBinIdx , self.residualSignal.samplingFrequency)
        Atom.frame = self.maxFrameIdx;
#        print self.maxBinIdx, Atom.reducedFrequency
        if not HF:
            Atom.mdct_value =  self.maxValue;
        else:
            Atom.mdct_value =  self.maxHFValue;
        
        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesizeAtom()
        
        if HF:
            Atom.waveform = Atom.waveform*(self.maxHFValue/self.maxValue);
        return Atom
    
    def getWindow(self):
        return self.wLong
    
    # update the part of the residual that has been changed and update inner products    
    def update(self , newResidual , startFrameIdx=0 , stopFrameIdx=-1):
#        print "block update called : " , self.scale , newResidual.length , self.frameNumber
        self.residualSignal = newResidual;
        
        if stopFrameIdx <0:
            endFrameIdx = self.frameNumber -1
        else:
            endFrameIdx = stopFrameIdx
        
        L = self.scale

        self.enframedDataMatrix[startFrameIdx*L/2 : endFrameIdx*L/2 + L] = self.residualSignal.dataVec[startFrameIdx*self.frameLength : endFrameIdx*self.frameLength + 2*self.frameLength]

        self.computeTransform(startFrameIdx , stopFrameIdx);
        
        self.getMaximum()
        
    # inner product computation through MDCT
    def computeTransform(self, startingFrame=1 , endFrame = -1):
        if self.wLong is None:
            self.initialize()
            
        if endFrame <0:
            endFrame = self.frameNumber -1
        
        # debug -> changed from 1: be sure signal is properly zero -padded
        if startingFrame<1:
            startingFrame = 1
    
        
        # Wrapping C code call for fast implementation
        if self.useC:     
            try:              
                parallelProjections.project(self.enframedDataMatrix, self.bestScoreTree,
                                                 self.projectionMatrix , 
                                                 self.locCoeff , 
                                                 self.post_twidVec ,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale ,0)

            except SystemError:
                print sys.exc_info()[0];
                print sys.exc_info()[1];
                raise
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
        else:
            try:
                import fftw3
            except ImportError:
                print '''Impossible to load fftw3, you need to use the C extension library or have
                        a local installation of the python fftw3 wrapper '''
                return;    
        # create a forward fft plan, since fftw is not faster for computing DCT's no need to use it for that
            L = self.scale
            K = L/2;
            T = K/2
            normaCoeffs = sqrt(2/float(K))
            if self.fft is None:
                self.inputa = self.enframedDataMatrix[K - T: K + L - T].astype(complex)
                self.outputa = None
                self.fft = fftw3.Plan(self.inputa,self.outputa, direction='forward', flags=['estimate'])                 
                            
            self.project(startingFrame,endFrame, L , K , T , normaCoeffs);
        
        if self.HF:
            for i in range(startingFrame , endFrame):
                self.bestScoreHFTree[i] = abs(self.projectionMatrix[(i+self.HFlimit)*self.scale/2 : (i+1)*self.scale/2]).max()
            
    # UPDATE: need to keep this slow version for non C99 compliant systems (windows)
    def project(self,startingFrame, endFrame, L,  K, T, normaCoeffs):
    
        for i in range(startingFrame , endFrame):
            # initialize input data
            self.inputa[:]=0
            self.inputa += self.enframedDataMatrix[i*K - T: i*K + L - T]*self.locCoeff
            
            #compute fft
            self.fft()
     
            # post-twiddle and store for max search          
            self.projectionMatrix[i*K : (i+1)*K] = normaCoeffs*(self.inputa[0:K]* self.post_twidVec).real
#            self.projectionMatrix[i , :] = normaCoeffs*(self.inputa[0:K]* self.post_twidVec).real

#            # store new max score in tree
            self.bestScoreTree[i] = abs(self.projectionMatrix[i*K : (i+1)*K]).max()


        
        
    # synthesizes the best atom through ifft computation (much faster than closed form)
    def synthesizeAtom(self , value=None):
        ###################" new version ############"
        global _PyServer
#        print len(_PyServer.Waveforms)
        if  value is None:
            return self.maxValue * _PyServer.getWaveForm(self.scale , self.maxBinIdx)
        else:
            return value * _PyServer.getWaveForm(self.scale , self.maxBinIdx)
        ###################  old version ############
#        temp = zeros(2*self.scale)
#        temp[self.scale/2 + self.maxBinIdx] = self.maxValue
#        waveform = zeros(2*self.scale)
#        #Number of frames : only 4 we need zeroes on the border before overlap-adding
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
#            y = y *  self.pre_i_twidVec;
#
#            # compute ifft
#            x = ifft(y)
#
#            # do the post-twiddle 
#            x = x * self.post_i_twidVec
#
#            x = 2*sqrt(1/float(L))*L*x.real*self.wLong;  
#
#            # overlapp - add
#            waveform[i*K : i*K +L] = waveform[i*K : i*K +L] +  x ;
#
#        # scrap zeroes on the borders    
#        return waveform[L/2:-L/2]
        
    def plotScores(self):
        plt.figure()
#        plt.subplot(211)
#        plt.plot(self.bestScoreTree)
#        plt.subplot(212)
        plt.plot(self.projectionMatrix)
        plt.title("Block-" + str(self.scale)+" best Score of" + str(self.maxValue) 
                  + " p :"+ str(self.maxFrameIdx) +" , k: "+ str(self.maxBinIdx) )
        
class pymp_LOBlock(pymp_MDCTBlock):
    """ Class that inherit classic MDCT block class and deals with local optimization
        This is the main class for differentiating LOMP from MP """
    
    # Typically all attributes are the same than mother class:
    maxTimeShift = 0
    adjustTimePos = True
    fftplan = None
    # constructor - initialize residual signal and projection matrix
    def __init__(self , length = 0 , resSignal = None , frameLen = 0 , tinvOptim = True, useC=True , forceHF=False):
        self.scale = length;
        self.residualSignal = resSignal;
        self.adjustTimePos = tinvOptim
        self.useC = useC;
        self.HF = forceHF;
        
        if frameLen==0:
            self.frameLength = length/2;
        else:
            self.frameLength = frameLen;        
        if self.residualSignal ==None:
            raise ValueError("no signal given")

        self.enframedDataMatrix = self.residualSignal.dataVec;
        self.frameNumber = len(self.enframedDataMatrix) / self.frameLength
        
        # only difference , here, we keep the complex values for the cross correlation
#        self.projectionMatrix = zeros(len(self.enframedDataMatrix) , complex);
        self.projectionMatrix = zeros(len(self.enframedDataMatrix) ,float)
        
        
        
        
#    # only the update method is interesting for us : we're hacking it to experiment 
#    def update(self , newResidual , startFrameIdx=0 , stopFrameIdx=-1):
#        self.residualSignal = newResidual;
#        
#        if stopFrameIdx <0:
#            endFrameIdx = self.frameNumber -1
#        else:
#            endFrameIdx = stopFrameIdx        
#        L = self.scale
#
#        # update residual signal
#        self.enframedDataMatrix[startFrameIdx*L/2 : endFrameIdx*L/2 + L] = self.residualSignal.dataVec[startFrameIdx*self.frameLength : endFrameIdx*self.frameLength + 2*self.frameLength]
#        
#        # TODO changes here
##        self.computeMDCT(startFrameIdx , stopFrameIdx);
#        self.computeMCLT(startFrameIdx , stopFrameIdx);
#        
#        # TODO changes here
#        self.getMaximum()

    # inner product computation through MDCT
    def computeTransform(self, startingFrame=1 , endFrame = -1):
        if self.wLong is None:
            self.initialize()
        
        # due to later time-shift optimizations , need to ensure nothing is selected too close to the borders!!
        
        if endFrame <0 or endFrame>self.frameNumber -3:
            endFrame = self.frameNumber -3
        
        if startingFrame<2:
            startingFrame = 2
        # new version: C binding
        if self.useC:
            parallelProjections.project_mclt(self.enframedDataMatrix, self.bestScoreTree,
                                                 self.projectionMatrix , 
                                                 self.locCoeff , 
                                                 self.post_twidVec ,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale )
        else:
            L = self.scale;
            K = L/2;
            T = K/2
            normaCoeffs = sqrt(2/float(K))
            
            locenframedDataMat = self.enframedDataMatrix
#            locfftMat = self.fftMat
#            locprojMat = self.projectionMatrix
            
            preTwidCoeff = self.locCoeff
            postTwidCoeff = self.post_twidVec
    
            # Bottleneck here !! need to fasten this loop : do it by Matrix technique? or bind to a C++ file?
            for i in range(startingFrame , endFrame):
                x = locenframedDataMat[i*K - T: i*K + L - T]
                if len(x) !=L:
                    x =zeros(L , complex);
                            
                # do the pre-twiddle 
                x = x * preTwidCoeff
                
                # compute fft            
#                locfftMat[: , i] = fft(x , L)
    
                # post-twiddle
#                y = locfftMat[0:K , i] * postTwidCoeff
                y = (fft(x , L)[0:K]) * postTwidCoeff
    #            y = self.doPretwid(locfftMat[0:K , i], postTwidCoeff)
                
                # we work with MCLT now
                self.projectionMatrix[i*K : (i+1)*K] = normaCoeffs*y;
                
                # store new max score in tree
                self.bestScoreTree[i] = abs(self.projectionMatrix[i*K : (i+1)*K]).max() 
                
#        if self.HF:
#            for i in range(startingFrame , endFrame):
#                self.bestScoreHFTree[i] = abs(self.projectionMatrix[(i+self.HFlimit)*self.scale/2 : (i+1)*self.scale/2]).max()
#             
    
    # construct the atom that best correlates with the signal
    def getMaxAtom(self , debug = 0):    
        
        self.maxFrameIdx = floor(self.maxIdx / (0.5*self.scale));
        self.maxBinIdx = self.maxIdx - self.maxFrameIdx * (0.5*self.scale)         

        # hack here : let us project the atom waveform on the neighbouring signal in the FFt domain,
        # so that we can find the maximum correlation and best adapt the time-shift             
        Atom = pymp_MDCTAtom.pymp_MDCTAtom(self.scale , 1 , max((self.maxFrameIdx  * self.scale/2) - self.scale/4 , 0)  , self.maxBinIdx , self.residualSignal.samplingFrequency)
        Atom.frame = self.maxFrameIdx;
        
        # re-compute the atom amplitude for IMDCT      
#        if self.maxValue.real < 0:
#            self.maxValue = -abs(self.maxValue)
#        else:
#            self.maxValue = abs(self.maxValue)
            
        Atom.mdct_value = self.maxValue;        
        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesizeAtom(value=1)
        Atom.timeShift = 0
        Atom.projectionScore = 0.0
        
        input1 = self.enframedDataMatrix[(self.maxFrameIdx-1.5)  * self.scale/2 : (self.maxFrameIdx+2.5)  * self.scale/2];
        input2 = concatenate( (concatenate((zeros(self.scale/2) , Atom.waveform) ) , zeros(self.scale/2) ) );
        
        # debug cases: sometimes when lot of energy on the borders , pre-echo artifacts tends
        # to appears on the border and can lead to seg fault if not monitored
        # therefore we need to prevent any time shift leading to positions outside of original signals
        # ie in the first and last frames.
#        if debug>0:
#            print self.maxFrameIdx , self.maxIdx , self.frameNumber
#            if self.maxFrameIdx ==1:
#                plt.figure()
#                plt.plot(Atom.waveform)
#                plt.plot(self.enframedDataMatrix[0 : Atom.timePosition + Atom.length],'r')
#                plt.show()
#             
#            print (self.maxFrameIdx-1.5)  * self.scale/2 , (self.maxFrameIdx+2.5)  * self.scale/2
#            print len(input1) , len(input2);
        if len(input1) != len(input2):
            print self.maxFrameIdx , self.maxIdx , self.frameNumber
            print len(input1) , len(input2);
            #if debug>0:
            print "atom in the borders , no timeShift calculated"
            return Atom;

        # retrieve additional timeShift
        if self.useC:
            scoreVec = array([0.0]);
            Atom.timeShift = parallelProjections.project_atom(input1,input2 , scoreVec , self.scale)

            if abs(Atom.timeShift) > Atom.length/2:
                print "out of limits: found time shift of" , Atom.timeShift
                Atom.timeShift = 0
                return Atom
            
            self.maxTimeShift = Atom.timeShift
            Atom.timePosition += Atom.timeShift
            
            # retrieve newly projected waveform
            Atom.projectionScore = scoreVec[0];
            Atom.waveform *= Atom.projectionScore;
#            Atom.waveform = input2[self.scale/2:-self.scale/2];
        else:
            sigFft = fft(self.enframedDataMatrix[(self.maxFrameIdx-1.5)  * self.scale/2 : (self.maxFrameIdx+2.5)  * self.scale/2] , 2*self.scale)
            atomFft = fft(concatenate( (concatenate((zeros(self.scale/2) , Atom.waveform) ) , zeros(self.scale/2) ) ) , 2*self.scale)
                    
            
            Atom.timeShift , score = Xcorr.GetMaxXCorr(atomFft , sigFft , maxlag =self.scale/2)        
            self.maxTimeShift = Atom.timeShift
            Atom.projectionScore = score
    #        print "found correlation max of " , float(score)/sqrt(2/float(self.scale))
            
            # CAses That might happen: time shift result in choosing another atom instead
            if abs(Atom.timeShift) > Atom.length/2:
                print "out of limits: found time shift of" , Atom.timeShift
                Atom.timeShift = 0
                return Atom
     
            Atom.timePosition += Atom.timeShift
            
            # now let us re-project the atom on the signal to adjust it's energy: Only if no pathological case        
            
            # TODO optimization : pre-compute energy (better: find closed form)
            if score <0:
                Atom.amplitude = -sqrt(-score);
                Atom.waveform = (-sqrt(-score/sum(Atom.waveform**2)) )*Atom.waveform
            else:
                Atom.amplitude = sqrt(score);
                Atom.waveform = (sqrt(score/sum(Atom.waveform**2)) )*Atom.waveform
#        projOrtho = sum(Atom.waveform * self.residualSignal.dataVec[Atom.timePosition : Atom.timePosition + Atom.length])
        
#        if score <0:
#            Atom.amplitude = -1;
#            Atom.waveform = -Atom.waveform
        
        return Atom
    
    def plotScores(self):
        maxFrameIdx = floor(self.maxIdx / (0.5*self.scale));
        maxBinIdx = self.maxIdx - maxFrameIdx * (0.5*self.scale)
        plt.figure()
#        plt.subplot(211)
#        plt.plot(self.bestScoreTree)
#        plt.subplot(212)
        plt.plot(abs(self.projectionMatrix) )
        plt.title("Block-" + str(self.scale)+" best Score of" + str(abs(self.maxValue)) + " at " + str(self.maxIdx) 
                  + " p :"+ str(maxFrameIdx) 
                  +" , k: "+ str(maxBinIdx)
                  +" , l: "+ str(self.maxTimeShift) )
        
class pymp_FullBlock(pymp_MDCTBlock):
    
    # parameters
    maxKidx = 0
    maxLidx = 0
    
    
    # constructor - initialize residual signal and projection matrix
    def __init__(self , length = 0 , resSignal = None , frameLen = 0  ):
        self.scale = length;
        self.residualSignal = resSignal;
        
        
        if frameLen==0:
            self.frameLength = length/2;
        else:
            self.frameLength = frameLen;        
        if self.residualSignal ==None:
            raise ValueError("no signal given")

        self.enframedDataMatrix = self.residualSignal.dataVec;
        self.frameNumber = len(self.enframedDataMatrix) / self.frameLength
        
        # ok here the mdct will be computed for every possible time shift
        # so allocate enough memory for all the transforms
        
        # here the projection matrix is actually a list of K by K matrices
        self.projectionMatrix = dict()
        self.bestScoreTree = dict()
        
        # initialize an empty matrix of size len(data) for each shift
        for i in range(self.scale/2):
#            self.projectionMatrix[i] = zeros((self.scale/2 , self.scale/2))
            self.projectionMatrix[i] = zeros(len(self.enframedDataMatrix))
            self.bestScoreTree[i] = zeros(self.frameNumber);

    def initialize(self ):        
        
        #Windowing  
        L = self.scale;
        
        self.wLong = array([ sin(float(l + 0.5) *(pi/L)) for l in range(L)] )

        # twidlle coefficients
        self.pre_twidVec = array([exp(n*(-1j)*pi/L) for n in range(L)]);
        self.post_twidVec = array([exp((float(n) + 0.5) * -1j*pi*(L/2 +1)/L) for n in range(L/2)]) ;    
        
        if self.windowType == 'half1':
            self.wLong[0:L/2] = 0;
            # twidlle coefficients
            self.pre_twidVec[0:L/2] = 0;
  
        # OPTIM -> do pre-twid directly in the windows
        self.locCoeff = self.wLong * self.pre_twidVec
        
    # The update method is nearly the same as CCBlock
    def update(self , newResidual , startFrameIdx=0 , stopFrameIdx=-1):
        self.residualSignal = newResidual;
        
        if stopFrameIdx <0:
            endFrameIdx = self.frameNumber -1
        else:
            endFrameIdx = stopFrameIdx        
            
        # debug : recompute absolutely all the products all the time

        L = self.scale
        
#        print "Updating Block " , L , " from frame " , startFrameIdx , " to " , endFrameIdx
        
        # update residual signal
        self.enframedDataMatrix[startFrameIdx*L/2 : endFrameIdx*L/2 + L] = self.residualSignal.dataVec[startFrameIdx*self.frameLength : endFrameIdx*self.frameLength + 2*self.frameLength]
        
        # TODO changes here
        self.computeTransform(startFrameIdx , stopFrameIdx);
        
        # TODO changes here
        self.getMaximum()
        
    # inner product computation through MCLT with all possible time shifts
    def computeTransform(self, startingFrame=1 , endFrame = -1):
        if self.wLong is None:
            self.initialize()
            
        if endFrame <0:
            endFrame = self.frameNumber -1
        
        if startingFrame<1:
            startingFrame = 1
        
        startingFrame=1 
        endFrame = self.frameNumber -1
        
        L = self.scale
        K = L/2;
#        T = K/2
#        normaCoeffs = sqrt(2/float(K))
#        print startingFrame , endFrame
        for l in range(-K/2,K/2,1):
            parallelProjections.project(self.enframedDataMatrix, 
                                                 self.bestScoreTree[l+K/2],
                                                 self.projectionMatrix[l+K/2] , 
                                                 self.locCoeff , 
                                                 self.post_twidVec ,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale ,l)

            
    def getMaximum(self):  
        K = self.scale/2;
        bestL = 0;
        treeMaxIdx = 0;
        bestSCore = 0;
        for l in range(-K/2,K/2,1):
            if self.bestScoreTree[l+K/2].max() > bestSCore:
                bestSCore = self.bestScoreTree[l+K/2].max()
                treeMaxIdx = self.bestScoreTree[l+K/2].argmax()
                bestL = l;
        
        
        maxIdx = abs(self.projectionMatrix[bestL + K/2]).argmax()   

        self.maxLidx = bestL 
        
        self.maxIdx = maxIdx;
        self.maxFrameIdx = treeMaxIdx;
        self.maxValue = self.projectionMatrix[bestL + K/2][maxIdx]
        
        
#        print "Max Atom : " , self.maxIdx , self.maxLidx , self.maxValue
        
    # construct the atom that best correlates with the signal
    def getMaxAtom(self):    
        self.maxBinIdx = self.maxIdx - self.maxFrameIdx * (0.5*self.scale)   
        
        Atom = pymp_MDCTAtom.pymp_MDCTAtom(self.scale , 1 , max((self.maxFrameIdx  * self.scale/2) - self.scale/4 , 0)  , self.maxBinIdx , self.residualSignal.samplingFrequency)
        Atom.frame = self.maxFrameIdx;
        
        # re-compute the atom amplitude for IMDCT
        Atom.mdct_value = self.maxValue;

        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesizeAtom()
        
        
        
        Atom.timeShift = -self.maxLidx #+ self.scale/4
        self.maxTimeShift = Atom.timeShift
        
        Atom.timePosition += Atom.timeShift
        
        return Atom
    
    
    def plotScores(self):
        ''' For debug purposes only... '''
        
        plt.figure()
#        plt.subplot(211)
        plt.plot(abs(self.projectionMatrix[self.maxLidx + self.scale/4]))
#        plt.title("reference l=0 , best k of " 
#                  + str(bestK)
#                  + " max score of : " + str(abs(self.projectionMatrix[self.maxIdx][bestK,:]).max())
#                  + " at l: " + str(abs(self.projectionMatrix[self.maxIdx][bestK,:]).argmax() - self.scale/4) )
#        plt.subplot(212)
#        plt.imshow(abs(self.projectionMatrix[self.maxIdx]) , 
#                   aspect='auto', cmap = cm.get_cmap('greys') , 
#                   extent=(-self.scale/4 , self.scale/4, self.scale/2, 0) ,
#                   interpolation = 'nearest')
        plt.title("Block-" + str(self.scale)+" best Score of" + str(self.maxValue)
                  + " p :"+ str(self.maxFrameIdx) 
#                  +" , k: "+ str(self.maxKidx)
                  +" , l: "+ str(self.maxLidx) )

        
class pymp_SpreadBlock(pymp_MDCTBlock):
    ''' Spread MP is a technique in which you penalize the selection of atoms near existing ones
        in a  predefined number of iterations, or you specify a perceptual TF masking to enforce the selection of
        different features. The aim is to have maybe a loss in compressibility in the first iteration but
        to have the most characteristics / discriminant atoms be chosen in the first iterations '''
        
    # parameters
    distance = None
    mask = None
    maskSize = None;
    penalty = None;
    
    
    def __init__(self , length = 0 , resSignal = None , frameLen = 0 , useC =True , forceHF=False , 
                 debugLevel = None , penalty=0.5 ,maskSize = 1):
        
        if debugLevel is not None:
            _Logger.setLevel(debugLevel)
        
        self.scale = length;
        self.residualSignal = resSignal;
        
        if frameLen==0:
            self.frameLength = length/2;
        else:
            self.frameLength = frameLen;        
        if self.residualSignal ==None:
            raise ValueError("no signal given")

        self.enframedDataMatrix = self.residualSignal.dataVec;
        self.frameNumber = len(self.enframedDataMatrix) / self.frameLength
        self.projectionMatrix = zeros(len(self.enframedDataMatrix));
        self.useC = useC;
        self.HF = forceHF;
        _Logger.info('new MDCT block constructed size : ' + str(self.scale));
        
        self.penalty = penalty;
        # initialize the mask: so far no penalty
        self.mask = ones(len(self.enframedDataMatrix))
        self.maskSize = maskSize
    
    def getMaximum(self, it = -1):
        ''' Apply the mask to the projection before choosing the maximum '''
        
        # cannot use the tree indexing ant more... too bad
#        treeMaxIdx = self.bestScoreTree.argmax();                
        self.projectionMatrix *= self.mask;        
#        print self.enframedDataMatrix.shape, self.mask.shape
        
#        plt.figure()
#        plt.imshow(reshape(self.mask,(self.frameNumber,self.scale/2)),interpolation='nearest',aspect='auto')
        self.maxIdx = argmax(abs(self.projectionMatrix)); 
        
#        print self.maxIdx
        
#        maxIdx = abs(self.projectionMatrix[treeMaxIdx*self.scale/2 : (treeMaxIdx+1)*self.scale/2]).argmax()                       
#        self.maxIdx = maxIdx + treeMaxIdx*self.scale/2;
        self.maxValue = self.projectionMatrix[self.maxIdx]
    
    def getMaxAtom(self , HF = False):    
        self.maxFrameIdx = floor(self.maxIdx / (0.5*self.scale));
        self.maxBinIdx = self.maxIdx - self.maxFrameIdx * (0.5*self.scale)         
        
        # update the mask : penalize the choice of an atom overlapping in time and or frequency
        for i in range(-self.maskSize,self.maskSize+1,1):            
            self.mask[((self.maxFrameIdx+i)*self.scale/2) + (self.maxBinIdx- self.maskSize) : ((self.maxFrameIdx+i)*self.scale/2) + (self.maxBinIdx + self.maskSize)] = self.penalty
            
        # proceed as usual
        Atom = pymp_MDCTAtom.pymp_MDCTAtom(self.scale , 1 , max((self.maxFrameIdx  * self.scale/2) - self.scale/4 , 0)  , self.maxBinIdx , self.residualSignal.samplingFrequency)
        Atom.frame = self.maxFrameIdx;
        
        Atom.mdct_value =  self.maxValue;

        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesizeAtom()
        
        return Atom   
        