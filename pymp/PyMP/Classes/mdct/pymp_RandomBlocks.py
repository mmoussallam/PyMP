'''
#                                                                            
#                       Classes.mdct.pymp_RandomBlocks                                     
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

Please refer to superclass for documentation

This file handle blocks that are used in Randomized Matching Pursuits
see [1] for details

[1] M. Moussallam, L. Daudet, et G. Richard, 
"Matching Pursuits with Random Sequential Subdictionaries"
Signal Processing, vol. 92, pp. 2532-2544 2012.
                                                                         
'''

from Classes import  PyWinServer , pymp_Log
from Classes.mdct.pymp_MDCTBlock import pymp_MDCTBlock 
from Classes.mdct import pymp_MDCTAtom
from numpy.fft import fft
from numpy import array , zeros , ones ,concatenate , sin , pi , random , abs , argmax ,max, sum,multiply,nan_to_num
from scipy.stats import gmean 
from math import sqrt , floor
from cmath import exp
import matplotlib.pyplot as plt
from Tools import  Xcorr , Misc 
import parallelProjections, sys


# declare global PyWinServer shared by all MDCT blocks instances
global _PyServer , _Logger
_PyServer = PyWinServer.PyServer()
_Logger = pymp_Log.pymp_Log('RandomMDCTBlock', level = 0)

class pymp_RandomBlock(pymp_MDCTBlock):
    """ block in which the time-shift pattern is predefined """
    
    # properties
    randomType = 'random'
    TSsequence =  []
    currentTS = 0
    currentSubF = 0
    nbSim = 1
    wLong = None
    
    # constructor - initialize residual signal and projection matrix
    def __init__(self , length = 0 , resSignal = None , frameLen = 0 ,randomType='random' , nbSim = 1 , windowType = None):
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
        
        self.randomType = randomType
        if self.randomType == 'scale':
            self.TSsequence = range(self.scale/2)
        elif self.randomType == 'random':
            self.TSsequence = [floor((self.scale/2)* (i-0.5) ) for i in random.random(self.scale)]
        elif self.randomType == 'gaussian':
            self.TSsequence = [floor(self.scale/8* i)  for i in random.randn(self.scale/2)]
            for k in self.TSsequence:
                k = min(k , self.scale/4);
                k = max(k , -self.scale/4);
            
        elif self.randomType == 'binom':
            self.TSsequence = Misc.binom(range(self.scale/2))
        elif self.randomType == 'dicho':
            self.TSsequence = Misc.dicho([] , range(self.scale/2))
        elif self.randomType == 'jump':
            self.TSsequence = Misc.jump(range(self.scale/2))
        elif self.randomType == 'sine':
            self.TSsequence = Misc.sine(range(self.scale/2))
#        elif self.randomType == 'triangle':
#            self.TSsequence = Misc.triangle(range(self.scale/2))
        elif self.randomType == 'binary':
            self.TSsequence = Misc.binary(range(self.scale/2))
            
        else:
            self.TSsequence = zeros(self.scale/2)
    
        self.nbSim = nbSim
        self.windowType = windowType
        
    # The update method is nearly the same as CCBlock
    def update(self , newResidual , startFrameIdx=0 , stopFrameIdx=-1 ,iterationNumber=0):
        """ at each update, one need to pick a time shift from the sequence """ 
        
        # change the current Time-Shift if enough iterations have been done
        if (self.nbSim > 0 ):
            if (  iterationNumber % self.nbSim == 0):
                self.currentTS = self.TSsequence[(iterationNumber / self.nbSim) % len(self.TSsequence)]                
        self.residualSignal = newResidual;
        
        if stopFrameIdx <0:
            endFrameIdx = self.frameNumber -1
        else:
            endFrameIdx = stopFrameIdx        
        L = self.scale

        # update residual signal
        self.enframedDataMatrix[startFrameIdx*L/2 : endFrameIdx*L/2 + L] = self.residualSignal.dataVec[startFrameIdx*self.frameLength : endFrameIdx*self.frameLength + 2*self.frameLength]
        
        # TODO changes here
        self.computeTransform(startFrameIdx , stopFrameIdx ) ;
        
        # TODO changes here
        self.getMaximum()
        
    # inner product computation through MCLT with all possible time shifts
    def computeTransform(self,   startingFrame=1 , endFrame = -1 ):
        if self.wLong is None:
            self.initialize()
            
        if endFrame <0:
            endFrame = self.frameNumber -2
        
        if startingFrame<2:
            startingFrame = 2
#        L = self.scale
#        K = L/2;
        
        ############" changes  ##############
#        T = K/2 + self.currentTS
        # new version with C calculus
        parallelProjections.project(self.enframedDataMatrix, self.bestScoreTree,
                                                 self.projectionMatrix , 
                                                 self.locCoeff , 
                                                 self.post_twidVec ,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale ,
                                                 int(self.currentTS))
#        computeMCLT.project(self.enframedDataMatrix, self.bestScoreTree,
#                                                 self.projectionMatrix , 
#                                                 self.locCoeff , 
#                                                 self.post_twidVec ,
#                                                 startingFrame,
#                                                 endFrame,
#                                                 self.scale ,
#                                                 int(self.currentTS))
        
        
#        normaCoeffs = sqrt(2/float(K))
##        locCoeff = self.wLong * self.pre_twidVec
#        for i in range(startingFrame , endFrame):
#
#
#            x = self.enframedDataMatrix[i*K - T: i*K + L - T]
#            if len(x) !=L:
#                x =zeros(L , complex);
#
#            # compute windowing and pre-twiddle simultaneously : suppose first and last frame are always zeroes
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
#            self.bestScoreTree[i] = abs(self.projectionMatrix[i*K : (i+1)*K]).max()
            
    def getMaximum(self):  
        treeMaxIdx = self.bestScoreTree.argmax();        
        maxIdx = abs(self.projectionMatrix[treeMaxIdx*self.scale/2 : (treeMaxIdx+1)*self.scale/2]).argmax()                
        
        self.maxIdx = maxIdx + treeMaxIdx*self.scale/2;
        self.maxValue = self.projectionMatrix[self.maxIdx]

        
    # construct the atom 
    def getMaxAtom(self):    
        self.maxFrameIdx = floor(self.maxIdx / (0.5*self.scale));
        self.maxBinIdx = self.maxIdx - self.maxFrameIdx * (0.5*self.scale)   
        Atom = pymp_MDCTAtom.pymp_MDCTAtom(self.scale , 1 , max((self.maxFrameIdx  * self.scale/2) - self.scale/4 , 0) , self.maxBinIdx , self.residualSignal.samplingFrequency)
        Atom.frame = self.maxFrameIdx;
        Atom.mdct_value =  self.maxValue;
        
        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesizeAtom()
        
        if self.windowType == 'half1':
            Atom.waveform[0:self.scale/2] = 0;
        
        Atom.timeShift = self.currentTS     
        Atom.timePosition -= Atom.timeShift
        
    
        return Atom
    
class pymp_SubRandomBlock(pymp_RandomBlock):   
    """ Everything is inherited from the above class but the max atom is searched for in a subset of window frames """
    def __init__(self , length = 0 , resSignal = None , frameLen = 0 ,randomType='scale' , nbSim = 1 , windowType = None, subFactor = 2):
        
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
        
        self.randomType = randomType

#        elif self.randomType == 'random':
        # For now we just care about the uniform case : we have to add the subsampling factor so as to cover for the entire space
        self.TSsequence = [floor(subFactor*(self.scale/2)* (i-0.5) ) for i in random.random(self.scale)]
 
#        else:
#            self.TSsequence = zeros(self.scale/2)
    
        self.nbSim = nbSim
        self.windowType = windowType     
     
    # SPECIFIC PARAMETER: SUBSAMPLING FACTOR
        self.subFactor = subFactor;
     
    def computeTransform(self,   startingFrame=1 , endFrame = -1 ):
        if self.wLong is None:
            self.initialize()
            
        if endFrame <0:
            endFrame = self.frameNumber - (self.subFactor +1)
        
        if startingFrame<self.subFactor:
            startingFrame = self.subFactor +1


        ############" changes  ##############
        # C calculus : do not compute the odd frames
        parallelProjections.subproject(self.enframedDataMatrix, self.bestScoreTree,
                                                 self.projectionMatrix , 
                                                 self.locCoeff , 
                                                 self.post_twidVec ,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale ,
                                                 int(self.currentTS),
                                                 self.subFactor)


class pymp_VarSizeRandomBlock(pymp_RandomBlock):   
    """ Everything is inherited from the above class but the max atom is searched for in a subset of window frames. 
    The subset size may change at each iteration """
    def __init__(self , length = 0 , resSignal = None , frameLen = 0 ,randomType='scale' , nbSim = 1 , windowType = None, subFactorList = 2):
        
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
        
        self.randomType = randomType

        if isinstance(subFactorList, (int, long)):
            self.subFactorList = ones((self.scale,))*subFactorList;
               
        else:
            self.subFactorList = subFactorList;  
        
#        elif self.randomType == 'random':
        # For now we just care about the uniform case : we have to add the subsampling factor so as to cover for the entire space
        
         
        randomsS = random.random(len(self.subFactorList));
        self.TSsequence = [floor(self.subFactorList[i]*(self.scale/2)* (randomsS[i]-0.5) ) for i in range(len(self.subFactorList))]
 
#        else:
#            self.TSsequence = zeros(self.scale/2)
    
        self.nbSim = nbSim
        self.windowType = windowType     
    
    # The update method is nearly the same as RandomBlock
    def update(self , newResidual , startFrameIdx=0 , stopFrameIdx=-1 ,iterationNumber=0):
        """ at each update, one need to pick a time shift from the sequence """ 
        
        # change the current Time-Shift if enough iterations have been done
        if (self.nbSim > 0 ):
            if (  iterationNumber % self.nbSim == 0):
                self.currentTS = self.TSsequence[(iterationNumber / self.nbSim) % len(self.TSsequence)]     
                self.currentSubF = self.subFactorList[(iterationNumber / self.nbSim) % len(self.subFactorList)]                   
        self.residualSignal = newResidual;
        
        if stopFrameIdx <0:
            endFrameIdx = self.frameNumber -1
        else:
            endFrameIdx = stopFrameIdx        
        L = self.scale

        # update residual signal
        self.enframedDataMatrix[startFrameIdx*L/2 : endFrameIdx*L/2 + L] = self.residualSignal.dataVec[startFrameIdx*self.frameLength : endFrameIdx*self.frameLength + 2*self.frameLength]
        
        # TODO changes here
        self.computeTransform(startFrameIdx , stopFrameIdx ) ;
        
        # TODO changes here
        self.getMaximum() 
     
    def computeTransform(self,   startingFrame=1 , endFrame = -1 ):
        if self.wLong is None:
            self.initialize()
            
        if endFrame <0:
            endFrame = self.frameNumber - (self.currentSubF +1)
        
        if startingFrame<self.currentSubF:
            startingFrame = self.currentSubF +1

        self.bestScoreTree = zeros(self.frameNumber);
        ############" changes  ##############
        # C calculus : do not compute the odd frames
#        print int(self.currentSubF)
        parallelProjections.subproject(self.enframedDataMatrix, self.bestScoreTree,
                                                 self.projectionMatrix , 
                                                 self.locCoeff , 
                                                 self.post_twidVec ,
                                                 startingFrame,
                                                 endFrame,
                                                 self.scale ,
                                                 int(self.currentTS),
                                                 int(self.currentSubF))




