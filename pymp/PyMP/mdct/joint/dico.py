                                                                         
import mdct.dico as mdct_dico
import log 
import mdct.joint.block as joint_block
import mdct.random.dico as random_dico
import math
#from numpy import  abs
from numpy import random, zeros ,sqrt, abs, sum , array, median , mean


class SetDico(mdct_dico.Dico):
    """ This class handles multiple dictionaries; the best atom is selected when best explaining all the signals
        Then a refinement phase is performed late to determine the local time shifts and amplitudes 
        This class handles a set of dictionaries, one for each Jointly decomposed signals
        update method just call update in all dictionaries, just like initialize
        getBestAtom starts with retrieving all the projections, sums them all and then decide what atom is the 
        best choice. Then it is locally adapted to each signal, so a list of atoms is returned instead of one
        """

    # parameters
    dictionaryList = [];
    sizes = [];
    tolerances = [];
#    blocksList = [];
    blocks = []
    bestCurrentBlock = None;
    maxBlockScores = [];
    nature = None; # this parameter controls the atom selection function: sum sums scores across signals
                    # prod : multiply them
                    # maximin : selects 
    
    startingTouchedIndex = [];
    endingTouchedIndex = [];
    
    def __init__(self, sizes, useC = True , selectNature = 'sum' , tol =None , nonLinear=False, params=None):
        " Create an set of dictionaries, with blocks and everything."
        self.useC = useC;
        self.sizes = sizes;
        self.type = type;
        self.projections = None;
        self.nature = selectNature;
        self.isNL = nonLinear;
        if tol is not None:
            self.tolerances = tol;
        else:
            self.tolerances = [2]*len(self.sizes);
        
#        print self.sizes, self.tolerances;

        if selectNature == 'median' or selectNature == 'penalized'or selectNature == 'weighted':
            self.isNL = True
            print "NL dico detected"
            self.params = params;
            
    def initialize(self , residualSignalList):
        self.blocks = [];
        self.bestCurrentBlock = None;

        
        ######### Optimized version : 1 block to handle all projections with same window size########
        for mdctSize,tolerance in zip(self.sizes,self.tolerances):
#            print mdctSize , " Tolerance : " , tolerance
            if tolerance > 1:
#                print "Adaptation assumed"
                if not self.isNL:
                    self.blocks.append(joint_block.SetLOBlock(mdctSize , 
                                                               residualSignalList,
                                                                useC=self.useC , 
                                                                nature= self.nature,
                                                                tolerance = tolerance ))
                else:
                    self.blocks.append(joint_block.SetNLLOBlock(mdctSize , 
                                                               residualSignalList,
                                                                useC=self.useC , 
                                                                nature= self.nature,
                                                                tolerance = tolerance ,
                                                                lambd = self.params))
            else:
#                print "Tolerance too weak to allow atom optimization: No adaptation assumed"
                self.blocks.append(joint_block.SetBlock(mdctSize, 
                                                             residualSignalList,
                                                             useC=self.useC , 
                                                             nature= self.nature,
                                                             tolerance = tolerance))
        
        
        self.startingTouchedIndex = [0]*len(residualSignalList);
        self.endingTouchedIndex = [-1]*len(residualSignalList);
                
    def update(self , residualSignalList , iteratioNumber=0 , debug=0):       
        # Update all the blocks : Size by Size: so we can add the projections and have find the maximum
        self.maxBlockScore = 0;
        self.bestCurrentBlock = None;
        
        startingTouchedFrameList = [0]*len(residualSignalList)
        endingTouchedFrameList = [-1]*len(residualSignalList)
        
        # if parrallel library not available        
        for block in self.blocks:            
            
            for sigIdx in range(len(residualSignalList)):
                
#                startingTouchedFrameList[sigIdx] = 0;
#                endingTouchedFrameList[sigIdx] = -1;
#                print "DEBUG : recomputing all"
                startingTouchedFrameList[sigIdx] = int(math.floor(self.startingTouchedIndex[sigIdx] / (block.scale/2)))
                
                if self.endingTouchedIndex[sigIdx] > 0:
                    endingTouchedFrameList[sigIdx] = int(math.floor(self.endingTouchedIndex[sigIdx] / (block.scale/2) )) + 1; # TODO check this
                else:
                    endingTouchedFrameList[sigIdx] = -1;
                
#            print "block: " , block.scale , " : " , startingTouchedFrame , endingTouchedFrame
            block.update(residualSignalList , startingTouchedFrameList , endingTouchedFrameList )
                
            if abs(block.maxValue) > self.maxBlockScore: 
                self.maxBlockScore = abs(block.maxValue)
                self.bestCurrentBlock = block;

                
    def getBestAtom(self , debug , noAdapt=False):
        if self.bestCurrentBlock is None:
            score = 0;
            for block in self.blocks:
                maxValue = max(block.bestScoreTree);
                if abs(maxValue) > score: 
                    self.maxBlockScore = abs(maxValue)
                    self.bestCurrentBlock = block;
        
        if self.bestCurrentBlock is None:
            raise ValueError("no best block constructed, make sure inner product have been updated")

#        print 'Best Size is ', self.sizes[self.bestSize] , ' with score ' , self.bestScore , ' for atom ', self.maxAtomIdx
        
        # call on the best block to return a list of atoms that are adapted to each of the signals
        self.bestAtoms = self.bestCurrentBlock.getAdaptedBestAtoms(debug ,  noAdapt=False);
#        for sigIdx in range(len(self.blocksList)):
#            self.bestAtoms.append(self.blocksList[sigIdx][self.bestSize].getAdaptedMaxAtom(self.maxAtomIdx))

        return self.bestAtoms
    
    def getMeanAtom(self , getFirstAtom=True):
        ''' retrieve a mean best atom: with mean position and amplitude '''
        
        if self.bestAtoms is None or len(self.bestAtoms) <1:
            print " Empty set of Atoms: cannot create template"
        
        self.meanAtom = self.bestAtoms[0].copy()
        
        if getFirstAtom:
            return self.meanAtom;
        
        value = 0.0;
        timePos = 0;
        
#        for atom in self.bestAtoms:
##            value += sqrt(abs(atom.getAmplitude()));
#            timePos += atom.timePosition;
        value = mean([abs(at.projectionScore) for at in self.bestAtoms])
        timePos = median([at.timePosition for at in self.bestAtoms])

#        self.meanAtom.timePosition = int(timePos)#/len(self.bestAtoms);
        self.meanAtom.mdct_value = float(value);
        
        self.meanAtom.waveform /= sqrt(sum(self.meanAtom.waveform**2));
        self.meanAtom.waveform *= float(value)#/float(len(self.bestAtoms))
        
#        print "Mean Value of ",float(value)#/float(len(self.bestAtoms))
        return self.meanAtom

    def computeTouchZone(self , sigIdx , atom):
        #print "Updating : " ,blockIdx
        # Each block need be recomputed on a different fraction of the signal
        if atom is not None:
            self.startingTouchedIndex[sigIdx] = atom.timePosition - atom.length/2 
            self.endingTouchedIndex[sigIdx] = atom.timePosition + 1.5*atom.length 
        # if the atom is not selected , then no update is necessary
        else:
            self.endingTouchedIndex[sigIdx] = self.startingTouchedIndex[sigIdx];

class RandomSetDico(SetDico,random_dico.RandomDico):
    """ This class handles multiple dictionaries; the best atom is selected when best explaining all the signals
        Then a refinement phase is performed late to determine the local time shifts and amplitudes 
        This class handles a set of dictionaries, one for each Jointly decomposed signals
        update method just call update in all dictionaries, just like initialize
        getBestAtom starts with retrieving all the projections, sums them all and then decide what atom is the 
        best choice. Then it is locally adapted to each signal, so a list of atoms is returned instead of one
        
        USES RSSMP
        
        """

    # parameters
    dictionaryList = [];
    sizes = [];
    tolerances = [];
#    blocksList = [];
    blocks = []
    bestCurrentBlock = None;
    maxBlockScores = [];
    nature = None; # this parameter controls the atom selection function: sum sums scores across signals
                    # prod : multiply them
                    # maximin : selects 
    
    startingTouchedIndex = [];
    endingTouchedIndex = [];
    TsSequence = None;
    
    def __init__(self, sizes, useC = True , selectNature = 'sum' , tol =None , nonLinear=False, params=None):
        " Create an set of dictionaries, with blocks and everything."
        self.useC = useC;
        self.sizes = sizes;
        self.type = type;
        self.projections = None;
        self.nature = selectNature;
        self.isNL = nonLinear;
        if tol is not None:
            self.tolerances = tol;
        else:
            self.tolerances = [2]*len(self.sizes);
        
#        print self.sizes, self.tolerances;

        if selectNature == 'median' or selectNature == 'penalized'or selectNature == 'weighted':
            self.isNL = True
            print "NL dico detected"
            self.params = params;
            
    def initialize(self , residualSignalList):
        self.blocks = [];
        self.bestCurrentBlock = None;

        
        ######### Optimized version : 1 block to handle all projections with same window size########
        for mdctSize,tolerance in zip(self.sizes,self.tolerances):
#            print mdctSize , " Tolerance : " , tolerance
#            if tolerance > 1:
#                if not self.isNL:
#                    self.blocks.append(Block.py_pursuit_SetCCBlock(mdctSize , 
#                                                               residualSignalList,
#                                                                useC=self.useC , 
#                                                                nature= self.nature,
#                                                                tolerance = tolerance ))
#                else:
#                    self.blocks.append(Block.py_pursuit_SetNLCCBlock(mdctSize , 
#                                                               residualSignalList,
#                                                                useC=self.useC , 
#                                                                nature= self.nature,
#                                                                tolerance = tolerance ,
#                                                                lambd = self.params))
#            else:
#                print "Tolerance too weak to allow atom optimization: No adaptation assumed"
            self.blocks.append(joint_block.RandomSetBlock(mdctSize, 
                                                                 residualSignalList,
                                                                 useC=self.useC , 
                                                                 nature= self.nature,
                                                                 tolerance = tolerance))
        
        
        self.startingTouchedIndex = [0]*len(residualSignalList);
        self.endingTouchedIndex = [-1]*len(residualSignalList);
    

    def update(self , residualSignalList , iteratioNumber=0 , debug=0):       
        # Update all the blocks : Size by Size: so we can add the projections and have find the maximum
        self.maxBlockScore = 0;
        self.bestCurrentBlock = None;
        
        startingTouchedFrameList = [0]*len(residualSignalList)
        endingTouchedFrameList = [-1]*len(residualSignalList)
        
        # if parrallel library not available        
        for block in self.blocks:            
            
#            for sigIdx in range(len(residualSignalList)):
#                
##                startingTouchedFrameList[sigIdx] = 0;
##                endingTouchedFrameList[sigIdx] = -1;
##                print "DEBUG : recomputing all"
#                startingTouchedFrameList[sigIdx] = int(math.floor(self.startingTouchedIndex[sigIdx] / (block.scale/2)))
#                
#                if self.endingTouchedIndex[sigIdx] > 0:
#                    endingTouchedFrameList[sigIdx] = int(math.floor(self.endingTouchedIndex[sigIdx] / (block.scale/2) )) + 1; # TODO check this
#                else:
#                    endingTouchedFrameList[sigIdx] = -1;
                
#            print "block: " , block.scale , " : " , startingTouchedFrame , endingTouchedFrame
            block.update(residualSignalList , startingTouchedFrameList , endingTouchedFrameList ,iteratioNumber)
                
            if abs(block.maxValue) > self.maxBlockScore: 
                self.maxBlockScore = abs(block.maxValue)
                self.bestCurrentBlock = block;