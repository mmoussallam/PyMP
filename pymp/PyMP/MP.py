#*****************************************************************************/
#                                                                            */
#                               MP.py                                        */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
#                                                                            */
#                                                                            */
#  This program is free software; you can redistribute it and/or             */
#  modify it under the terms of the GNU General Public License               */
#  as published by the Free Software Foundation; either version 2            */
#  of the License, or (at your option) any later version.                    */
#                                                                            */
#  This program is distributed in the hope that it will be useful,           */
#  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
#  GNU General Public License for more details.                              */
#                                                                            */
#  You should have received a copy of the GNU General Public License         */
#  along with this program; if not, write to the Free Software               */
#  Foundation, Inc., 59 Temple Place - Suite 330,                            */
#  Boston, MA  02111-1307, USA.                                              */
#                                                                            */
#******************************************************************************

'''
Module MP 
=========

A Simple MP algorithm using the pymp objects
--------------------------------------------

'''


#from MatchingPursuit import py_pursuit_Atom as Atom 
from Classes import pymp_Approx as Approx
from Classes import pymp_Dico as Dico
from Classes import pymp_Signal as Signal
from Classes import PyWinServer as Server
from Classes import pymp_Log as Log
import math;
#from numpy import math
import numpy as np

# these imports are for debugging purposes 
import matplotlib.pyplot as plt
import os.path

# this import is needed for using the fast projection written in C extension module
try:
    import parallelProjections
except ImportError:
    print ''' Failed to load the parallelProjections extension module
            Please run parallelProjections/install_and_test.py for further details'''
#from MatchingPursuit.TwoD.py_pursuit_2DApprox import py_pursuit_2DApprox
# declare gloabl waveform server
global _PyServer , _Logger
_PyServer = Server.PyServer()
_Logger = Log.pymp_Log('MP' , noContext=True)

def MP( originalSignal , 
        dictionary ,  
        targetSRR ,  
        maxIteratioNumber , 
        debug=0 , 
        padSignal=True ,
        itClean=False ,
         silentFail=False,
         plot = False,
         debugSpecific=-1,
         cutBorders=False):
    """Common Matching Pursuit Loop Options are detailed below:
    
    Args:
    
         `originalSignal`:    the original signal (as a :class:`.pymp_Signal` object) :math:`x` to decompose
         
         `dictionary`:        the dictionary (as a :class:`.pymp_Dico` object) :math:`\Phi` on which to decompose :math:x
         
         `targetSRR`:         a target Signal to Residual Ratio
         
         `maxIteratioNumber`: maximum number of iteration allowed
         
    Returns:
    
         `approx`:  A :class:`.pymp_Approx` object encapsulating the approximant
         
         `decay`:   A list of residual's energy across iterations
    
    Example::
                
        >>> approx,decay = MP.MP(x, D, 10, 1000)
        
    For decomposing the signal x on the Dictionary D at either SRR of 10 dB or using 1000 atoms:
    x must be a :class:`.pymp_Signal` and D a :class:`.pymp_Dico` object
    
    """
    
    # back compatibility - use debug levels now
    if debug is not None: 
        _Logger.setLevel(debug)
    
    # optional add zeroes to the edge
    if padSignal:
        originalSignal.pad(dictionary.getN())
    residualSignal = originalSignal.copy();

    # FFTW Optimization for C code: initialize module global variables
    try:
        if parallelProjections.initialize_plans(np.array(dictionary.sizes), np.array([2 for i in dictionary.sizes])) != 1:
            raise ValueError("Something failed during FFTW initialization step ");
    except:
        _Logger.error("Initialization step failed");
        raise;
    # initialize blocks
    dictionary.initialize(residualSignal)
    
    # initialize approximant
    currentApprox = Approx.pymp_Approx(dictionary, [], originalSignal, debugLevel=debug);    
    
    # residualEnergy
    resEnergy = []
    
    iterationNumber = 0;
    approxSRR = currentApprox.computeSRR();
    
    # check if signal has null energy
    if residualSignal.energy == 0:
        _Logger.info(" Null signal energy ")
        return currentApprox , resEnergy
    resEnergy.append(residualSignal.energy)
        
    if plot:
        plt.ion()
        fig = plt.figure()
        ax1 = plt.subplot(211);
        ax2 = plt.subplot(212);
    
    
    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (approxSRR < targetSRR) & (iterationNumber < maxIteratioNumber):
        maxBlockScore = 0;
        bestBlock = None;
        
        if (iterationNumber ==debugSpecific):
            debug=3;
            _Logger.setLevel(3)        
            
        # Compute inner products and selects the best atom
        dictionary.update(residualSignal , iterationNumber )     
        
        # retrieve the best correlated atom
        bestAtom = dictionary.getBestAtom(debug) ;
        
        if bestAtom is None :
            print 'No atom selected anymore'
            return currentApprox , resEnergy
                  
        if debug>0:
            # TODO reformulate for both 1D and 2D
            _Logger.debug("It: "+ str(iterationNumber)+ " Selected atom "+str(dictionary.bestCurrentBlock.maxIdx)+" of scale " + str(bestAtom.length) + " frequency bin " + str(bestAtom.frequencyBin) 
                                            + " at " + str(bestAtom.timePosition) 
                                            + " value : "  + str(bestAtom.getAmplitude())
                                            + " time shift : " +  str(bestAtom.timeShift) 
                                            + " Frame : " + str(bestAtom.frame));
#                                            + " K = " + str(np.sqrt(bestAtom.mdct_value**2 / residualSignal.energy)))
            if bestAtom.phase is not None:
                _Logger.debug('    phase of : ' + str(bestAtom.phase));
        
        try:            
            residualSignal.subtract(bestAtom , debug)                            
            dictionary.computeTouchZone(bestAtom)
        except ValueError:
            if not silentFail:
                _Logger.error("Something wrong happened at iteration " + str(iterationNumber) + " atom substraction abandonned")
                # TODO refactoring
                print "Atom scale : " , bestAtom.length ," , frequency Bin: " ,bestAtom.frequencyBin," , location : " ,bestAtom.timePosition, " ,Time Shift : " ,bestAtom.timeShift , " , value : " , bestAtom.getAmplitude(); 
            if debug>1:
                plt.figure()
                plt.plot(residualSignal.dataVec[bestAtom.timePosition : bestAtom.timePosition + bestAtom.length ])
                plt.plot(bestAtom.waveform)
                plt.plot(residualSignal.dataVec[bestAtom.timePosition : bestAtom.timePosition + bestAtom.length ] - bestAtom.waveform , ':')
                plt.legend(('Signal' , 'Atom substracted', 'residual'))
                plt.title('Iteration ' + str(iterationNumber))
                
                dictionary.bestCurrentBlock.plotScores()
                plt.show()
            return currentApprox , resEnergy

            
        if debug>1:
            _Logger.debug("new residual energy of " + str(sum(residualSignal.dataVec **2)))
        
        if not cutBorders:
            resEnergy.append(residualSignal.energy)
        else:
            # only compute the energy without the padded borders wher eventually energy has been created
            padd = 256; # assume padding is max dictionaty size
            resEnergy.append(np.sum(residualSignal.dataVec[padd:-padd]**2))
        
        # add atom to dictionary        
        currentApprox.addAtom(bestAtom , dictionary.bestCurrentBlock.getWindow())

        # compute new SRR and increment iteration Number
        approxSRR = currentApprox.computeSRR(residualSignal);
        
        if plot:            
            ax1.clear()
#            plt.subplot(121)
            plt.title('Iteration : ' + str(iterationNumber) + ' , SRR : ' + str(approxSRR))
            ax1.plot(currentApprox.recomposedSignal.dataVec)
            ax2.clear()
            
            plt.draw()
        
        _Logger.debug("SRR reached of " + str(approxSRR) + " at iteration " + str(iterationNumber))

        iterationNumber += 1;
        
        # cleaning
        if itClean:
            del bestAtom.waveform
    
    # VERY IMPORTANT CLEANING STAGE!
    if parallelProjections.clean_plans(np.array(dictionary.sizes)) != 1:
        raise ValueError("Something failed during FFTW cleaning stage ");
    
    # end of loop output the approximation and the residual energies
    if plot:
        plt.ioff()
            
    return currentApprox , resEnergy





def MPContinue(currentApprox , originalSignal , dictionary ,  targetSRR ,  maxIteratioNumber , debug=0 , padSignal=True ):
    """ routine that restarts a decomposition from an existing, incomplete approximation """
    
    if not isinstance(currentApprox, Approx.pymp_Approx):
        raise TypeError("provided object is not a py_pursuit_Approx");
            
    # optional add zeroes to the edge
    if padSignal:
        originalSignal.pad(dictionary.getN())
    
    residualSignal = originalSignal.copy();
    
    # retrieve approximation from residual
    if currentApprox.recomposedSignal is not None:
        residualSignal.dataVec -=  currentApprox.recomposedSignal.dataVec
        residualSignal.energy = sum(residualSignal.dataVec**2);
    else:
        for atom in currentApprox.atoms:
            residualSignal.subtract(atom, debug)
    
    try:
        if parallelProjections.initialize_plans(np.array(dictionary.sizes), np.array([2 for i in dictionary.sizes])) != 1:
            raise ValueError("Something failed during FFTW initialization step ");
    except:
        _Logger.error("Initialization step failed");
        raise;
    
    # initialize blocks
    dictionary.initialize(residualSignal)
         
    # residualEnergy
    resEnergy = [residualSignal.energy]
    
    iterationNumber = 0;
    approxSRR = currentApprox.computeSRR();
    
    while (approxSRR < targetSRR) & (iterationNumber < maxIteratioNumber):
        maxBlockScore = 0;
        bestBlock = None;
        # Compute inner products and selects the best atom
        dictionary.update(residualSignal)     
        
        # new version - also computes the waveform by inverse mdct
        # new version - also computes max cross correlation and adjust atom's waveform timeshift
        bestAtom = dictionary.getBestAtom(debug)
        
        if bestAtom is None :
            print 'No atom selected anymore'
            return currentApprox , resEnergy
                  
        if debug>0:
            strO = ("It: "+ str(iterationNumber)+ " Selected atom of scale " + str(bestAtom.length) + " frequency bin " + str(bestAtom.frequencyBin) 
                                            + " at " + str(bestAtom.timePosition) 
                                            + " value : "  + str(bestAtom.mdct_value)
                                            + " time shift : " +  str(bestAtom.timeShift))
            print strO
        
        try:
            residualSignal.subtract(bestAtom , debug)
            dictionary.computeTouchZone(bestAtom)
        except ValueError:
            print "Something wrong happened at iteration " ,iterationNumber , " atom substraction abandonned"
            if debug>1:
                plt.figure()
                plt.plot(residualSignal.dataVec[bestAtom.timePosition : bestAtom.timePosition + bestAtom.length ])
                plt.plot(bestAtom.waveform)
                plt.plot(residualSignal.dataVec[bestAtom.timePosition : bestAtom.timePosition + bestAtom.length ] - bestAtom.waveform , ':')
                plt.legend(('Signal' , 'Atom substracted', 'residual'))
                plt.show()
            approxPath = "currentApprox_failed_iteration_" + str(iterationNumber) + ".xml"
            signalPath = "currentApproxRecomposedSignal_failed_iteration_" + str(iterationNumber) + ".wav"
            currentApprox.writeToXml(approxPath)
            currentApprox.recomposedSignal.write(signalPath)
            print " approx saved to " , approxPath
            print " recomposed signal saved to " , signalPath    
            return currentApprox , resEnergy
        
        if debug>1:
            plt.figure()
            plt.plot(originalSignal.dataVec)
            plt.plot(residualSignal.dataVec , '--')
            plt.legend(("original", "residual at iteration " + str(iterationNumber)))
            plt.show()
            
        if debug>0:
            print "new residual energy of " , sum(residualSignal.dataVec **2)
        
        # BUGFIX? 
        resEnergy.append(residualSignal.energy)
#        resEnergy.append(sum(residualSignal.dataVec **2))
        
        
        # add atom to dictionary        
        currentApprox.addAtom(bestAtom , dictionary.bestCurrentBlock.wLong)
                  
        # compute new SRR and increment iteration Number
#        approxSRR = currentApprox.computeSRR(residualSignal);
        approxSRR = currentApprox.computeSRR();
        iterationNumber += 1;
        
        if debug>0:
            print "SRR reached of " , approxSRR , " at iteration " , iterationNumber
        
        # cleaning
        del bestAtom.waveform
    
    # VERY IMPORTANT CLEANING STAGE!
    if parallelProjections.clean_plans(np.array(dictionary.sizes)) != 1:
        raise ValueError("Something failed during FFTW cleaning stage ");
        
    return currentApprox , resEnergy
    

def MP_LongSignal(originalLongSignal , dictionary , targetSRR = 10 , maxIteratioNumber=100 , debug=False , padSignal =True , outputDir='',doWrite=False):
    """NOT FULLY TESTED treats long signals , return a collection of approximants , one by segment 
        You can use the FusionApprox method of the pymp_Approx class to recover a single approx object from this 
        collection"""
    
    outputDir = outputDir + str(targetSRR) + '/'
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    # test inputs
    if not isinstance(originalLongSignal , Signal.pymp_LongSignal):
        raise TypeError('Signal is not a py_pursuitLongSignal object - TODO correct')
    
    if not isinstance(dictionary , Dico.pymp_Dico):
        raise TypeError('Dico is not a py_pursuit_Dico object - TODO correct')
    
    Nsegments = int(originalLongSignal.segmentNumber)
    
    # initialize output
    approximants = [];
    decays = [];
    # Loop on segments
    for segIdx in range(Nsegments):
        if debug >0:
            print 'Starting work on segment ' +str(segIdx)+ ' over ' + str(Nsegments)
#        subSignal = originalLongSignal.getSubSignal( segIdx , 1 ,forceMono=True, doNormalize=False , padSignal = (dictionary.getN())/2)
        subSignal = originalLongSignal.getSubSignal( segIdx , 1 ,True, False , 0 , 0)
        approx , decay = MP(subSignal, dictionary, targetSRR, maxIteratioNumber, debug=debug, padSignal=padSignal)
        approximants.append(approx)
        decays.append(decay)
        
        # save the output in xml formatting
        if doWrite:
            approximants[segIdx].writeToXml('Srr' + str(targetSRR) + '_Seg'+str(segIdx)+'_Over_'+str(Nsegments)+'.xml', outputDir)

    return approximants , decays

# Experimental 
def GP( originalSignal , 
        dictionary ,  
        targetSRR ,  
        maxIteratioNumber , 
        debug=0 , 
        padSignal=True ,
        itClean=False , doWatchSRR=False):
    # EXPERIMENTAL: NOT TESTED! AND DEPRECATED USE AT YOUR OWN RISKS 
    #Gradient Pursuit Loop """
    #===========================================================================
    # from scipy.sparse import lil_matrix
    #===========================================================================
    # back compatibility - use debug levels now
    if not debug: 
        doWatchSRR = True
        debug =0;
    
    # optional add zeroes to the edge
    if padSignal:
        originalSignal.pad(dictionary.getN())
    residualSignal = originalSignal.copy();
    
    # initialize blocks
    dictionary.initialize(residualSignal)
    
    # initialize approximant
    currentApprox = Approx.pymp_Approx(dictionary, [], originalSignal);    
    
    # residualEnergy
    resEnergy = []
    
    iterationNumber = 0;
    approxSRR = currentApprox.computeSRR();
    
    
#    projMatrix = lil_matrix((currentApprox.length , maxIteratioNumber))
#    projMatrix = np.zeros((currentApprox.length , maxIteratioNumber))
    columnIndexes = [];
    projMatrix = [];
    gradient  =[]
    G  = [];
    indexes = []
    
    gains = [];
    # loop is same as MP except for orthogonal projection of the atom on the residual
    while (approxSRR < targetSRR) & (iterationNumber < maxIteratioNumber):
        maxBlockScore = 0;
        bestBlock = None;

        if iterationNumber % 100 == 0:
            print "reached iteration " , iterationNumber

        # Compute inner products and selects the best atom
        dictionary.update(residualSignal , iterationNumber)     
        
        # retrieve best correlated atom
        bestAtom = dictionary.getBestAtom(debug ) ;
        

        if bestAtom is None :
            print 'No atom selected anymore'
            return currentApprox , resEnergy
                  
        if debug>0:
            strO = ("It: "+ str(iterationNumber)+ " Selected atom of scale " + str(bestAtom.length) + " frequency bin " + str(bestAtom.frequencyBin) 
                                            + " at " + str(bestAtom.timePosition) 
                                            + " value : "  + str(bestAtom.mdct_value)
                                            + " time shift : " +  str(bestAtom.timeShift))
            print strO

        newIndex = dictionary.getAtomKey(bestAtom , currentApprox.length);

        # add the new index to the subset of indexes
        isNew = False
        if not newIndex in indexes:
            indexes.append(newIndex)
            isNew = True;
        
        # retrive the gradient from the previously computed inner products
        gradient = np.array(dictionary.getProjections( indexes  , currentApprox.length))
        
        if isNew:
#            columnIndexes.append(iterationNumber);
#            projMatrix[ bestAtom.timePosition : bestAtom.timePosition+ bestAtom.length, columnIndexes[-1]] = bestAtom.waveform/bestAtom.mdct_value
                # add atom to projection matrix
            vec1 = np.concatenate( (np.zeros((bestAtom.timePosition,1)) , bestAtom.waveform.reshape(bestAtom.length,1)/bestAtom.mdct_value  ))
            vec2 = np.zeros((originalSignal.length - bestAtom.timePosition - bestAtom.length,1))
            atomVec = np.concatenate( (vec1 , vec2) ) 
#            atomVec = atomVec/math.sqrt(sum(atomVec**2))
            if iterationNumber>0 :
                
                projMatrix = np.concatenate((projMatrix , atomVec) , axis=1)
                gains = np.concatenate((gains , np.zeros((1,)) ))

            else:                
                projMatrix = atomVec;
                gains = 0

            # add it to the collection
            currentApprox.addAtom(bestAtom)
            
        # calcul c : direction
#        c =projMatrix[: , columnIndexes].tocsc() * gradient;
        c = np.dot( projMatrix , gradient);
#        spVec[indexes] = gradient;
#        c = dictionary.synthesize( spVec , currentApprox.length)    
        
        # calcul step size
        alpha = np.dot(residualSignal.dataVec , c)/np.dot(c.T , c);
        
        # update residual
        gains += alpha*gradient            
        
        if doWatchSRR:
            # recreate the approximation
            currentApprox.recomposedSignal.dataVec = np.dot(projMatrix , gains );
#            currentApprox.recomposedSignal.dataVec = gains * projMatrix[: , columnIndexes].tocsc().T;
        # update residual
#        substract(residualSignal , alpha , c)
        residualSignal.dataVec -= alpha*c;
#        residualSignal.dataVec = originalSignal.dataVec - currentApprox.recomposedSignal.dataVec
        
        
        
        if debug>1:
            print alpha 
            print gradient 
#            print bestAtom.mdct_value 
#            print gains
            
            plt.figure()
            plt.plot(originalSignal.dataVec,'b--')
            plt.plot(currentApprox.recomposedSignal.dataVec,'k-')
            plt.plot(residualSignal.dataVec,'r:')
            plt.plot(alpha*c , 'g')
            plt.legend(('original' ,'recomposed so far' , 'residual' , 'subtracted'))
            plt.show()
            
#        resEnergy.append(sum((originalSignal.dataVec - currentApprox.recomposedSignal.dataVec)**2))
        resEnergy.append(np.dot(residualSignal.dataVec.T , residualSignal.dataVec));
        
        if doWatchSRR:
            recomposedEnergy = np.dot(currentApprox.recomposedSignal.dataVec.T , currentApprox.recomposedSignal.dataVec);
                    
            # compute new SRR and increment iteration Number
            approxSRR = 10*math.log10( recomposedEnergy/ resEnergy[-1] ) 
            if debug>0:
                print "SRR reached of " , approxSRR , " at iteration " , iterationNumber

        iterationNumber += 1;
        
        # cleaning
        if itClean:
            del bestAtom.waveform
    
    # recreate the approxs
#    for atom ,i in zip(currentApprox.atoms , range(currentApprox.atomNumber)):
#        atom.mdct_value = gains[i];
    
    return currentApprox , resEnergy

#def substract(residualSignal , alpha , c):
#    residualSignal.dataVec -= alpha*c;
#    return residualSignal

def OMP( originalSignal , 
        dictionary ,  
        targetSRR ,  
        maxIteratioNumber , 
        debug=0 , 
        padSignal=True ,
        itClean=False ):
    # EXPERIMENTAL: NOT TESTED! AND DEPRECATED USE AT YOUR OWN RISKS
    # Orthogonal Matching Pursuit Loop """
    
    
    # back compatibility - use debug levels now
    if not debug: 
        debug =0;
    
    # optional add zeroes to the edge
    if padSignal:
        originalSignal.pad(dictionary.getN())
    residualSignal = originalSignal.copy();
    
    # FFTW C code optimization
    if parallelProjections.initialize_plans(np.array(dictionary.sizes)) != 1:
            raise ValueError("Something failed during FFTW initialization step ");
    
    # initialize blocks
    dictionary.initialize(residualSignal)
    
    # initialize approximant
    currentApprox = Approx.pymp_Approx(dictionary, [], originalSignal);    
    
    # residualEnergy
    resEnergy = []
    
    iterationNumber = 0;
    approxSRR = currentApprox.computeSRR();
    
#    projMatrix = np.zeros((originalSignal.length,1))
    projMatrix = []
    atomKeys = [];
    # loop is same as MP except for orthogonal projection of the atom on the residual
    while (approxSRR < targetSRR) & (iterationNumber < maxIteratioNumber):
        maxBlockScore = 0;
        bestBlock = None;

        # Compute inner products and selects the best atom
        dictionary.update(residualSignal , iterationNumber)     
        
        bestAtom = dictionary.getBestAtom(debug ) ;
        
        if bestAtom is None :
            print 'No atom selected anymore'
            return currentApprox , resEnergy
                  
        if debug>0:
            strO = ("It: "+ str(iterationNumber)+ " Selected atom of scale " + str(bestAtom.length) + " frequency bin " + str(bestAtom.frequencyBin) 
                                            + " at " + str(bestAtom.timePosition) 
                                            + " value : "  + str(bestAtom.mdct_value)
                                            + " time shift : " +  str(bestAtom.timeShift))
            print strO
                
 
        # need to recompute all the atoms projection scores to orthogonalise residual
        # approximate with moore-penrose inverse
        
        # add atom to projection matrix
        vec1 = np.concatenate( (np.zeros((bestAtom.timePosition,1)) , bestAtom.waveform.reshape(bestAtom.length,1)/bestAtom.mdct_value  ))
        vec2 = np.zeros((originalSignal.length - bestAtom.timePosition - bestAtom.length,1))
        atomVec = np.concatenate( (vec1 , vec2) ) 
        
#        atomVec = atomVec/math.sqrt(sum(atomVec**2))
        atomKey = (bestAtom.length,bestAtom.timePosition,bestAtom.frequencyBin);

        if iterationNumber>0:
            if atomKey not in atomKeys:
                projMatrix = np.concatenate((projMatrix , atomVec) , axis=1)
                atomKeys.append(atomKey)
        else:
            projMatrix = atomVec;
            atomKeys.append(atomKey)
        
        # Orthogonal projection via pseudo inverse calculation
        ProjectedScores = np.dot(np.linalg.pinv(projMatrix),originalSignal.dataVec.reshape(originalSignal.length,1))
        
        # update residual
        currentApprox.recomposedSignal.dataVec  = np.dot(projMatrix, ProjectedScores)[:,0]
        
        residualSignal.dataVec = originalSignal.dataVec - currentApprox.recomposedSignal.dataVec
        
#        # add atom to dictionary        
#        currentApprox.addAtom(bestAtom , dictionary.bestCurrentBlock.wLong)       
#        
#        for atom,i in zip(currentApprox.atoms , range(currentApprox.atomNumber)):
#            atom.projectionScore = ProjectedScores[i];
#            atom.waveform = (ProjectedScores[i]/math.sqrt(sum(atom.waveform**2))) *  atom.waveform;
        
        
#        residualSignal.subtract(bestAtom , debug)
#        dictionary.computeTouchZone(bestAtom)
            
        resEnergy.append(np.dot(residualSignal.dataVec.T ,residualSignal.dataVec ))
        recomposedEnergy = np.dot(currentApprox.recomposedSignal.dataVec.T , currentApprox.recomposedSignal.dataVec);
                
        # compute new SRR and increment iteration Number
        approxSRR = 10*math.log10( recomposedEnergy/ resEnergy[-1] )  
#        approxSRR = currentApprox.computeSRR();
        if debug>0:
            print "SRR reached of " , approxSRR , " at iteration " , iterationNumber

        iterationNumber += 1;
        
        
        
        # cleaning
        if itClean:
            del bestAtom.waveform
        
    # VERY IMPORTANT CLEANING STAGE!
    if parallelProjections.clean_plans(np.array(dictionary.sizes)) != 1:
        raise ValueError("Something failed during FFTW cleaning stage ");
    
    return currentApprox , resEnergy


def JointMP( originalSignalList , 
        dictionary ,  
        targetSRR ,  
        maxIteratioNumber , 
        debug=0 , 
        padSignal=True,
        escape = False,
        Threshold = 0.4,
        ThresholdMap = None,
        doEval=False,
        sources=None,
        interval=100,
        doClean = True,
        waitbar=True,
        noAdapt=False,
        silentFail=False):
    """ Joint Matching Pursuit : Takes a bunch of signals in entry and decomposes the common part out of them
        Gives The common model and the sparse residual for each signal in return """
    
    # back compatibility - use debug levels now
    if debug is not None: 
        _Logger.setLevel(debug)
    _Logger.info("Call to MP.JointMP")
    # We now work on a list of signals: we have a list of approx and residuals    
    residualSignalList = [];
    currentApproxList = [];
    
    
    resEnergyList = [];
    
    approxSRRList = [];

    if escape:
        escapeApproxList = [];
        escItList = [];
        criterions = [];
    if doEval:
        SDRs = [];
        SIRs = [];
        SARs = [];
#    dictionaryList = []
    ThresholdDict = {};
    if ThresholdMap is None:        
        for size in dictionary.sizes:
            ThresholdDict[size] = Threshold;
    else:
        for size,value in zip(dictionary.sizes,ThresholdMap):
            ThresholdDict[size] = value;
    
    # create a mean approx of the background
    meanApprox = Approx.pymp_Approx(dictionary, [], originalSignalList[0],  debugLevel=debug)
    k = 1;
    for originalSignal in originalSignalList:
        
        _Logger.debug("Initializing Signal Number " + str(k))
        
#        if padSignal:
#            originalSignal.pad(dictionary.getN())    
        residualSignalList.append(originalSignal.copy());
    
        # initialize approximant
        currentApproxList.append(Approx.pymp_Approx(dictionary, [], originalSignal, debugLevel=debug));  
        
        if escape:
            escapeApproxList.append(Approx.pymp_Approx(dictionary, [], originalSignal, debugLevel=debug));  
            escItList.append([]);
        # residualEnergy
        resEnergyList.append([]);
        approxSRRList.append(currentApproxList[-1].computeSRR());
        k += 1;
    
    # initialize blocks using the first signal: they shoudl all have the same length        
    _Logger.debug("Initializing Dictionary")
    dictionary.initialize(residualSignalList)    
        
    # FFTW Optimization for C code: initialize module global variables
    try:
        if parallelProjections.initialize_plans(np.array(dictionary.sizes), np.array(dictionary.tolerances)) != 1:
            raise ValueError("Something failed during FFTW initialization step ");
    except:
        _Logger.error("Initialization step failed");
        raise;

    iterationNumber = 0;

    approxSRR = max(approxSRRList);
    
    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (approxSRR < targetSRR) & (iterationNumber < maxIteratioNumber):
           
        _Logger.info("MP LOOP : iteration " + str(iterationNumber+1))   
        # Compute inner products and selects the best atom
        dictionary.update( residualSignalList , iterationNumber )     
        
        if debug>0:
            maxScale = dictionary.bestCurrentBlock.scale
            maxFrameIdx = math.floor(dictionary.bestCurrentBlock.maxIdx / (0.5*maxScale));
            maxBinIdx = dictionary.bestCurrentBlock.maxIdx - maxFrameIdx * (0.5*maxScale)
            
            _Logger.debug("It: "+ str(iterationNumber)+ " Selected atom "+str(dictionary.bestCurrentBlock.maxIdx)+" of scale " + str(maxScale) + " frequency bin " + str(maxBinIdx)                                             
                                            + " value : "  + str(dictionary.maxBlockScore)                                            
                                            + " Frame : " + str(maxFrameIdx));
        
        # retrieve the best correlated atoms, locally adapted to the signal
        bestAtomList = dictionary.getBestAtom(debug,noAdapt=noAdapt) ;
        
        if bestAtomList is None :
            print 'No atom selected anymore'
            raise ValueError('Failed to select an atom')                  
        
        escapeThisAtom = False;

        if escape:
            # Escape mechanism : if variance of amplitudes is too big : assign it to the biggest only
            mean = np.mean([abs(atom.getAmplitude()) for atom in bestAtomList])
            std = np.std([abs(atom.getAmplitude()) for atom in bestAtomList])
            maxValue = np.max([abs(atom.getAmplitude()) for atom in bestAtomList])
            _Logger.debug("Mean : " +str(mean)+" - STD : " + str(std))
        
            criterions.append(std/mean)
#            print criterions[-1]
            if (std/mean) > ThresholdDict[atom.length]:
                escapeThisAtom = True;
#                print "Escaping Iteration ",iterationNumber,": mean " , mean , " std " , std
                _Logger.debug("Escaping!!")
        
        for sigIdx in range(len(residualSignalList)):    

            if not escapeThisAtom:
                # add atom to current regular approx
                currentApproxList[sigIdx].addAtom(bestAtomList[sigIdx] , clean=False)
                
                dictionary.computeTouchZone(sigIdx ,bestAtomList[sigIdx])
                
                # subtract atom from residual
                try:
                    residualSignalList[sigIdx].subtract(bestAtomList[sigIdx] , debug)
                except ValueError:
                    if silentFail:
                        continue
                    else:
                        raise ValueError("Subtraction of atom failed")
            else:
                # Add this atom to the escape approx only if this signal is a maxima
                if abs(bestAtomList[sigIdx].getAmplitude()) == maxValue:
#                if True:
#                    print "Added Atom to signal " + str(sigIdx)
                    escapeApproxList[sigIdx].addAtom(bestAtomList[sigIdx])
                    currentApproxList[sigIdx].addAtom(bestAtomList[sigIdx])
                    escItList[sigIdx].append(iterationNumber);
            
                    # subtract atom from residual
                    residualSignalList[sigIdx].subtract(bestAtomList[sigIdx] , debug)
                    
                    dictionary.computeTouchZone(sigIdx ,bestAtomList[sigIdx] )
                
                else:
                    _Logger.debug("Atom not subtracted in this signal");
                    dictionary.computeTouchZone(sigIdx , bestAtomList[sigIdx])
                    
            # update energy decay curves
            resEnergyList[sigIdx].append(residualSignalList[sigIdx].energy)
            
            if debug>0 or (iterationNumber %interval ==0):
                approxSRRList[sigIdx] = currentApproxList[sigIdx].computeSRR(residualSignalList[sigIdx])
    
                _Logger.debug("Local adaptation of atom "+str(sigIdx)+
                              " - Position : "  + str(bestAtomList[sigIdx].timePosition) + 
                              " Amplitude : " + str(bestAtomList[sigIdx].projectionScore) +
                              " TimeShift : " + str(bestAtomList[sigIdx].timeShift))
#            if doClean and sigIdx>0:
#                del bestAtomList[sigIdx].waveform;
            
        # also add the mean atom to the background model UNLESS this is an escaped atom
        if not escapeThisAtom:     
#            meanApprox.addAtom(bestAtomList[0] )
            meanApprox.addAtom(dictionary.getMeanAtom(getFirstAtom=False) , clean=doClean)
            _Logger.debug("Atom added to common rep ");
        
        if doClean:
            for sigIdx in range(len(residualSignalList)):    
                del bestAtomList[sigIdx].waveform;
         
#        dictionary.computeTouchZone()

#        approxSRR = currentApprox.computeSRR();
        
        _Logger.debug("SRRs reached of " + str(approxSRRList) + " at iteration " + str(iterationNumber))

#        if doEval and ( (iterationNumber+1) % interval ==0):            
#            estimSources = np.zeros(sources.shape)
#            # first estim the source for the common part
#            estimSources[0, :,0] = meanApprox.recomposedSignal.dataVec
#            
#            for sigIdx in range(len(residualSignalList)):
##                estimSources[sigIdx+1, :,0] = currentApproxList[sigIdx].recomposedSignal.dataVec + escapeApproxList[sigIdx].recomposedSignal.dataVec;
#                estimSources[sigIdx+1, :,0] = escapeApproxList[sigIdx].recomposedSignal.dataVec;
#            [SDR,ISR,SIR,SAR] = bss_eval_images_nosort(estimSources,sources)
#            
##            print SDR , SIR
#            SDRs.append(SDR)
#            SIRs.append(SIR)
#            SARs.append(SAR)
#
##            print approxSRRList

        iterationNumber += 1;
        if (iterationNumber %interval ==0):
            print iterationNumber
            print [resEnergy[-1] for resEnergy in resEnergyList]
    # VERY IMPORTANT CLEANING STAGE!
    if parallelProjections.clean_plans(np.array(dictionary.sizes)) != 1:
        raise ValueError("Something failed during FFTW cleaning stage ");
#    if waitbar and (iterationNumber %(maxIteratioNumber/100) ==0):
#        print float(iterationNumber)/float(maxIteratioNumber/100) , "%", 

    if not escape:
        return meanApprox, currentApproxList, resEnergyList , residualSignalList
    else:
        # time to add the escaped atoms to the corresponding residuals:
#        for sigIdx in range(len(residualSignalList)):            
#            residualSignalList[sigIdx].dataVec += escapeApproxList[sigIdx].recomposedSignal.dataVec;
        if doEval:
            return meanApprox, currentApproxList, resEnergyList , residualSignalList , escapeApproxList , criterions , SDRs, SIRs , SARs
        return meanApprox, currentApproxList, resEnergyList , residualSignalList , escapeApproxList , escItList

