#*****************************************************************************/
#                                                                            */
#                               mp.py                                        */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */


'''
Module mp
=========

A Simple mp algorithm using the pymp objects
--------------------------------------------

'''
import math
import numpy as np

# these imports are for debugging purposes
import matplotlib.pyplot as plt
import os.path

# PyMP object imports
import approx as Approx
from base import BaseDico as Dico
import signals as Signal
import log
import win_server

# this import is needed for using the fast projection written in C extension
# module
try:
    import parallelProjections
except ImportError:
    print ''' Failed to load the parallelProjections extension module'''
# from MatchingPursuit.TwoD.py_pursuit_2DApprox import py_pursuit_2DApprox
# declare gloabl waveform server

_PyServer = win_server.get_server()
_Logger = log.Log('mp', noContext=True)


def mp(orig_signal,
        dictionary,
        target_srr,
        max_it_num,
        debug=0,
        pad=True,
        clean=False,
        silent_fail=False,
        debug_iteration=-1,
        unpad=False):
    """Common Matching Pursuit Loop Options are detailed below:

    Args:

         `orig_signal`:    the original signal (as a :class:`.Signal` object) :math:`x` to decompose

         `dictionary`:        the dictionary (as a :class:`.Dico` object) :math:`\Phi` on which to decompose :math:x

         `target_srr`:         a target Signal to Residual Ratio

         `max_it_num`: maximum number of iteration allowed

    Returns:

         `approx`:  A :class:`.Approx` object encapsulating the approximant

         `decay`:   A list of residual's energy across iterations

    Example::

        >>> approx, decay = mp.mp(x, D, 10, 1000)

    For decomposing the signal x on the Dictionary D at either SRR of 10 dB or using 1000 atoms:
    x must be a :class:`.Signal` and D a :class:`.Dico` object

    """

    # back compatibility - use debug levels now
    if debug is not None:
        _Logger.set_level(debug)

    # CHECKING INPUTS
    if not isinstance(orig_signal, Signal.Signal):
        raise TypeError("MP will only accept objects inheriting from PyMP.signals.Signal as input signals")
    
    
    if not isinstance(dictionary, Dico):
        raise TypeError("MP will only accept objects inheriting from PyMP.base.BaseDico as dictionaries")
    
    # optional add zeroes to the edge
    if pad:
        orig_signal.pad(dictionary.get_pad())
    res_signal = orig_signal.copy()

    # FFTW Optimization for C code: initialize module global variables
    _initialize_fftw(dictionary)
    
    # initialize blocks
    dictionary.initialize(res_signal)

    # initialize approximant
    current_approx = Approx.Approx(
        dictionary, [], orig_signal, debug_level=debug)

    # residualEnergy
    res_energy_list = []

    it_number = 0
    current_srr = current_approx.compute_srr()

    # check if signal has null energy
    if res_signal.energy == 0:
        raise ValueError(" Null signal energy ")
        
    res_energy_list.append(res_signal.energy)

    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (current_srr < target_srr) & (it_number < max_it_num):

        if (it_number == debug_iteration):
            debug = 3
            _Logger.set_level(3)


        best_atom = _mp_loop(dictionary, debug,
                             silent_fail, unpad, res_signal,
                             current_approx,
                             res_energy_list, it_number)

        # compute new SRR and increment iteration Number
        current_srr = current_approx.compute_srr(res_signal)

        _Logger.debug("SRR reached of " + str(current_srr) +
                      " at iteration " + str(it_number))

        it_number += 1

        # cleaning for memory consumption control
        if clean:
            del best_atom.waveform

    # VERY IMPORTANT CLEANING STAGE!
    _clean_fftw()

    return current_approx, res_energy_list

def mp_continue(current_approx,
                orig_signal,
                dictionary,
                target_srr,
                max_it_num,
                debug=0,
                pad=True,
                silent_fail=False,
                unpad=False):
    """ routine that restarts a decomposition from an existing, incomplete approximation """

    if not isinstance(current_approx, Approx.Approx):
        raise TypeError("provided object is not a py_pursuit_Approx")

    # optional add zeroes to the edge
    if pad:
        orig_signal.pad(dictionary.get_pad())

    res_signal = orig_signal.copy()

    # retrieve approximation from residual
    if current_approx.recomposed_signal is not None:
        res_signal.data -= current_approx.recomposed_signal.data
        res_signal.energy = sum(res_signal.data ** 2)
    else:
        for atom in current_approx.atoms:
            res_signal.subtract(atom, debug)

    _initialize_fftw(dictionary)

    # initialize blocks
    dictionary.initialize(res_signal)

    # residualEnergy
    res_energy_list = [res_signal.energy]

    it_num = 0
    current_srr = current_approx.compute_srr()
        
    
    while (current_srr < target_srr) & (it_num < max_it_num): 
        
        # lauches a mp loop: update projection, retrieve best atom and subtract from residual
#        _mp_loop(dictionary, current_approx,
#              res_signal, res_energy_list,
#              it_num, debug)
        
        _mp_loop(dictionary, debug, 
                 silent_fail, unpad, 
                 res_signal, current_approx, 
                 res_energy_list, it_num)
    
        # compute new SRR and increment iteration Number
        current_srr = current_approx.compute_srr(res_signal)
        it_num += 1
    
        if debug > 0:
            print "SRR reached of ", current_srr, " at iteration ", it_num

    # VERY IMPORTANT CLEANING STAGE!
    _clean_fftw()

    return current_approx, res_energy_list

def mp_long(orig_longsignal,
            dictionary,
            target_srr=10,
            max_it_num=100,
            debug=False,
            pad=True,
            output_dir='',
            write=False):
    """NOT FULLY TESTED treats long signals , return a collection of approximants , one by segment
        You can use the FusionApprox method of the Approx class to recover a single approx object from this
        collection"""

    output_dir = output_dir + str(target_srr) + '/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # test inputs
    if not isinstance(orig_longsignal, Signal.LongSignal):
        raise TypeError(
            'Signal is not a py_pursuitLongSignal object - TODO correct')

    if not isinstance(dictionary, Dico):
        raise TypeError('Dico is not a py_pursuit_Dico object - TODO correct')

    Nsegments = int(orig_longsignal.n_seg)

    # initialize output
    approximants = []
    decays = []
    # Loop on segments
    for segIdx in range(Nsegments):
        if debug > 0:
            print 'Starting work on segment ' + str(
                segIdx) + ' over ' + str(Nsegments)
# subSignal = orig_longsignal.get_sub_signal( segIdx , 1 ,forceMono=True,
# doNormalize=False , pad = (dictionary.get_pad())/2)
        subSignal = orig_longsignal.get_sub_signal(
            segIdx, 1, True, False, 0, 0)
        approx, decay = mp(subSignal, dictionary, target_srr,
                           max_it_num, debug=debug, pad=pad)
        approximants.append(approx)
        decays.append(decay)

        # save the output in xml formatting
        if write:
            approximants[segIdx].write_to_xml('Srr' + str(target_srr) +
                                              '_Seg' + str(segIdx) + '_Over_' + str(Nsegments) + '.xml', output_dir)

    return approximants, decays


def _itprint_(it_num, best_atom):
    """ debug printing"""
    strO = ("It: " + str(it_num) + " Selected atom" + str(best_atom))
    print strO


def _plot_proj(best_atom, residual_sig):
    plt.figure()
    plt.plot(residual_sig.data[best_atom.
                                 time_position: best_atom.time_position + best_atom.length])
    plt.plot(best_atom.waveform)
    plt.plot(residual_sig.data[best_atom.time_position: best_atom.time_position +
                                 best_atom.length] - best_atom.waveform, ':')
    plt.legend(('Signal', 'Atom substracted', 'residual'))
    plt.show()


def _mp_loop(dictionary, debug, silent_fail,
             unpad, res_signal, current_approx,
             res_energy, it_number):
    #===========================================================================
    # # Internal MP loop
    #===========================================================================
    
    # Compute inner products and selects the best atom
    dictionary.update(res_signal, it_number)
    # retrieve the best correlated atom
    best_atom = dictionary.get_best_atom(debug)
    if best_atom is None:
        print 'No atom selected anymore'
#            return current_approx, res_energy
    if debug > 0:
        print _itprint_(it_number, best_atom)
        
    try:
        res_signal.subtract(best_atom, debug)
        dictionary.compute_touched_zone(best_atom)
    except ValueError:
        if not silent_fail:
            _Logger.error("Something wrong happened at iteration " + str(it_number) + " atom substraction abandonned")
            print "Atom Selected: ", best_atom
                  
            if debug > 1:
                plt.figure()
                plt.plot(res_signal.data[best_atom.time_position:best_atom.time_position + best_atom.length])
                plt.plot(best_atom.waveform)
                plt.plot(res_signal.data[best_atom.time_position:best_atom.time_position + best_atom.length] - best_atom.waveform, ':')
                plt.legend(('Signal', 'Atom substracted', 'residual'))
                plt.title('Iteration ' + str(it_number))
                dictionary.best_current_block.plot_proj_matrix()
                plt.show()
        
            raise ValueError()
        else:
            _Logger.warning("Something wrong happened at iteration %d ATTEMPT TO CONTINUE"%it_number)
            # attempt to continue after recomputing all projections
            dictionary.compute_touched_zone(best_atom, panic=False)
            SILENT_FAIL = False
            return _mp_loop(dictionary, debug, SILENT_FAIL, unpad, res_signal, current_approx, res_energy, it_number)
              
#            return current_approx, res_energy
    if debug > 1:
        _Logger.debug("new residual energy of " + str(sum(res_signal.data ** 2)))
    if not unpad:
        res_energy.append(res_signal.energy)
    else:
        padd = dictionary.get_pad()
        # assume padding is max dictionaty size
        res_energy.append(np.sum(res_signal.data[padd:-padd] ** 2)) # only compute the energy without the padded borders where
    # eventually energy has been created
    # add atom to dictionary
    current_approx.add(best_atom)
    return best_atom

#def _loop(dictionary, current_approx,
#          residual_sig, res_energy_list,
#          it_num, debug):
#            
#        # Compute inner products and selects the best atom
#        dictionary.update(residual_sig)
#    
#        # Retrieve best atom among candidates
#        bestAtom = dictionary.get_best_atom(debug)
#    
#        if bestAtom is None:
#            print 'No atom selected anymore'            
#    
#        if debug > 0:
#            _itprint_(it_num, bestAtom)
#    
#        try:
#            residual_sig.subtract(bestAtom, debug)
#            dictionary.compute_touched_zone(bestAtom)
#        
#        except ValueError:
#            print "Something wrong happened at iteration ", it_num, " atom substraction abandonned"
#            if debug > 1:
#                _plot_proj(bestAtom, residual_sig)
#                
#            approxPath = "currentApprox_failed_iteration_" + str(
#                it_num) + ".pymp"
#            signalPath = "currentApproxRecomposedSignal_failed_iteration_" + str(
#                it_num) + ".wav"
#            current_approx.dump(approxPath)
#            current_approx.recomposed_signal.write(signalPath)
#            print " approx saved to ", approxPath
#            print " recomposed signal saved to ", signalPath
#            return current_approx, res_energy_list
#    
#        if debug > 0:
#            print "new residual energy of ", sum(residual_sig.data ** 2)
#    
#        # BUGFIX?
#        res_energy_list.append(residual_sig.energy)
#    #        res_energy_list.append(sum(residual_sig.dataVec **2))
#    
#        # add atom to dictionary
#        current_approx.add(bestAtom)
#            
#        # cleaning
#        del bestAtom.waveform


def _initialize_fftw(dictionary):    
    if parallelProjections.initialize_plans(np.array(dictionary.sizes), np.array([2]*len(dictionary.sizes))) != 1:
        raise ValueError("Something failed during FFTW initialization step ")

def _clean_fftw():
    if parallelProjections.clean_plans() != 1:
        raise ValueError("Something failed during FFTW cleaning stage ")






# Experimental


# def GP(originalSignal,
#        dictionary,
#        target_srr,
#        max_it_num,
#        debug=0,
#        pad=True,
#        itClean=False, doWatchSRR=False):
#    # EXPERIMENTAL: NOT TESTED! AND DEPRECATED USE AT YOUR OWN RISKS
#    #Gradient Pursuit Loop """
#    #==========================================================================
#    # from scipy.sparse import lil_matrix
#    #==========================================================================
#    # back compatibility - use debug levels now
#    if not debug:
#        doWatchSRR = True
#        debug = 0
#
#    # optional add zeroes to the edge
#    if pad:
#        originalSignal.pad(dictionary.get_pad())
#    residualSignal = originalSignal.copy()
#
#    # initialize blocks
#    dictionary.initialize(residualSignal)
#
#    # initialize approximant
#    currentApprox = Approx.Approx(dictionary, [], originalSignal)
#
#    # residualEnergy
#    resEnergy = []
#
#    iterationNumber = 0
#    approxSRR = currentApprox.compute_srr()
#
##    projMatrix = lil_matrix((currentApprox.length , max_it_num))
##    projMatrix = np.zeros((currentApprox.length , max_it_num))
#    columnIndexes = []
#    projMatrix = []
#    gradient = []
#    G = []
#    indexes = []
#
#    gains = []
#    # loop is same as mp except for orthogonal projection of the atom on the
#    # residual
#    while (approxSRR < target_srr) & (iterationNumber < max_it_num):
#        maxBlockScore = 0
#        bestBlock = None
#
#        if iterationNumber % 100 == 0:
#            print "reached iteration ", iterationNumber
#
#        # Compute inner products and selects the best atom
#        dictionary.update(residualSignal, iterationNumber)
#
#        # retrieve best correlated atom
#        bestAtom = dictionary.get_best_atom(debug)
#
#        if bestAtom is None:
#            print 'No atom selected anymore'
#            return currentApprox, resEnergy
#
#        if debug > 0:
#            strO = ("It: " + str(iterationNumber) + " Selected atom of scale " + str(bestAtom.length) + " frequency bin " + str(bestAtom.frequencyBin)
#                                            + " at " + str(
#                                                bestAtom.timePosition)
#                                            + " value : " + str(
#                                                bestAtom.mdct_value)
#                                            + " time shift : " + str(bestAtom.timeShift))
#            print strO
#
#        newIndex = dictionary.getAtomKey(bestAtom, currentApprox.length)
#
#        # add the new index to the subset of indexes
#        isNew = False
#        if not newIndex in indexes:
#            indexes.append(newIndex)
#            isNew = True
#
#        # retrive the gradient from the previously computed inner products
#        gradient = np.array(
#            dictionary.getProjections(indexes, currentApprox.length))
#
#        if isNew:
##            columnIndexes.append(iterationNumber);
## projMatrix[ bestAtom.timePosition : bestAtom.timePosition+ bestAtom.length,
## columnIndexes[-1]] = bestAtom.waveform/bestAtom.mdct_value
#                # add atom to projection matrix
#            vec1 = np.concatenate((np.zeros((bestAtom.timePosition, 1)), bestAtom.
#                waveform.reshape(bestAtom.length, 1) / bestAtom.mdct_value))
#            vec2 = np.zeros((originalSignal.length - bestAtom.
#                timePosition - bestAtom.length, 1))
#            atomVec = np.concatenate((vec1, vec2))
##            atomVec = atomVec/math.sqrt(sum(atomVec**2))
#            if iterationNumber > 0:
#
#                projMatrix = np.concatenate((projMatrix, atomVec), axis=1)
#                gains = np.concatenate((gains, np.zeros((1,))))
#
#            else:
#                projMatrix = atomVec
#                gains = 0
#
#            # add it to the collection
#            currentApprox.add(bestAtom)
#
#        # calcul c : direction
##        c =projMatrix[: , columnIndexes].tocsc() * gradient;
#        c = np.dot(projMatrix, gradient)
##        spVec[indexes] = gradient;
##        c = dictionary.synthesize( spVec , currentApprox.length)
#
#        # calcul step size
#        alpha = np.dot(residualSignal.data, c) / np.dot(c.T, c)
#
#        # update residual
#        gains += alpha * gradient
#
#        if doWatchSRR:
#            # recreate the approximation
#            currentApprox.recomposed_signal.data = np.dot(
#                projMatrix, gains)
## currentApprox.recomposedSignal.dataVec = gains * projMatrix[: ,
## columnIndexes].tocsc().T;
#        # update residual
##        substract(residualSignal , alpha , c)
#        residualSignal.data -= alpha * c
## residualSignal.dataVec = originalSignal.dataVec -
## currentApprox.recomposedSignal.dataVec
#
#        if debug > 1:
#            print alpha
#            print gradient
##            print bestAtom.mdct_value
##            print gains
#
#            plt.figure()
#            plt.plot(originalSignal.data, 'b--')
#            plt.plot(currentApprox.recomposed_signal.data, 'k-')
#            plt.plot(residualSignal.data, 'r:')
#            plt.plot(alpha * c, 'g')
#            plt.legend(
#                ('original', 'recomposed so far', 'residual', 'subtracted'))
#            plt.show()
#
## resEnergy.append(sum((originalSignal.dataVec -
## currentApprox.recomposedSignal.dataVec)**2))
#        resEnergy.append(
#            np.dot(residualSignal.data.T, residualSignal.data))
#
#        if doWatchSRR:
#            recomposedEnergy = np.dot(currentApprox.recomposed_signal.
#                data.T, currentApprox.recomposed_signal.data)
#
#            # compute new SRR and increment iteration Number
#            approxSRR = 10 * math.log10(recomposedEnergy / resEnergy[-1])
#            if debug > 0:
#                print "SRR reached of ", approxSRR, " at iteration ", iterationNumber
#
#        iterationNumber += 1
#
#        # cleaning
#        if itClean:
#            del bestAtom.waveform
#
#    # recreate the approxs
##    for atom ,i in zip(currentApprox.atoms , range(currentApprox.atomNumber)):
##        atom.mdct_value = gains[i];
#
#    return currentApprox, resEnergy

# def substract(residualSignal , alpha , c):
#    residualSignal.dataVec -= alpha*c;
#    return residualSignal


# def OMP(originalSignal,
#        dictionary,
#        target_srr,
#        max_it_num,
#        debug=0,
#        pad=True,
#        itClean=False):
#    # EXPERIMENTAL: NOT TESTED! AND DEPRECATED USE AT YOUR OWN RISKS
#    # Orthogonal Matching Pursuit Loop """
#
#    # back compatibility - use debug levels now
#    if not debug:
#        debug = 0
#
#    # optional add zeroes to the edge
#    if pad:
#        originalSignal.pad(dictionary.get_pad())
#    residualSignal = originalSignal.copy()
#
#    # FFTW C code optimization
#    if parallelProjections.initialize_plans(np.array(dictionary.sizes)) != 1:
#            raise ValueError(
#                "Something failed during FFTW initialization step ")
#
#    # initialize blocks
#    dictionary.initialize(residualSignal)
#
#    # initialize approximant
#    currentApprox = Approx.Approx(dictionary, [], originalSignal)
#
#    # residualEnergy
#    resEnergy = []
#
#    iterationNumber = 0
#    approxSRR = currentApprox.compute_srr()
#
##    projMatrix = np.zeros((originalSignal.length,1))
#    projMatrix = []
#    atomKeys = []
#    # loop is same as mp except for orthogonal projection of the atom on the
#    # residual
#    while (approxSRR < target_srr) & (iterationNumber < max_it_num):
#        maxBlockScore = 0
#        bestBlock = None
#
#        # Compute inner products and selects the best atom
#        dictionary.update(residualSignal, iterationNumber)
#
#        bestAtom = dictionary.get_best_atom(debug)
#
#        if bestAtom is None:
#            print 'No atom selected anymore'
#            return currentApprox, resEnergy
#
#        if debug > 0:
#            strO = ("It: " + str(iterationNumber) + " Selected atom of scale " + str(bestAtom.length) + " frequency bin " + str(bestAtom.frequencyBin)
#                                            + " at " + str(
#                                                bestAtom.timePosition)
#                                            + " value : " + str(
#                                                bestAtom.mdct_value)
#                                            + " time shift : " + str(bestAtom.timeShift))
#            print strO
#
#        # need to recompute all the atoms projection scores to orthogonalise
#        # residual
#        # approximate with moore-penrose inverse
#        # add atom to projection matrix
#        vec1 = np.concatenate((np.zeros((bestAtom.timePosition, 1)), bestAtom.
#            waveform.reshape(bestAtom.length, 1) / bestAtom.mdct_value))
#        vec2 = np.zeros((originalSignal.length - bestAtom.
#            timePosition - bestAtom.length, 1))
#        atomVec = np.concatenate((vec1, vec2))
#
##        atomVec = atomVec/math.sqrt(sum(atomVec**2))
#        atomKey = (
#            bestAtom.length, bestAtom.timePosition, bestAtom.frequencyBin)
#
#        if iterationNumber > 0:
#            if atomKey not in atomKeys:
#                projMatrix = np.concatenate((projMatrix, atomVec), axis=1)
#                atomKeys.append(atomKey)
#        else:
#            projMatrix = atomVec
#            atomKeys.append(atomKey)
#
#        # Orthogonal projection via pseudo inverse calculation
#        ProjectedScores = np.dot(np.linalg.pinv(projMatrix),
#            original_signal.data.reshape(originalSignal.length, 1))
#
#        # update residual
#        currentApprox.recomposed_signal.data = np.dot(
#            projMatrix, ProjectedScores)[:, 0]
#
#        residualSignal.data = originalSignal.data - currentApprox.recomposed_signal.data
#
##        # add atom to dictionary
##        currentApprox.add(bestAtom , dictionary.bestCurrentBlock.wLong)
##
## for atom,i in zip(currentApprox.atoms , range(currentApprox.atomNumber)):
##            atom.projectionScore = ProjectedScores[i];
## atom.waveform = (ProjectedScores[i]/math.sqrt(sum(atom.waveform**2))) *
## atom.waveform;
#
##        residualSignal.subtract(bestAtom , debug)
##        dictionary.compute_touched_zone(bestAtom)
#        resEnergy.append(
#            np.dot(residualSignal.data.T, residualSignal.data))
#        recomposedEnergy = np.dot(currentApprox.recomposed_signal.
#            data.T, currentApprox.recomposed_signal.data)
#
#        # compute new SRR and increment iteration Number
#        approxSRR = 10 * math.log10(recomposedEnergy / resEnergy[-1])
##        approxSRR = currentApprox.compute_srr();
#        if debug > 0:
#            print "SRR reached of ", approxSRR, " at iteration ", iterationNumber
#
#        iterationNumber += 1
#
#        # cleaning
#        if itClean:
#            del bestAtom.waveform
#
#    # VERY IMPORTANT CLEANING STAGE!
#    if parallelProjections.clean_plans(np.array(dictionary.sizes)) != 1:
#        raise ValueError("Something failed during FFTW cleaning stage ")
#
#    return currentApprox, resEnergy


def mp_joint(orig_sig_list,
             dictionary,
             target_srr,
             max_it_num,
             debug=0,
             pad=True,
             escape=False,
             escape_threshold=0.4,
             escape_thr_map=None,
             bss_eval=False,
             sources=None,
             interval=100,
             clean=True,
             waitbar=True,
             no_adapt=False,
             silent_fail=False):
    """ Joint Matching Pursuit : Takes a bunch of signals in entry and decomposes the common part out of them
        Gives The common model and the sparse residual for each signal in return """

    # back compatibility - use debug levels now
    if debug is not None:
        _Logger.set_level(debug)
    _Logger.info("Call to mp.mp_joint")
    # We now work on a list of signals: we have a list of approx and residuals
    res_sig_list = []
    current_approx_list = []
    res_energy_list = []
    current_srr_list = []

    if escape:
        esc_approx_list = []
        esc_it_list = []
        criterions = []

#    if bss_eval:
#        SDRs = [];
#        SIRs = [];
#        SARs = [];
#    dictionaryList = []
    threshold_dict = {}
    if escape_thr_map is None:
        for size in dictionary.sizes:
            threshold_dict[size] = escape_threshold
    else:
        for size, value in zip(dictionary.sizes, escape_thr_map):
            threshold_dict[size] = value

    # create a mean approx of the background
    mean_approx = Approx.Approx(
        dictionary, [], orig_sig_list[0], debug_level=debug)
    k = 1
    for orig_signal in orig_sig_list:

        _Logger.debug("Initializing Signal Number " + str(k))

#        if pad:
#            orig_signal.pad(dictionary.get_pad())
        res_sig_list.append(orig_signal.copy())

        # initialize approximant
        current_approx_list.append(
            Approx.Approx(dictionary, [], orig_signal, debug_level=debug))

        if escape:
            esc_approx_list.append(Approx.Approx(dictionary, [
            ], orig_signal, debug_level=debug))
            esc_it_list.append([])
        # residualEnergy
        res_energy_list.append([])
        current_srr_list.append(current_approx_list[-1].compute_srr())
        k += 1

    # initialize blocks using the first signal: they shoudl all have the same
    # length
    _Logger.debug("Initializing Dictionary")
    dictionary.initialize(res_sig_list)

    # FFTW Optimization for C code: initialize module global variables
    try:
        if parallelProjections.initialize_plans(np.array(dictionary.sizes), np.array(dictionary.tolerances)) != 1:
            raise ValueError(
                "Something failed during FFTW initialization step ")
    except:
        _Logger.error("Initialization step failed")
        raise

    iterationNumber = 0

    approxSRR = max(current_srr_list)

    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (approxSRR < target_srr) & (iterationNumber < max_it_num):

        _Logger.info("mp LOOP : iteration " + str(iterationNumber + 1))
        # Compute inner products and selects the best atom
        dictionary.update(res_sig_list, iterationNumber)

        if debug > 0:
            maxScale = dictionary.best_current_block.scale
            maxFrameIdx = math.floor(
                dictionary.best_current_block.maxIdx / (0.5 * maxScale))
            maxBinIdx = dictionary.best_current_block.maxIdx - maxFrameIdx * (
                0.5 * maxScale)

            _Logger.debug("It: " + str(iterationNumber) + " Selected atom "
                          + str(dictionary.best_current_block.maxIdx)
                          + " of scale " + str(maxScale) + " frequency bin "
                          + str(maxBinIdx)
                          + " value : " + str(
                          dictionary.max_block_score)
                          + " Frame : " + str(maxFrameIdx))

        # retrieve the best correlated atoms, locally adapted to the signal
        best_atom_list = dictionary.get_best_atom(debug, noAdapt=no_adapt)

        if best_atom_list is None:
            print 'No atom selected anymore'
            raise ValueError('Failed to select an atom')

        escape_current_atom = False

        if escape:
            # Escape mechanism : if variance of amplitudes is too big : assign
            # it to the biggest only
            mean = np.mean([abs(atom.get_value()) for atom in best_atom_list])
            std = np.std([abs(atom.get_value()) for atom in best_atom_list])
            maxValue = np.max(
                [abs(atom.get_value()) for atom in best_atom_list])
            _Logger.debug("Mean : " + str(mean) + " - STD : " + str(std))

            criterions.append(std / mean)
#            print criterions[-1]
            if (std / mean) > threshold_dict[atom.length]:
                escape_current_atom = True
# print "Escaping Iteration ",iterationNumber,": mean " , mean , " std " , std
                _Logger.debug("Escaping!!")

        for sigIdx in range(len(res_sig_list)):

            if not escape_current_atom:
                # add atom to current regular approx
                current_approx_list[sigIdx].add(
                    best_atom_list[sigIdx], clean=False)

                dictionary.compute_touched_zone(sigIdx, best_atom_list[sigIdx])

                # subtract atom from residual
                try:
                    res_sig_list[
                        sigIdx].subtract(best_atom_list[sigIdx], debug)
                except ValueError:
                    if silent_fail:
                        continue
                    else:
                        raise ValueError("Subtraction of atom failed")
            else:
                # Add this atom to the escape approx only if this signal is a
                # maxima
                if abs(best_atom_list[sigIdx].get_value()) == maxValue:
#                if True:
#                    print "Added Atom to signal " + str(sigIdx)
                    esc_approx_list[sigIdx].add(best_atom_list[sigIdx])
                    current_approx_list[sigIdx].add(best_atom_list[sigIdx])
                    esc_it_list[sigIdx].append(iterationNumber)

                    # subtract atom from residual
                    res_sig_list[
                        sigIdx].subtract(best_atom_list[sigIdx], debug)

                    dictionary.compute_touched_zone(sigIdx, best_atom_list[sigIdx])

                else:
                    _Logger.debug("Atom not subtracted in this signal")
                    dictionary.compute_touched_zone(sigIdx, best_atom_list[sigIdx])

            # update energy decay curves
            res_energy_list[sigIdx].append(res_sig_list[sigIdx].energy)

            if debug > 0 or (iterationNumber % interval == 0):
                current_srr_list[sigIdx] = current_approx_list[
                    sigIdx].compute_srr(res_sig_list[sigIdx])

                _Logger.debug("Local adaptation of atom " + str(sigIdx) +
                              " - Position : " + str(best_atom_list[sigIdx].time_position) +
                              " Amplitude : " + str(best_atom_list[sigIdx].proj_score) +
                              " TimeShift : " + str(best_atom_list[sigIdx].time_shift))
#            if clean and sigIdx>0:
#                del best_atom_list[sigIdx].waveform;

        # also add the mean atom to the background model UNLESS this is an
        # escaped atom
        if not escape_current_atom:
#            mean_approx.add(best_atom_list[0] )
            mean_approx.add(
                dictionary.get_mean_atom(getFirstAtom=False), clean=clean)
            _Logger.debug("Atom added to common rep ")

        if clean:
            for sigIdx in range(len(res_sig_list)):
                del best_atom_list[sigIdx].waveform

#        dictionary.compute_touched_zone()

#        approxSRR = currentApprox.compute_srr();

        _Logger.debug("SRRs reached of " + str(current_srr_list) +
                      " at iteration " + str(iterationNumber))

#        if bss_eval and ( (iterationNumber+1) % interval ==0):
#            estimSources = np.zeros(sources.shape)
#            # first estim the source for the common part
#            estimSources[0, :,0] = mean_approx.recomposedSignal.dataVec
#
#            for sigIdx in range(len(res_sig_list)):
# estimSources[sigIdx+1, :,0] =
# current_approx_list[sigIdx].recomposedSignal.dataVec +
# esc_approx_list[sigIdx].recomposedSignal.dataVec;
# estimSources[sigIdx+1, :,0] =
# esc_approx_list[sigIdx].recomposedSignal.dataVec;
#            [SDR,ISR,SIR,SAR] = bss_eval_images_nosort(estimSources,sources)
#
##            print SDR , SIR
#            SDRs.append(SDR)
#            SIRs.append(SIR)
#            SARs.append(SAR)
#
##            print current_srr_list

        iterationNumber += 1
        if (iterationNumber % interval == 0):
            print iterationNumber
            print [resEnergy[-1] for resEnergy in res_energy_list]
    # VERY IMPORTANT CLEANING STAGE!
    if parallelProjections.clean_plans(np.array(dictionary.sizes)) != 1:
        raise ValueError("Something failed during FFTW cleaning stage ")
#    if waitbar and (iterationNumber %(max_it_num/100) ==0):
#        print float(iterationNumber)/float(max_it_num/100) , "%",

    if not escape:
        return mean_approx, current_approx_list, res_energy_list, res_sig_list
    else:
        # time to add the escaped atoms to the corresponding residuals:
#        for sigIdx in range(len(res_sig_list)):
# res_sig_list[sigIdx].dataVec +=
# esc_approx_list[sigIdx].recomposedSignal.dataVec;
#        if bss_eval:
# return mean_approx, current_approx_list, res_energy_list , res_sig_list ,
# esc_approx_list , criterions , SDRs, SIRs , SARs
        return mean_approx, current_approx_list, res_energy_list, res_sig_list, esc_approx_list, esc_it_list
