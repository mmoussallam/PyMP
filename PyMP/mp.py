# This Python file uses the following encoding: utf-8

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

Greedy algorithms using the pymp objects
----------------------------------------

'''
import math
import numpy as np
# these imports are for debugging purposes
import os.path

# PyMP object imports
import PyMP.approx as Approx
from PyMP.base import BaseDico as Dico
import PyMP.signals as Signal
import PyMP.log as log
import PyMP.win_server as win_server

# this import is needed for using the fast projection written in C extension
# module
try:
    import PyMP.parallelProjections as parproj
except ImportError:
    print ''' Failed to load the parallelProjections extension module'''

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
        unpad=False,
        max_thread_num=None):
    """Matching Pursuit Decomposition    

    This is the original, plain Matching Pursuit Algorithm

    Parameters
    ----------
    orig_signal :  Signal
        the original signal (as a :class:`.Signal` object)  to decompose
    dictionary : BaseDico
        the dictionary (as a :class:`.Dico` object)  on which to decompose `orig_signal`
    target_srr : float
        a target Signal to Residual Ratio (SRR)
    max_it_num : int
        the maximum number of iteration allowed
    debug : int, optional
        requested debug level, default is 0
    pad : bool, optional
        whether to pad the signal's edges with zeroes
            
        
    Returns
    -------
    approx : Approx
        A :class:`.Approx` object encapsulating the approximant
    decay : list
        A list of residual's energy across iterations

    Other Parameters
    ----------------
    clean : bool, optional
        whether perform on the fly cleaning of atom's waveforms to 
        limit memory consumption
    silent_fail : bool, optional
        whether to allow energy increase when subtracting atoms
    debug_iteration : int, optional
        specify a specific iteration number at which the debug level 
        should be raised to maximum
    unpad : bool, optional
        whether to remove added edges zeroes after decomposition
    max_thread_num : int, optional
        the maximum number of threads allocated

    Raises
    ------
    ValueError
        If a selected atom causes the residual signal's energy to increase, see :func:`.Signal.subtract`

    Examples
    --------
    For decomposing the signal `x` on the Dictionary D at either SRR of 10 dB or using 1000 atoms:
    x must be a :class:`.Signal` and D a :class:`.Dico` object
    
    >>> approx, decay = mp.mp(x, D, 10, 1000)

    

    See Also
    --------
    greedy : a decomposition method where the update strategy (MP, OMP, GP or CMP)
        can be set as a parameter

    Notes
    -----
    This is the original Matching Pursuit Algorithm transcribed to 
    work with PyMP objects encapsulating signals :math:`x` and dictionaries :math:`\Phi`
    It builds an approximant :math:`\hat{x}` of :math:`x`
    as a linear combination of atoms from :math:`\Phi`
    
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
    _initialize_fftw(dictionary, max_thread_num)

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
    """ routine that restarts a decomposition from an existing,
    incomplete approximation """

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

        # lauches a mp loop: update projection,
        # retrieve best atom and subtract from residual

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
    """NOT FULLY TESTED treats long signals , return a collection of approximants ,
        one by segment
        You can use the FusionApprox method of the Approx class to recover a
        single approx object from this collection"""

    output_dir = output_dir + str(target_srr) + '/'

    if not os.path.exists(output_dir) and write:
        os.makedirs(output_dir)

    # test inputs
    if not isinstance(orig_longsignal, Signal.LongSignal):
        raise TypeError(
            'Signal is not a LongSignal object ')

    if not isinstance(dictionary, Dico):
        raise TypeError('Dico is not a Dico object')

    Nsegments = int(orig_longsignal.n_seg)

    # initialize output
    approximants = []
    decays = []
    # Loop on segments
    for segIdx in range(Nsegments):
        if debug > 0:
            print 'Starting work on segment ' + str(
                segIdx) + ' over ' + str(Nsegments)

        subSignal = orig_longsignal.get_sub_signal(
            segIdx, 1, True, False, 0, 0)
        approx, decay = mp(subSignal, dictionary, target_srr,
                           max_it_num, debug=debug, pad=pad)
        approximants.append(approx)
        decays.append(decay)

        # save the output in xml formatting
        if write:
            approximants[segIdx].write_to_xml('Srr' + str(target_srr) +
                                              '_Seg' + str(segIdx) +
                                              '_Over_' + str(Nsegments) +
                                              '.xml', output_dir)

    return approximants, decays


def locomp(original_signal,
           dictionary,
           target_srr,
           max_it_num,
           debug=0,
           pad=True,
           clean=False,
           silent_fail=False,
           unpad=False,
           approximate=False):
    """
    # EXPERIMENTAL: NOT FULLY TESTED Yet
    # Orthogonal Matching Pursuit and Gradient Pursuit Loop

    We try try to implement the Localized OMP/GP described by Mailhe et al in:
    Mailhé, B., Gribonval, R., Vandergheynst, P., & Bimbot, F. (2011).
    Fast orthogonal sparse approximation algorithms over local dictionaries.
    published in Signal Processing.

    This means that least square (for OMP) or gradient descent (for GP)
     is only performed on a local subdictionary
    in order to accelerate computation

    parameters
    ----------

    approximate : if True, perform Gradient Pursuit instead of locOMP

    outputs:
    -------

    same as mp

    """

    # back compatibility - use debug levels now
    if debug is not None:
        _Logger.set_level(debug)

    # optional add zeroes to the edge
    if pad:
        original_signal.pad(dictionary.get_pad())
    res_signal = original_signal.copy()

    # FFTW C code optimization
    _initialize_fftw(dictionary)

    # initialize blocks
    dictionary.initialize(res_signal)

    # initialize approximant
    current_approx = Approx.Approx(dictionary, [], original_signal)

    # residualEnergy
    res_energy_list = []

    it_number = 0
    current_srr = current_approx.compute_srr()

    # loop is same as mp except for orthogonal projection of the atom on the
    # residual
    # Decomposition loop: stopping criteria is either SNR or iteration number
    # check if signal has null energy
    if res_signal.energy == 0:
        raise ValueError(" Null signal energy ")

    res_energy_list.append(res_signal.energy)

    while (current_srr < target_srr) & (it_number < max_it_num):

        if not approximate:
            best_atom = _locomp_loop(dictionary, debug,
                                     silent_fail, unpad, res_signal,
                                     current_approx,
                                     res_energy_list, it_number)
        else:
            best_atom = _locgp_loop(dictionary, debug,
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


def greedy(orig_signal,
           dictionary,
           target_srr,
           max_it_num,
           debug=0,
           pad=True,
           clean=False,
           silent_fail=False,
           debug_iteration=-1,
           unpad=False,
           update='mp',
           max_thread_num=None):

    """Common Greedy Pursuit Loop Options are detailed below:

    Parameters
    ----------
    orig_signal :  Signal
        the original signal (as a :class:`.Signal` object)  to decompose
    dictionary : BaseDico
        the dictionary (as a :class:`.Dico` object)  on which to decompose `orig_signal`
    target_srr : float
        a target Signal to Residual Ratio (SRR)
    max_it_num : int
        the maximum number of iteration allowed
    debug : int, optional
        requested debug level, default is 0
    pad : bool, optional
        whether to pad the signal's edges with zeroes
    update : {'mp', 'omp', 'locomp', 'locgp'}, optional
        choice of update loop to consider
            
        
    Returns
    -------
    approx : Approx
        A :class:`.Approx` object encapsulating the approximant
    decay : list
        A list of residual's energy across iterations

    Other Parameters
    ----------------
    clean : bool, optional
        whether perform on the fly cleaning of atom's waveforms to 
        limit memory consumption
    silent_fail : bool, optional
        whether to allow energy increase when subtracting atoms
    debug_iteration : int, optional
        specify a specific iteration number at which the debug level 
        should be raised to maximum
    unpad : bool, optional
        whether to remove added edges zeroes after decomposition
    max_thread_num : int, optional
        the maximum number of threads allocated

    Raises
    ------
    ValueError
        If a selected atom causes the residual signal's energy to increase, see :func:`.Signal.subtract`


    Examples
    --------
    For decomposing the signal `x` on the Dictionary D at either SRR of 10 dB or using 1000 atoms:
    x must be a :class:`.Signal` and D a :class:`.Dico` object
    
    >>> approx, decay = mp.greedy(x, D, 10, 1000, 'mp')

    

    See Also
    --------
    mp : a decomposition method with plain mp only        

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
    _initialize_fftw(dictionary, max_thread_num=max_thread_num)

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

    if update not in ['mp', 'locomp', 'locgp', 'omp']:
        raise ValueError('Unrecognized update rule')

    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (current_srr < target_srr) & (it_number < max_it_num):

        if (it_number == debug_iteration):
            debug = 3
            _Logger.set_level(3)

        if update == 'mp':
            best_atom = _mp_loop(dictionary, debug,
                                 silent_fail, unpad, res_signal,
                                 current_approx,
                                 res_energy_list, it_number)
        elif update == 'locomp':
            best_atom = _locomp_loop(dictionary, debug,
                                     silent_fail, unpad, res_signal,
                                     current_approx,
                                     res_energy_list, it_number)
        elif update == 'omp':
            best_atom = _omp_loop(dictionary, debug,
                                  silent_fail, unpad, res_signal,
                                  current_approx,
                                  res_energy_list, it_number)
        if update == 'locgp':
            best_atom = _locgp_loop(dictionary, debug,
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


def _itprint_(it_num, best_atom):
    """ debug printing"""
    strO = ("It: " + str(it_num) + " Selected atom" + str(best_atom))
#    print strO
    return strO


def _plot_proj(best_atom, residual_sig):
    import matplotlib.pyplot as plt
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
    #=========================================================================    
    # Compute inner products and selects the best atom
    dictionary.update(res_signal, it_number)
    # retrieve the best correlated atom
    best_atom = dictionary.get_best_atom(debug)
    if best_atom is None:
        print 'No atom selected anymore'
#            return current_approx, res_energy
    if debug > 0:
        _Logger.debug(_itprint_(it_number, best_atom))
    try:
        res_signal.subtract(best_atom, debug)
        dictionary.compute_touched_zone(best_atom)
    except ValueError:
        if not silent_fail:
            _Logger.error("Something wrong happened at iteration " +
                          str(it_number) + " atom substraction abandonned")
            print "Atom Selected: ", best_atom
            print res_signal.length, best_atom.time_position, best_atom.length            
            if debug > 1:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(res_signal.data[best_atom.time_position:
                         best_atom.time_position + best_atom.length],'o-')
                plt.plot(best_atom.waveform,'x--')
                plt.plot(res_signal.data[best_atom.time_position:
                                         best_atom.time_position + best_atom.length] - best_atom.waveform, '+:')
                plt.legend(('Signal', 'Atom substracted', 'residual'))
                plt.title('Iteration ' + str(it_number))
                dictionary.best_current_block.plot_proj_matrix()
                plt.show()

            raise ValueError()
        else:
            _Logger.warning("Something wrong happened at iteration %d ATTEMPT TO CONTINUE" % it_number)
            # attempt to continue after recomputing all projections
            dictionary.compute_touched_zone(best_atom, panic=True)
            SILENT_FAIL = False
            return _mp_loop(dictionary, debug, SILENT_FAIL, unpad,
                            res_signal, current_approx, res_energy, it_number)

#            return current_approx, res_energy
    if debug > 1:
        _Logger.debug(
            "new residual energy of " + str(np.sum(res_signal.data ** 2)))
    if not unpad:
        res_energy.append(res_signal.energy)
    else:
        padd = dictionary.get_pad()
        # assume padding is max dictionaty size
        res_energy.append(np.sum(res_signal.data[padd:-padd] ** 2))
        # only compute the energy without the padded borders where
    # eventually energy has been created
    # add atom to dictionary
    current_approx.add(best_atom)
    return best_atom


def _get_local_subdico(best_atom, current_approx, dictionary):
    # THIS IS CRITICAL STEP
    # The interval is defined so that any atom that overlaps the selected one
    # is selected
    t_interv = [int(best_atom.time_position - dictionary.get_pad()),
                int(best_atom.time_position + best_atom.length)]
    t_interv[0] = max(t_interv[0], 0)
    t_interv[1] = min(t_interv[1], current_approx.length -
                      dictionary.get_pad())
# ov_sel_atoms = current_approx.filter_atoms(time_interv=t_interv,
# index=True)
    ov_sel_atoms = current_approx.get_neighbors(best_atom)

    _Logger.debug(
        "%d neighbouring atoms selected in [%d - %d]" % (len(ov_sel_atoms),
                                                         t_interv[0],
                                                         t_interv[1]))

    # now build the subdictionary matrix
#    subdico = scipy.sparse.csr_matrix((t_interv[1] + dictionary.get_pad(
#    ) - t_interv[0], len(ov_sel_atoms) + 1))
    subdico = np.zeros((t_interv[1] + dictionary.get_pad(
    ) - t_interv[0], len(ov_sel_atoms) + 1))

    subdico[best_atom.time_position - t_interv[0]:
            best_atom.time_position - t_interv[0] + best_atom.length,
            -1] = _PyServer.get_waveform(best_atom.length, best_atom.freq_bin)

    for atomidx in range(len(ov_sel_atoms)):
        at = current_approx.atoms[ov_sel_atoms[atomidx]]

#        wf = at.get_waveform() / np.sqrt(np.sum(at.get_waveform() ** 2))

        subdico[
            at.time_position - t_interv[0]: at.time_position - t_interv[0] + at.length,
            atomidx] = _PyServer.get_waveform(at.length, at.freq_bin)

    return subdico, t_interv, ov_sel_atoms

# def _grad_proj(locproj, ov_sel_atoms, best_atom, t_interv, dict, current_approx):
#    """ avoids the numpy dot product, slow when numpy is not parallelized
#        also there are a lot of zeroes in the subdictionary, no need to take them
#        into account"""
#    # initialize output
#    outvec =  np.zeros((t_interv[1] + dict.get_pad() - t_interv[0], ))
#
#    # projection loop
#    for at_i in range(len(ov_sel_atoms)):
#        at = current_approx[ov_sel_atoms[at_i]]
#        wf = _PyServer.get_waveform( at.length, at.freq_bin)
#        slice = range(int(at.time_position - t_interv[0]),
#                      int(at.time_position - t_interv[0] + at.length))
#        outvec[slice] += locproj[at_i]*wf
#
#    # don't forget the last atom
#    wf = _PyServer.get_waveform( best_atom.length, best_atom.freq_bin)
#    slice = range(int(best_atom.time_position - t_interv[0]),
#                  int(best_atom.time_position - t_interv[0] + best_atom.length))
#    outvec[slice] += locproj[-1]*wf
#
#    return outvec


def _locomp_loop(dictionary, debug, silent_fail,
                 unpad, res_signal, current_approx,
                 res_energy, it_number):
    #===========================================================================
    # # Internal OMP loop with local least squares projection
    #=========================================================================

    # Compute inner products and selects the best atom
    # SAME AS MP
    dictionary.update(res_signal, it_number)
    # retrieve the best correlated atom
    best_atom = dictionary.get_best_atom(debug)
    if best_atom is None:
        _Logger.info('No atom selected anymore')

    if debug > 0:
        _Logger.debug(_itprint_(it_number, best_atom))

    # This is where things change a little:
    # first we need to retrieve all atoms previsously
    # selected that overlaps with the selected one
    subdico, t_interv, ov_sel_atoms = _get_local_subdico(
        best_atom, current_approx, dictionary)

    # recompute the weights by least squares on the residual signal
    weights = np.dot(np.linalg.pinv(subdico), res_signal.data[t_interv[
                     0]:t_interv[1] + dictionary.get_pad()])

    # update the approximant: change atom values
    current_approx.update(ov_sel_atoms, weights)

    # update the approximant waveform, locally
    current_approx.recomposed_signal.data[t_interv[0]:
                                          t_interv[1] + dictionary.get_pad()] += np.dot(subdico, weights)

    # TODO remove or optimize
    current_approx.recomposed_signal.energy = np.sum(
        current_approx.recomposed_signal.data ** 2)

    # add atom to dictionary: but not the waveform, already taken care of
    current_approx.add(best_atom, noWf=True)

    # update the residual, locally
    res_signal.data = current_approx.original_signal.data - \
        current_approx.recomposed_signal.data
    res_signal.energy = np.sum(res_signal.data ** 2)

    # compute the touched zone to accelerate further processings
    dictionary.compute_touched_zone(best_atom)

    if debug > 1:
        _Logger.debug(
            "new residual energy of " + str(np.sum(res_signal.data ** 2)))
    if not unpad:
        res_energy.append(res_signal.energy)
    else:
        padd = dictionary.get_pad()
        # assume padding is max dictionaty size
        res_energy.append(np.sum(res_signal.data[padd:-padd] ** 2))
        # only compute the energy without the padded borders where

    return best_atom


def _locgp_loop(dictionary, debug, silent_fail,
                unpad, res_signal, current_approx,
                res_energy, it_number):
    #===========================================================================
    # # Internal GP loop with local gradient updates
    #=========================================================================

    # Compute inner products and selects the best atom
    # SAME AS MP
    dictionary.update(res_signal, it_number)
    # retrieve the best correlated atom
    best_atom = dictionary.get_best_atom(debug)
    if best_atom is None:
        _Logger.info('No atom selected anymore')

    if debug > 0:
        _Logger.debug(_itprint_(it_number, best_atom))

    # Same as for locOMP : build neighbouring subdictionary
    # selected that overlaps with the selected one
    subdico, t_interv, ov_sel_atoms = _get_local_subdico(
        best_atom, current_approx, dictionary)

    # recompute the weights by least squares on the residual signal
    # Replace the pseudo-inverse by a gradient descent step
    if 'LO' in dictionary.nature:
        sigslice = res_signal.data[t_interv[0]:t_interv[1] +
                                   dictionary.get_pad()]

#         @TODO can we replace this step by a projection using fftw?
#         We don't have to do that, we already calculated them!!
# the projection score are all stored in the projection_matrix of the
# dictionary
        locProj = np.dot(subdico.T, sigslice)

    # HACK faster if weights have already been computed
    else:
        loc_projs2 = []
        for at_i in ov_sel_atoms:
            at = current_approx[at_i]
            block = dictionary.blocks[dictionary.find_block_by_scale(at.length)]
            loc_projs2.append(
                block.projs_matrix[at.frame * at.length / 2 + at.freq_bin])
        loc_projs2.append(best_atom.mdct_value)
        locProj = np.array(loc_projs2)

#    grad_proj = _grad_proj(locProj, ov_sel_atoms,
#                           best_atom, t_interv, dictionary, current_approx)
#    print subdico.shape, locProj.shape
    # projection

    grad_proj = np.dot(subdico, locProj)
    # refactoring: replace by a broadcasted computation since
    # all we need is the energy of the projected array
#    grad_proj_en = np.sum(np.sum((subdico*locProj),axis=1)**2)

#    print grad_proj_en - np.sum(grad_proj**2)

#    step = np.sum(locProj**2)/ np.sum(np.sum((subdico*locProj),axis=1)**2)
    step = np.sum(locProj ** 2) / np.sum(grad_proj ** 2)
#    print step
    # compute new weights
    weights = step * locProj

    # update the approximant: change atom values
    # @TODO optimize here: this is bringing serious overhead : DONE
    current_approx.update(ov_sel_atoms, weights)

    # update the approximant waveform, locally: np.dot(subdico,weights)
    # REFACTORED, no need to recompute dot product
    current_approx.recomposed_signal.data[t_interv[0]:
                                          t_interv[1] + dictionary.get_pad()] += step * grad_proj

    # @TODO remove or optimize
    current_approx.recomposed_signal.energy = np.sum(
        current_approx.recomposed_signal.data ** 2)

    # add atom to dictionary: but not the waveform, already taken care of
    current_approx.add(best_atom, noWf=True)

    # update the residual, locally
    res_signal.data = current_approx.original_signal.data - \
        current_approx.recomposed_signal.data
    res_signal.energy = np.sum(res_signal.data ** 2)

    # compute the touched zone to accelerate further processings
    dictionary.compute_touched_zone(best_atom)

    if debug > 1:
        _Logger.debug(
            "new residual energy of " + str(np.sum(res_signal.data ** 2)))
    if not unpad:
        res_energy.append(res_signal.energy)
    else:
        padd = dictionary.get_pad()
        # assume padding is max dictionaty size
        res_energy.append(np.sum(res_signal.data[padd:-padd] ** 2))
        # only compute the energy without the padded borders where

    return best_atom


def _omp_loop(dictionary, debug, silent_fail,
              unpad, res_signal, current_approx,
              res_energy, it_number):
    #===========================================================================
    # # Internal OMP loop with local gradient updates
    # BEWARE: VERY MEMORY CONSUMING!
    #=========================================================================

    # Compute inner products and selects the best atom
    # SAME AS MP
    dictionary.update(res_signal, it_number)
    # retrieve the best correlated atom
    best_atom = dictionary.get_best_atom(debug)
    if best_atom is None:
        _Logger.info('No atom selected anymore')

    if debug > 0:
        _Logger.debug(_itprint_(it_number, best_atom))

    if it_number == 0:
        dictionary.matrix = np.zeros((current_approx.length, 1))
    else:
        # add the current waveform to the subdictionary:
        dictionary.matrix = np.concatenate((dictionary.matrix,
                                            np.zeros((dictionary.matrix.shape[0], 1))), axis=1)

    t_interv = [int(best_atom.time_position - dictionary.get_pad()),
                int(best_atom.time_position + best_atom.length)]
    t_interv[0] = max(t_interv[0], 0)
    t_interv[1] = min(t_interv[1], current_approx.length -
                      dictionary.get_pad())
    # fill the new column with new waveform
    dictionary.matrix[best_atom.time_position:
                      best_atom.time_position +
                      best_atom.length, -1] = _PyServer.get_waveform(best_atom.length,
                                                                     best_atom.freq_bin)

    subdico = dictionary.matrix[:, 0:it_number + 1]

    # recompute the weights by least squares on the residual signal
#    print subdico.shape, res_signal.data.shape
    weights = np.dot(np.linalg.pinv(subdico), res_signal.data)

    # update the approximant: change atom values
    current_approx.update(range(it_number), weights)

    # update the approximant waveform, locally
    current_approx.recomposed_signal.data += np.dot(subdico, weights)

    # TODO remove or optimize
    current_approx.recomposed_signal.energy = np.sum(
        current_approx.recomposed_signal.data ** 2)

    # add atom to dictionary: but not the waveform, already taken care of
    current_approx.add(best_atom, noWf=True)

    # update the residual, locally
    res_signal.data = current_approx.original_signal.data - \
        current_approx.recomposed_signal.data
    res_signal.energy = np.sum(res_signal.data ** 2)

    # compute the touched zone to accelerate further processings
    dictionary.compute_touched_zone(best_atom)

    if debug > 1:
        _Logger.debug(
            "new residual energy of " + str(np.sum(res_signal.data ** 2)))
    if not unpad:
        res_energy.append(res_signal.energy)
    else:
        padd = dictionary.get_pad()
        # assume padding is max dictionaty size
        res_energy.append(np.sum(res_signal.data[padd:-padd] ** 2))
        # only compute the energy without the padded borders where

    return best_atom


def _initialize_fftw(dictionary, max_thread_num=None):
    if max_thread_num is not None:
        max_thread_flag = max_thread_num
    else:
        max_thread_flag = -1
    if parproj.initialize_plans(np.array(dictionary.sizes),
                                np.array([2] * len(dictionary.sizes)),
                                max_thread_flag) != 1:
        raise ValueError("Something failed during FFTW initialization step ")


def _clean_fftw():
    if parproj.clean_plans() != 1:
        raise ValueError("Something failed during FFTW cleaning stage ")


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
        if parproj.initialize_plans(np.array(dictionary.sizes), np.array(dictionary.tolerances)) != 1:
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

                    dictionary.compute_touched_zone(
                        sigIdx, best_atom_list[sigIdx])

                else:
                    _Logger.debug("Atom not subtracted in this signal")
                    dictionary.compute_touched_zone(
                        sigIdx, best_atom_list[sigIdx])

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
    if parproj.clean_plans(np.array(dictionary.sizes)) != 1:
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
