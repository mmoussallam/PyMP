#*****************************************************************************/
#                                                                            */
#                               mp_coder.py                                  */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Fri Nov 16 2012  */
# -------------------------------------------------------------------------- */


"""

Module mp_coder
===============
A collection of method handling the (theoretical) encoding of sparse approximations.



"""
import numpy as np
from . import approx
from mdct import atom as mdct_atom
from math import ceil, log, sqrt
from PyMP.parallelProjections import project_atom
from PyMP import mp

def simple_mdct_encoding(app,
                         target_bitrate,
                         Q=7,
                         encode_weights=True,
                         encode_indexes=True,
                         subsampling=1,
                         shift_penalty=False,
                         output_file_path=None,
                         output_all_indexes=False):
    """ Simple encoder of a sparse approximation

    Arguments:

        - `app`: a :class:`.Approx` object containing atoms from the decomposition

        - `target_bitrate`: a float indicating the target bitrate. Atoms are considered in decreasing amplitude order.
        The encoding will stop either when the target Bitrate it reached or when all atoms of `approx` have been considered

        - `Q`: The number of midtread quantizer steps. Default is 7, increase this number for higher bitrates

        - `shift_penalty`: a boolean indicating whether LOMP algorithm has been used, and addition time-shift parameters must be encoded

    Returns:

        - `snr`: The achieved Signal to Noise Ratio

        - `bitrate`: The achieved bitrate, not necessarily equal to the given target

        - `quantized_approx`: a :class:`.Approx` object containing the quantized atoms

    """
    # determine the fixed cost (in bits) of an atom's index
    index_cost = log(app.length * len(app.dico.sizes) / subsampling, 2)

    # determine the fixed cost (in bits) of an atom's weight
    weights_cost = Q

    # instantiate the quantized approximation
    quantized_approx = approx.Approx(app.dico,
                                    [],
                                    app.original_signal,
                                    app.length,
                                    app.fs)

    total = 0
    shift_cost = 0

    # First loop to evaluate the max of atom's weight and
    max_value = 0
    for atom in app.atoms:
        if abs(atom.mdct_value) > abs(max_value):
                max_value = atom.mdct_value

    # deduce the Quantizer step width
    quantizer_width = max_value / float(2 ** (Q - 1))

    if quantizer_width == 0:
        raise ValueError('zero found for quantizer step width...')

    if app.atom_number <= 0:
        raise ValueError('approx object has an empty list of atoms')

    # second loop: atom after atom
    for atom in app.atoms:

        value = atom.mdct_value

        # Quantize the atom's weight
        quantized_value = ceil(
            (value - quantizer_width / 2) / quantizer_width) * quantizer_width

        # If quantized value is 0, no need to include it
        if quantized_value == 0:
            # If one is 100% sure atom's weights are in decreasing order, we could stop right here
            # But in the general case, we're not...
            continue

        else:
            total += 1

        # instantiate the quantized atom
        quantized_atom = mdct_atom.Atom(atom.length,
                                       atom.amplitude,
                                       atom.time_position,
                                       atom.freq_bin,
                                       atom.fs,
                                       mdctCoeff=quantized_value)

        # recompute true waveform
        quantized_atom.frame = atom.frame
        quantized_atom.waveform = atom.waveform * (sqrt((
            quantized_atom.mdct_value ** 2) / (atom.mdct_value ** 2)))

        # add atom to quantized Approx
        quantized_approx.add(quantized_atom)

        # All time-shift optimal parameters live in an interval of length L/2 where
        # L is the atom scale
        if shift_penalty:
                shift_cost += log(atom.length / 2, 2)

        # estimate current bitrate: Each atom coding cost is fixed!!
        n_bits_sofar = (total * (index_cost + weights_cost)) + shift_cost
        br = float(n_bits_sofar) / (float(app.length) / float(app.fs))

        if br >= target_bitrate:
            break

    # Evaluate True Bitrate
    bitrate = float(n_bits_sofar) / (float(app.length) / float(app.fs))

    approx_energy = sum(quantized_approx.recomposed_signal.data ** 2)
    res_energy = sum(
        (app.original_signal.data - quantized_approx.recomposed_signal.data) ** 2)

    # Evaluate distorsion
    snr = 10.0 * log(approx_energy / res_energy, 10)

    return snr, bitrate, quantized_approx


def joint_coding_distortion(target_sig , ref_app, max_rate,
                          search_width , threshold=0 , doSubtract=True, 
                          debug=0 , discard = False,
                          precut=-1, initfftw=True):
    """ compute the joint coding distortion e.g. given a fixed maximum rate, 
    what is the distorsion achieved if coding the target_sig knowing the ref_app
    
    This is limited to a time adaptation of the atoms
    """    
    tolerances = [2]*len(ref_app.dico.sizes)
    # initialize the fftw    
    if initfftw:
        mp._initialize_fftw(ref_app.dico, max_thread_num=1)
    # initialize factored approx
    factorizedApprox = approx.Approx(ref_app.dico, 
                                       [], 
                                       target_sig, 
                                       ref_app.length, 
                                       ref_app.fs,
                                       fast_create=True)        
    timeShifts = np.zeros(ref_app.atom_number)
    atom_idx = 0
    rate = 0    
    residual = target_sig.data.copy()
    if debug > 0:
        print "starting factorization of " , ref_app.atom_number , " atoms"        
    while (rate < max_rate) and (atom_idx < ref_app.atom_number):
        # Stop only when the target rate is achieved or all atoms have been used        
        atom = ref_app[atom_idx]
        # make a copy of the atom
        newAtom = atom.copy()
        HalfWidth = (tolerances[ref_app.dico.sizes.index(atom.length)] -1) * atom.length/2;            
        
        # Search for best time shift using cross correlation        
        input1 = residual[atom.time_position - HalfWidth : atom.time_position + atom.length + HalfWidth]
        input2 = np.zeros((2*HalfWidth + atom.length))
        input2[HalfWidth:-HalfWidth] = atom.waveform        
        if not (input1.shape == input2.shape):
            raise ValueError("This certainly happens because you haven't sufficiently padded your signal") 
        scoreVec = np.array([0.0]);
        newts = project_atom(input1,input2 , scoreVec , atom.length)        
        score = scoreVec
    
        if debug>0:
            print "Score of " , score , " found"
        # handle MP atoms #
        if newAtom.time_shift is not None:
            newAtom.time_shift += newts
            newAtom.time_position += newts
        else:
            newAtom.time_shift = newts
            newAtom.time_position += newts
            atom.proj_score = atom.mdct_value
                        
        if debug>0:
            print "Factorizing with new offset: " , newts                
        
        if score <0:                    
            newAtom.waveform = -newAtom.waveform

        factorizedApprox.add(newAtom)
        if debug > 0:
            print "SRR Achieved of : " , factorizedApprox.compute_srr()            
        timeShifts[atom_idx] = newAtom.time_shift
            
        if doSubtract:                
            residual[newAtom.time_position : newAtom.time_position + newAtom.length ] -= newAtom.waveform
            
        rate += np.log2(abs(newts))+1
        if debug:
            print "Atom %d - rate of %1.3f"%(atom_idx, rate)
        atom_idx +=1
        
        # Use to prune the calculus
        if atom_idx>precut and precut>0:
            curdisto = factorizedApprox.compute_srr() 
            if curdisto<0:
                # pruning
                return curdisto
            else:
                # stop wasting time afterwards
                 precut = -1
                
    # cleaning
    if initfftw:
        mp._clean_fftw()

    # calculate achieved SNR :    
    return factorizedApprox.compute_srr()
