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

from . import approx
from mdct import atom as mdct_atom
from math import ceil, log, sqrt


def simple_mdct_encoding(app,
                         targetBitrate,
                         Q=7,
                         encodeCoeffs=True,
                         encodeIndexes=True,
                         subsampling=1,
                         TsPenalty=False,
                         saveOutputFilePath=None,
                         outputAllIndexes=False):
    """ Simple encoder of a sparse approximation

    Arguments:

        - `app`: a :class:`.Approx` object containing atoms from the decomposition

        - `targetBitrate`: a float indicating the target bitrate. Atoms are considered in decreasing amplitude order.
        The encoding will stop either when the target Bitrate it reached or when all atoms of `approx` have been considered

        - `Q`: The number of midtread quantizer steps. Default is 7, increase this number for higher bitrates

        - `TsPenalty`: a boolean indicating whether LOMP algorithm has been used, and addition time-shift parameters must be encoded

    Returns:

        - `SNR`: The achieved Signal to Noise Ratio

        - `Bitrate`: The achieved bitrate, not necessarily equal to the given target

        - `quantizedApprox`: a :class:`.Approx` object containing the quantized atoms

    """
    # determine the fixed cost (in bits) of an atom's index
    indexcost = log(app.length * len(app.dico.sizes) / subsampling, 2)

    # determine the fixed cost (in bits) of an atom's weight
    Coeffcost = Q

    # instantiate the quantized approximation
    quantizedApprox = approx.Approx(app.dico,
                                    [],
                                    app.original_signal,
                                    app.length,
                                    app.fs)

    total = 0
    TsCosts = 0

    # First loop to evaluate the max of atom's weight and
    maxValue = 0
    for atom in app.atoms:
        if abs(atom.mdct_value) > abs(maxValue):
                maxValue = atom.mdct_value

    # deduce the Quantizer step width
    quantizerWidth = maxValue / float(2 ** (Q - 1))

    if quantizerWidth == 0:
        raise ValueError('zero found for quantizer step width...')

    if app.atom_number <= 0:
        raise ValueError('approx object has an empty list of atoms')

    # second loop: atom after atom
    for atom in app.atoms:

        value = atom.mdct_value

        # Quantize the atom's weight
        quantizedValue = ceil(
            (value - quantizerWidth / 2) / quantizerWidth) * quantizerWidth

        # If quantized value is 0, no need to include it
        if quantizedValue == 0:
            # If one is 100% sure atom's weights are in decreasing order, we could stop right here
            # But in the general case, we're not...
            continue

        else:
            total += 1

        # instantiate the quantized atom
        quantizedAtom = mdct_atom.Atom(atom.length,
                                       atom.amplitude,
                                       atom.time_position,
                                       atom.freq_bin,
                                       atom.fs,
                                       mdctCoeff=quantizedValue)

        # recompute true waveform
        quantizedAtom.frame = atom.frame
        quantizedAtom.waveform = atom.waveform * (sqrt((
            quantizedAtom.mdct_value ** 2) / (atom.mdct_value ** 2)))

        # add atom to quantized Approx
        quantizedApprox.add(quantizedAtom)

        # All time-shift optimal parameters live in an interval of length L/2 where
        # L is the atom scale
        if TsPenalty:
                TsCosts += log(atom.length / 2, 2)

        # estimate current bitrate: Each atom coding cost is fixed!!
        nbBitsSoFar = (total * (indexcost + Coeffcost)) + TsCosts
        br = float(nbBitsSoFar) / (float(app.length) / float(app.fs))

        if br >= targetBitrate:
            break

    # Evaluate True Bitrate
    bitrate = float(nbBitsSoFar) / (float(app.length) / float(app.fs))

    approxEnergy = sum(quantizedApprox.recomposed_signal.data ** 2)
    resEnergy = sum(
        (app.original_signal.data - quantizedApprox.recomposed_signal.data) ** 2)

    # Evaluate distorsion
    SNR = 10.0 * log(approxEnergy / resEnergy, 10)

    return SNR, bitrate, quantizedApprox
