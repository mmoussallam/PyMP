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
