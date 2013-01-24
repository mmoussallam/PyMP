#                                                                            */
#                               win_server.py                                */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
#
'''
 This class avoid the recomputation of the atoms by ensuring waveforms are
 just computed once then re-used

 TO BE REFACTORED
                                                                       */
'''

import numpy as np
import cmath
import math
# import fftw3
#from numpy.fft import ifft
from scipy.signal import hann

# IMPORTANT: this class requires that the Fast Projection C library is
# compiled and accessible
try:
    import parallelProjections
except ImportError:
    print '''Failed to load C fast projection library, please make sure it is
            compiled and/or accessible
            Run test_Cmodule.py for more information'''
    raise ImportError('No module named parallelProjections')


class PyServer():
    """ Musical Signal and audio in general happens to be well described by a
    collection of atoms of variable sizes, frequency and time localizations.
    However, due their redundancy results in the fact that often,
    the same waveforms are used at different time locations. This being said,
    one might hope substantial improvement by avoiding to recompute these
    waveform each tile ther're needed! """

    type = 'MDCT'  # TODO so far we only create MDCT atoms with sinus windows
    Waveforms = {}
        # Waveforms are stored in a dictionary, the key is Length+Frequency

    PreTwidCoeffs = {}
        # dictionary of pre-twidlle coefficient: key is the scale
    PostTwidCoeff = {}
        # dictionary of post-twidlle coefficient: key is the scale
    Windows = {}  # Sine Windows : key is the scale

    __single = None

    def __init__(self, useC=True):
        """ The Server is global"""
        if PyServer.__single:
            raise PyServer.__single
        PyServer.__single = self
        self.use_c_optim = useC

    def get_waveform(self, scale, binIndex):
        ''' Check whether the waveform is already in the dictionary
            If waveform has already been created, no need to recompute it'''
        key = scale + binIndex
        if not key in self.Waveforms:
            self.Waveforms[key] = self.create_waveform(scale, binIndex)

        return self.Waveforms[key]

    def create_waveform(self, scale, binIndex):
        ''' By default, will use the C library to create the waveform'''
        if self.use_c_optim:
            return parallelProjections.get_atom(int(scale), int(binIndex))


def get_server(x=PyServer):
    """ You should only access servers via this class method:
        if no instance is available, it will create one, otherwise
        it will return the singleton """
    try:
        single = x()
    except PyServer, s:
        single = s
    return single


class PyGaborServer():
    """ Musical Signal and audio in general happens to be well described by a collection of atoms
        of variable sizes, frequency and time localizations. However, due their redundancy results
        in the fact that often, the same waveforms are used at different time locations. This being said,
        one might hope substantial improvement by avoiding to recompute these waveform each tile ther're
        needed! """

    type = 'Gabor'  # TODO so far we only create MDCT atoms with sinus windows
    Waveforms = {}
        # Waveforms are stored in a dictionary, the key is Length+Frequency

    Windows = {}  # Sine Windows : key is the scale

    # constructor is the same
    def __init__(self, useC=True):
        """ The Server is global"""
        self.use_c_optim = useC

    # no twiddling coefficient
    def initialize(self, L):
        """ instantiate the coefficients for the given scale """
        self.Windows[L] = math.sqrt(8.0 / (3.0 * L)) * hann(L)

    def get_waveform(self, scale, binIndex, phase):

        return parallelProjections.get_real_gabor_atom(int(scale), binIndex, phase)

#        # Check whether the waveform is already in the dictionary
#        key = scale + binIndex + float(phase)/(2*math.pi)
#        if not key in self.Waveforms:
#            self.Waveforms[key] = self.create_waveform(scale , binIndex , phase)
#
#        return self.Waveforms[key]

    def create_waveform(self, scale, binIndex, phase):
        if not scale in self.Windows:
            self.initialize(scale)
        wf = math.sqrt(2.0) * self.Windows[scale] * np.array([np.cos(
            binIndex * l + phase) for l in range(scale)])
        # scrap zeroes on the borders
        return wf
