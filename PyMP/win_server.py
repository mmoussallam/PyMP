#                                                                            */
#                               PyWinServer.py                               */
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
#     
'''
 This class avoid the recomputation of the atoms by ensuring waveforms are 
=======
#

 This class avoid the recomputation of the atoms by ensuring waveforms are
>>>>>>> 9386cf831009ecdea1833fd7850264d94aee2b98
 just computed once tehn re-used
                                                                       */
'''

from numpy import zeros , array , sin , cos , pi
import cmath , math
#import fftw3
from numpy.fft import ifft
from scipy.signal import hann

# IMPORTANT: this class requires that the Fast Projection C library is compiled and accessible
try:
    import parallelProjections
except ImportError:
    print '''Failed to load C fast projection library, please make sure it is compiled and/or accessible
            Run ParallelCBindingTests.py for more information'''
    raise ImportError('No module named parallelProjections')

class PyServer():
    """ Musical Signal and audio in general happens to be well described by a collection of atoms
        of variable sizes, frequency and time localizations. However, due their redundancy results
        in the fact that often, the same waveforms are used at different time locations. This being said,
        one might hope substantial improvement by avoiding to recompute these waveform each tile ther're
        needed! """

    type = 'MDCT'# TODO so far we only create MDCT atoms with sinus windows
    Waveforms = {} # Waveforms are stored in a dictionary, the key is Length+Frequency

    PreTwidCoeffs = {} # dictionary of pre-twidlle coefficient: key is the scale
    PostTwidCoeff = {} # dictionary of post-twidlle coefficient: key is the scale
    Windows = {}  # Sine Windows : key is the scale
#    fftplans = {} ; # dictionary of pre-defined fftwplans
#    fftinputs = {};
#    fftoutputs = {};

    def __init__(self , useC=True):
        """ The Server is global"""
        self.useC = useC;

    # twidlle coefficients
    def initialize(self , L):
        """ instantiate the coefficients for the given scale """
        self.Windows[L] = array([ sin(float(l + 0.5) *(pi/L)) for l in range(L)] )
        self.PreTwidCoeffs[L] = array([cmath.exp(1j*2*float(n)*pi*(float(float(L/4+0.5))/float(L))) for n in range(L)]);
        self.PostTwidCoeff[L] = array([cmath.exp(0.5*1j*2*pi*(float(n)+(L/4+0.5))/L) for n in range(L)]) ;

#        self.fftinputs[L] = zeros((L,1),complex);
#        self.fftoutputs[L] = zeros((L,1),complex);
#        self.fftplans[L] = fftw3.Plan(self.fftinputs[L],self.fftoutputs[L], direction='backward', flags=['estimate'])
#
    def getWaveForm(self , scale , binIndex):
        ''' Check whether the waveform is already in the dictionary
            If waveform has already been created, no need to recompute it'''
        key = scale + binIndex
        if not key in self.Waveforms:
            self.Waveforms[key] = self.createWaveform(scale , binIndex)

        return self.Waveforms[key]

    def createWaveform(self, scale , binIndex):
        ''' By default, will use the C library to create the waveform'''
        if self.useC:
            return parallelProjections.get_atom(int(scale) , int(binIndex))

        else:
            L = scale
            # Check whether coefficients are initialized
            if not scale in self.Windows:
                self.initialize(scale)
            K = L/2
            temp = zeros(2*L)
            temp[K + binIndex] = 1
            waveform = zeros(2*L)

            y = zeros(L, complex)
            x = zeros(L, complex)
            # we're note in Matlab anymore, loops are more straight than indexing
            for i in range(1,3):
                y[0:K] = temp[i*K : (i+1)*K]
    #
                # do the pre-twiddle
                y = y *  self.PreTwidCoeffs[L];
    #            self.fftinputs[L] = y;

                # compute ifft
    #            self.fftplans[L].execute()
    #            x = self.fftoutputs[L];
                x = ifft(y)
    #
    #            # do the post-twiddle
                x = x * self.PostTwidCoeff[L]
    #
                x = 2*math.sqrt(1/float(L))*L*x.real*self.Windows[L];

    #            self.fftinputs[L][0:K] += temp[i*K : (i+1)*K]*self.PreTwidCoeffs[L]

                #compute fft


    #            # post-twiddle and store for max search
    #            self.projectionMatrix[i*K : (i+1)*K] = normaCoeffs*(self.inputa[0:K]* self.post_twidVec).real

                # overlapp - add
    #            waveform[i*K : i*K +L] = waveform[i*K : i*K +L] + \
    #                     2*math.sqrt(1/float(L))*L*(self.fftoutputs[L]*self.PostTwidCoeff[L]).real*self.Windows[L] ;
                waveform[i*K : i*K +L] = waveform[i*K : i*K +L] +  x ;

            # scrap zeroes on the borders
            return waveform[L/2:-L/2]

class PyGaborServer():
    """ Musical Signal and audio in general happens to be well described by a collection of atoms
        of variable sizes, frequency and time localizations. However, due their redundancy results
        in the fact that often, the same waveforms are used at different time locations. This being said,
        one might hope substantial improvement by avoiding to recompute these waveform each tile ther're
        needed! """

    type = 'Gabor'# TODO so far we only create MDCT atoms with sinus windows
    Waveforms = {} # Waveforms are stored in a dictionary, the key is Length+Frequency


    Windows = {}  # Sine Windows : key is the scale

    # constructor is the same
    def __init__(self , useC=True):
        """ The Server is global"""
        self.useC = useC;

    # no twiddling coefficient
    def initialize(self , L):
        """ instantiate the coefficients for the given scale """
        self.Windows[L] = math.sqrt(8.0/(3.0*L)) * hann(L);

    def getWaveForm(self , scale , binIndex , phase):

        return parallelProjections.get_real_gabor_atom(int(scale), binIndex , phase)

#        # Check whether the waveform is already in the dictionary
#        key = scale + binIndex + float(phase)/(2*math.pi)
#        if not key in self.Waveforms:
#            self.Waveforms[key] = self.createWaveform(scale , binIndex , phase)
#
#        return self.Waveforms[key]

    def createWaveform(self, scale , binIndex , phase):
        if not scale in self.Windows:
            self.initialize(scale)
        wf = math.sqrt(2.0)* self.Windows[scale]  * array([ cos(binIndex*l + phase) for l in range(scale)])
        # scrap zeroes on the borders
        return wf


