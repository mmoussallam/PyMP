#
#                                                                            */
#                               signals.py                                   */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
#
'''
Module signals
==============

The main class is :class:`Signal`, it can be instantiated from a numpy array
using the main constructor (multichannel is allowed).

It can also be created from a file on the disk using the path as argument
in the constructor

Longer Signals are handled with :class:`LongSignal` objects.

'''


import math
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
from tools import SoundFile
import base
import log

global _Logger
_Logger = log.Log('Signal', imode=False)
# from operator import add, mul


class Signal(object):
    """This file defines the main class handling audio signal in the Pymp Framework.

    A Signal is fairly a numpy array (called data) and a collection of attributes.
    Longer signals should not be loaded in memory. the `LongSignal` class allows
    to define this kind of signal, and slice it in frames overlapping or not.

    Attributes:

        `channel_num`:The number of channel

        `length`:                The length in samples (integer) of the signal (total dimension of the numpy array is channel_num x length)

        `samplingFrequency`:     The sampling frequency

        `location`:              (optional) Where the original file is located on the disk

        `sample_width` :          Various bit format exist for wav file, this allows to handle it

        `is_normalized` :         A boolean telling is the numpy array has been normalized (which here means its values are between -1 and 1)

        `energy`:                The energy (:math:`\sum_i x[i]^2`) of the array

    Standard methods to manipulate the signal are:

        :func:`normalize` : makes sure all values of the array are between -1 and 1

        :func:`plot`: Plot using matlplotlib

        :func:`crop` : crop

        :func:`pad`:  with zeroes

        :func:`copy`:

        :func:`downsample`:

    For use in the PyMP framework, other methods need to be defined:

        :func:`add`:             adds an atom waveform to the current signal and updates energy

        :func:`subtract`:        the opposite operation

    Output Methods: :func:`write`

    """

    # att
    data = []
    channel_num = 0
    length = 0
    fs = 0
    location = ""
    sample_width = 2
    is_normalized = False
    energy = 0

    # Constructor
    def __init__(self, data=[], Fs=0, normalize=False, mono=False, debug_level=None):
        ''' Simple constructor from a numpy array (data) or a string '''

        if isinstance(data, str):
            self.location = data
            Sf = SoundFile.SoundFile(data)
            self.data = Sf.GetAsMatrix().reshape(Sf.nframes, Sf.nbChannel)
            self.sample_width = Sf.sample_width
            if mono:
                self.data = self.data[:, 0]

            Fs = Sf.sampleRate
        else:
            self.data = np.array(data)

        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.length = self.data.shape[0]
        self.fs = Fs

        if len(self.data.shape) > 1:
            self.channel_num = self.data.shape[1]
        else:
            self.channel_num = 1

        if normalize & (self.length > 0):
            _Logger.info('Normalizing Signal')
            normaFact = self.data.max()
            self.data = self.data.astype(float) / float(normaFact)
            self.is_normalized = True

        if len(data) > 0:
            self.energy = np.sum(self.data ** 2)

        _Logger.info('Signal created ' + str(self))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Signal(self.data[item], self.fs, normalize=False)
        else:
            raise TypeError("Argument not recognized as a slice")

    def __repr__(self):
        return '''Signal object located in %s
        length: %d
        energy of %2.3f
        sampling frequency %d
        number of channels %d''' % (self.location, self.length, self.energy, self.fs, self.channel_num)

    def normalize(self):
        ''' makes sure all values of the array are between -1 and 1 '''
        _Logger.info('Normalizing Signal')
        normaFact = abs(self.data).max()
        self.data = self.data.astype(float) / float(normaFact)
        self.energy = sum(self.data ** 2)
        self.is_normalized = True

    def plot(self, pltStr='b-', legend=None):
        ''' DEPRECATED plot the array using matplotlib '''
        plt.plot(self.data,pltStr)
        if legend != None:
            plt.legend((legend))
        plt.show()

    # cropping routine
    def crop(self, startIndex=0, stopIndex=None):
        ''' cropping routine, usage is quite obvious '''
        if (startIndex > self.length) | (startIndex < 0):
#            raise ValueError("Starting Index beyond signal dimensions")
            _Logger.error('Starting Index ' + str(
                startIndex) + ' is beyond signal length of ' + str(startIndex))
        if (stopIndex > self.length) | (stopIndex < 0):
            _Logger.error('Stopping Index ' + str(
                stopIndex) + ' is beyond signal length of ' + str(self.length))
#            raise ValueError("Stopping Index beyond signal dimensions")
        if (len(self.data.shape) > 1):
            if (self.data.shape[0] > self.data.shape[1]):
                self.data = self.data[startIndex:stopIndex, :]
            else:
                self.data = self.data[:, startIndex:stopIndex]
        else:
            self.data = self.data[startIndex:stopIndex]
        self.length = stopIndex - startIndex

    def window(self, K):
        """ apply a sine window on norders """
        self.data[0:K] *= np.sin((np.arange(K).astype(float))*np.pi/(2*K));
        self.data[-K:] *= np.sin((np.arange(K).astype(float))*np.pi/(2*K) + np.pi/2);

    def copy(self):
        copiedSignal = Signal(self.data.copy(), self.fs)
        copiedSignal.location = self.location
        copiedSignal.channel_num = self.channel_num
        copiedSignal.sample_width = self.sample_width
        copiedSignal.is_normalized = self.is_normalized
        return copiedSignal

    def subtract(self, atom, debug=0, preventEnergyIncrease=True):
        ''' Subtracts the atom waveform from the signal, at the position specified by the atom.timeLocalization property
            if preventEnergyIncrease is True, an error will be raised if subtracting the atom increases the signal's energy '''
        if not isinstance(atom, base.BaseAtom):
            raise TypeError("argument provided is not an atom")
        if atom.waveform is None:
#            print "Resynth"
            atom.synthesize()
        # Keep track of the energy to make
        oldEnergy = (self.data[atom.time_position: atom.
                               time_position + atom.length] ** 2).sum()

        # In high debugging levels, show the process using matplotlib
        if debug > 2:
            plt.figure()
            plt.plot(self.data[atom.time_position: atom.
                               time_position + atom.length])
            plt.plot(atom.waveform)
            plt.plot(self.data[atom.time_position: atom.
                               time_position + atom.length] - atom.waveform, ':')
            plt.legend(('Signal', 'Atom substracted', 'residual'))
            plt.show()

        # actual subtraction
        self.data[atom.time_position: atom.time_position + atom.
                  length] -= atom.waveform

        newEnergy = (self.data[atom.time_position: atom.
                               time_position + atom.length] ** 2).sum()

        # Test
        if (newEnergy > oldEnergy) and preventEnergyIncrease:
            # do not substract the atom
            self.data[atom.time_position: atom.
                      time_position + atom.length] += atom.waveform
            # get out of here
            _Logger.error('Warning : Substracted Atom created energy : at pos: '
                          + str(atom.time_position) + ' k: ' + ' Amp: ' + str(atom.get_value()) +
                          str(atom.freq_bin) + ' timeShift :' +
                          str(atom.time_shift) + ' scale : ' +
                          str(atom.length) + ' from ' + str(oldEnergy) + ' to ' + str(newEnergy))
            raise ValueError('see Logger')

        self.energy = self.energy - oldEnergy + newEnergy

    def downsample(self, newFs):
        """ downsampling the signal by taking only a portion of the data """

        if newFs >= self.fs:
            raise ValueError('new sampling frequency is bigger than actual, try upsampling instead')

        # ratio of selected points
        subRatio = float(self.fs) / float(newFs)
        indexes = np.floor(np.arange(0, self.length, subRatio))
#        print indexes
        self.data = self.data[indexes.tolist()]
        self.length = len(self.data)
        self.fs = newFs

    def add(self, atom):
        ''' adds the contribution of the atom at the position specified by the atom.timeLocalization property '''
        if not isinstance(atom, base.BaseAtom):
            raise TypeError("argument provided is not an atom")

        if atom.waveform is None:
            _Logger.info("Resynthesizing  waveform")
            atom.synthesize()

        localEnergy = np.sum(self.data[atom.time_position: atom.
                                       time_position + atom.length] ** 2)

        # do sum calculation
        try:
            self.data[atom.time_position: atom.
                      time_position + atom.length] += atom.waveform
        except:
            _Logger.error('Mispositionned atom: ' + str(atom.
                                                        time_position) + ' and length ' + str(atom.length))
#        # update energy value
        self.energy += np.sum(self.data[atom.time_position: atom.
                                        time_position + atom.length] ** 2) - localEnergy

    # Pad edges with zeroes
    def pad(self, zero_pad):
        ''' Pad edges with zeroes '''
        try:
            self.data = np.concatenate((np.concatenate(
                (np.zeros(zero_pad), self.data)), np.zeros(zero_pad)))
            self.length = len(self.data)
        except ValueError:
            print(self.data.shape)
            _Logger.error(
                "Wrong dimension number %s" % str(self.data.shape))

    def depad(self, zero_pad):
        ''' Remove zeroes from the edges WARNING: no test on the deleted data: make sure these are zeroes! '''
        try:
            self.data = self.data[zero_pad:-zero_pad]
            self.length = len(self.data)
        except ValueError:
            print(self.data.shape)
            _Logger.error(
                "Wrong dimension number %s" % str(self.data.shape))

#    def doWindow(self, K):
#        ''' DEPRECATED '''
#        self.data[0:K] *= np.sin((np.arange(K).astype(float)) * np.pi / (2 * K))
#        self.data[-K:] *= np.sin((np.arange(K).astype(float)) * np.pi / (
#            2 * K) + np.pi / 2)

    def write(self, fileOutputPath, pad=0):
        ''' Write the current signal at the specified location in wav format

            This is done using the wave python library'''
        if pad > 0:
            self.depad(pad)

        if self.energy == 0:
            _Logger.warning("Zero-energy signal")
            self.data = np.zeros(self.length,)
#            print "Warning!!!! Zero-energy signal:"

        wav_file = wave.open(fileOutputPath, 'wb')
        wav_file.setparams((self.channel_num,
                            self.sample_width,
                            self.fs,
                            self.length,
                            'NONE', 'not compressed'))

        # prepare binary output
        values = []
        if not self.is_normalized:
            self.normalize()

        for i in range(len(self.data)):
            if np.isnan(self.data[i]):
                _Logger.warning("NaN data found: replaced by 0")
                self.data[i] = 0
            if self.data[i] < -1:
                _Logger.warning(str(i) + 'th sample was below -1: cropping')
                self.data[i] = -1
            if self.data[i] > 1:
                _Logger.warning(str(i) + 'th sample was over 1: cropping')
                self.data[i] = 1
            value = int(16384 * self.data[i])
            packed_value = struct.pack('h', value)
            values.append(packed_value)

        value_str = ''.join(values)
        wav_file.writeframes(value_str)

#        else:
# _Logger.warning("Warning BUGFIX not done yet, chances are writing failed");
#            wav_file.writeframes((self.data).astype(int).tostring())
        wav_file.close()

    def wigner_plot(self, window=True):
        """ Calculate the wigner ville distribution and plots it
        WARNING: not sufficiently tested!"""
        from numpy.fft import ifft, fft, ifftshift, fftshift

        N = self.length
        Ex = self.data

        # ensure the signal is mono (take left canal if stereo)
        if len(Ex.shape) > 1 and not (Ex.shape[1] == 1):
            Ex = Ex[:, 0]

        # reshape anyway as a column vector
        Ex = Ex.reshape((N, 1))

        if window:
            from scipy.signal import hann
            Ex *= hann(N).reshape((N, 1))

        x = ifftshift((np.arange(0, N) - N / 2) * 2 * np.pi / (N - 1))
        #%   Generate linear vector
        X = np.arange(0, N) - N / 2

        x = x.reshape((N, 1))
        X = X.reshape((1, N))

        A = np.dot(1j * x, X / 2)
        EX1 = ifft(np.dot(fft(Ex), np.ones((1, N))) * np.exp(A))
        #%   +ve shift
        EX2 = ifft(
            np.dot(fft(Ex), np.ones((1, N))) * np.exp(np.dot(-1j * x, X / 2)))
        #%   -ve shift

        tmp0 = EX1 * np.conj(EX2)

        print

        print tmp0.shape
        tmp1 = fftshift(tmp0, (1, ))
        tmp2 = fft(tmp1, axis=1)

        W = np.real(fftshift(tmp2, (1,)))
        #%   Wigner function

        return W


# def InitFromFile(filepath, forceMono=False, doNormalize=False, debugLevel=None):
#    ''' Static method to create a Signal from a wav file on the disk
#        This is based on the wave Python library through the use of the Tools.SoundFile class
#        '''
#    if debugLevel is not None:
#        _Logger.set_level(debugLevel)
#    Sf = SoundFile.SoundFile(filepath)
#    #print Sf.GetAsMatrix().shape
#    reshapedData = Sf.GetAsMatrix().reshape(Sf.nframes, Sf.nbChannel)
#    if forceMono:
#        reshapedData = reshapedData[:, 0]
#
#    sig = Signal(reshapedData, Sf.sampleRate, doNormalize)
#    sig.sample_width = Sf.sample_width
#    sig.location = filepath
#
#    _Logger.info("Created Signal of length " + str(
#        sig.length) + " samples of " + str(sig.channel_num) + "channels")
#    # print "Created Signal of length " + str(Signal.length) +" samples " #of "
#    # + str(Signal.channel_num) + "channels"
#    return sig


class LongSignal(Signal):
    """ A class handling long audio signals

        Subclass of :class:`.Signal` where the data is not loaded at once for memory consumptions purposes
        Instead, the data is sliced in frames that can be loaded later individually. very useful to
        process large files such as audio archives

        Attributes:

            `filepath`:    The path to the audio file

            `frameSize`:   In samples, default is 16384*3

            `frameDuration`:  Alternative to frameSize, specify directly a frame duration in seconds (Defult is None)

            `forceMono`:      Only load the left (first) channel (default is False)

            `Noverlap` :      overlap (as a ratio r such that :math:`0\leq r < 1`)

    Standard methods to manipulate the signal are:

        :func:`get_sub_signal` : Loads a subSignal

    """

    # more attr
#    filetype = ''       # file extension : .wav or .mp3
#    nframes = 0;        # number of
#    sample_width = 0;    # bit width of each frame
# segmentSize = 0;      # in 16-bits samples : different from nframes which is
# internal wav representation
#    segmentNumber =0;     # numberof audio segments to be consideres
#    overlap = 0;         # overlapping rate between two consecutive segments

    # constructor
    def __init__(self, filepath, frameSize=16384 * 3, frameDuration=None, forceMono=False, Noverlap=0):
        self.location = filepath
        self.segmentSize = frameSize

        # overlaps methods from SoundFile object
#        if (filepath[-4:] =='.wav'):
        file = wave.open(filepath, 'r')
#        elif (filepath[-4:] =='.raw'):
#            file = open()
        self.filetype = filepath[len(filepath) - 3:]
        self.channel_num = file.getnchannels()
        self.fs = file.getframerate()
        self.nframes = file.getnframes()
        self.sample_width = file.getsampwidth()

        self.overlap = Noverlap

        if frameDuration is not None:
            # optionally set the length in seconds, adapt to the sigbal
            # sampling frequency
            self.segmentSize = math.floor(
                frameDuration * self.fs)

        self.segmentNumber = math.floor(
            self.nframes / (self.segmentSize * (1 - self.overlap)))

        if self.overlap >= 1:
            raise ValueError('Overlap must be in [0..1[ ')

        self.segmentNumber -= self.overlap / (1 - self.overlap)

        _Logger.info('Loaded ' + filepath + ' , ' + str(
            self.nframes) + ' frames of ' + str(self.sample_width) + ' bytes')
        _Logger.info('Type is ' + self.filetype + ' , ' + str(self.
                                                              channel_num) + ' channels at ' + str(self.fs))
        _Logger.info('Separated in ' + str(self.segmentNumber) + ' segments of size ' + str(self.segmentSize) + ' samples overlap of ' + str(self.overlap * self.segmentSize))
        self.length = self.segmentNumber * frameSize
        file.close()

#    def readFrames(self , frameIndexes):
#        #self.data = array.array('h') #creates an array of ints
#        file = wave.open(self.location, 'r')
#        str_bytestream = file.readframes(self.nframes)
#        self.data = fromstring(str_bytestream,'h')
#        file.close()

    def get_sub_signal(self, startSegment, segmentNumber, mono=False, normalize=False, channel=0, padSignal=0):
        """ Routine to actually read from the buffer and return a smaller signal instance

        :Returns:

            a :class:`Signal` object

        :Example:

            longSig = LongSignal(**myLongSigFilePath**, frameDuration=5) # Initialize long signal
            subSig = longSig.get_sub_signal(0,10) # Loads the first 50 seconds of data

        """

        # convert frame into bytes positions
        bFrame = startSegment * (self.segmentSize * (1 - self.overlap))
        nFrames = int(segmentNumber * self.segmentSize)
        file = wave.open(self.location, 'r')
        file.setpos(bFrame)
        str_bytestream = file.readframes(nFrames)
        data = np.fromstring(str_bytestream, 'h')
        file.close()

        if self.channel_num > 1:
            reshapedData = data.reshape(nFrames, self.channel_num)
        else:
            reshapedData = data.reshape(nFrames, )
        if mono:
            if len(reshapedData.shape) > 1:
                reshapedData = reshapedData[:, channel]

            reshapedData = reshapedData.reshape(nFrames, )

        SubSignal = Signal(reshapedData, self.fs, normalize)
        SubSignal.location = self.location

        if padSignal != 0:
            SubSignal.pad(padSignal)

# print "Created Signal of length " + str(SubSignal.length) +" samples " #of "
# + str(Signal.channel_num) + "channels"
        return SubSignal
