#
#                                                                            */
#                               signals.py                                   */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
#

import math
import wave
import struct
import numpy as np

from tools import SoundFile
import base
import log

global _Logger
_Logger = log.Log('Signal', imode=False)
# from operator import add, mul


class Signal(base.BaseSignal):
    """This file defines the main class handling audio signal in the Pymp Framework.

    A Signal is fairly a numpy array (called data) and a collection of attributes.
    Longer signals should not be loaded in memory. the `LongSignal` class allows
    to define this kind of signal, and slice it in frames overlapping or not.

    Attributes
    ----------
    *data* : array
        a numpy array containing the signal data
    *channel_num* : int
        The number of channel
    *length* : int
        The length in samples (integer) of the signal (total dimension of the numpy array is channel_num x length)
    *fs* : int
        The sampling frequency
    *location* :  str , optional
        Where the original file is located on the disk
    *sample_width* : int
        Various bit format exist for wav file, this allows to handle it
    *is_normalized* : bool
        A boolean telling is the numpy array has been normalized (which here means its values are between -1 and 1)
    *energy* : float
        The energy (:math:`\sum_i x[i]^2`) of the array
    


    Notes
    -----
    For use in the PyMP framework, other methods need to be defined:
        :func:`add` :             adds an atom waveform to the current signal and updates energy
        :func:`subtract` :        the opposite operation
    

    """

    # att
    data = np.array([])
    channel_num = 0
    length = 0
    fs = 0
    location = ""
    sample_width = 2
    is_normalized = False
    energy = 0

    # Constructor
    def __init__(self, data=[], Fs=0, normalize=False, mono=False, debug_level=None):
        """ Simple constructor from a numpy array (data) or a string 
    parameters
    ----------
    data : array-like
        a numpy array containing the signal data
    Fs :  int
        the sampling frequency
    normalize : bool, optionnal
        whether to normalize the signal (e.g. max will be 1)
    mono :  bool, optionnal
        keep only the first channel

        """

        if isinstance(data, str):
            self.location = data
            Sf = SoundFile.SoundFile(data)
            self.data = Sf.GetAsMatrix().reshape(Sf.nframes, Sf.nbChannel)
            self.sample_width = Sf.sample_width

            Fs = Sf.sampleRate
        else:

            self.data = np.array(data)
            # remove any nans or infs data
            self.data[np.isnan(self.data)] = 0
            self.data[np.isinf(self.data)] = 0

            if len(self.data.shape) > 2:
                raise ValueError("Cannot process more than 2D arrays")

            if len(self.data.shape) > 1:
                if self.data.shape[0] < self.data.shape[1]:
                    # by convention we store channels as columns...
                    self.data = self.data.transpose()

        # keeping only the left channel
        if mono and len(self.data.shape) > 1:
            self.data = self.data[:, 0]

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
        ''' makes sure all values of the array are between -1 and 1
            CAUTION: this is an amplitude normalization not an energy one'''
        _Logger.info('Normalizing Signal')
        normaFact = abs(self.data).max()
        self.data = self.data.astype(float) / float(normaFact)
        self.energy = sum(self.data ** 2)
        self.is_normalized = True

    def plot(self, pltStr='b-', legend=None):
        ''' DEPRECATED plot the array using matplotlib '''
        import matplotlib.pyplot as plt
        plt.plot(self.data, pltStr)
        if legend != None:
            plt.legend((legend))
        plt.show()

    def play(self, prevent_too_long=True, int_type=np.int16):
        '''EXPERIMENTAL: routine to play the signal using wave and pyaudio'''
        import pyaudio
        p = pyaudio.PyAudio()

        # TODO: allow other types of integers
#        p.get_format_from_width(2)
        stream = p.open(format=8,
                        channels=self.channel_num,
                        rate=self.fs,
                        output=True)

        # data = wf.readframes(CHUNK)
        # Warning: check whether the flow has been normalized
        # in which case we need to remultiply it

        if self.is_normalized:
            data = (self.data * 16384).astype(int_type).tostring()
        else:
            data = self.data.astype(int_type).tostring()

        # TODO : check the size of the input
        if (self.length / self.fs > 10) and prevent_too_long:
            raise ValueError("File is longer than 10 secs, please call with prevent_too_long=False")
        stream.write(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

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
        if K > 0:
            self.data[0:K] *= np.sin((np.arange(K).astype(float)
                                      ) * np.pi / (2 * K))
            self.data[-K:] *= np.sin(
                (np.arange(K).astype(float)) * np.pi / (2 * K) + np.pi / 2)

    def copy(self):
        copiedSignal = Signal(self.data.copy(), self.fs)
        copiedSignal.location = self.location
        copiedSignal.channel_num = self.channel_num
        copiedSignal.sample_width = self.sample_width
        copiedSignal.is_normalized = self.is_normalized
        return copiedSignal

    def subtract(self, atom, debug=0, prevent_energy_increase=True):
        ''' Subtracts the atom waveform from the signal, at the position specified by the atom.timeLocalization property
            if prevent_energy_increase is True, an error will be raised if subtracting the atom increases the signal's energy '''
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
            import matplotlib.pyplot as plt
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
        if (newEnergy > oldEnergy) and prevent_energy_increase:
            # do not substract the atom
            self.data[atom.time_position: atom.
                      time_position + atom.length] += atom.waveform
            # get out of here
            err_str = """"Warning : Substracted Atom created energy:
                          from %1.6f to from %1.6f :
%s""" % (oldEnergy, newEnergy, atom)
            _Logger.error(err_str)
            raise ValueError('see Logger')

        self.energy = self.energy - oldEnergy + newEnergy

    def resample(self, newFs):
        """ resampling the signal """
        from scipy.signal import resample
        resamp_data = resample(self.data, self.get_duration()*newFs)
        self.data = resamp_data
        self.length = len(self.data)
        self.fs = newFs
    
    def downsample(self, newFs):
        """ downsampling the signal by taking only a portion of the data """

        if newFs >= self.fs:
            print 'WARNING new sampling frequency is bigger than actual, trying upsampling instead'
            return self.resample(newFs)

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
            _Logger.error('Mispositionned atom: from %d to %d - sig length of %d'%(atom.time_position,
                                                                atom.length,
                                                                self.length))
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

    def get_duration(self):
        ''' returns the duration in seconds '''
        return float(self.length) / float(self.fs)

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
#                 "Data Clipped during writing"
                self.data[i] = -1
            if self.data[i] > 1:
                _Logger.warning(str(i) + 'th sample was over 1: cropping')
                self.data[i] = 1
#                print "Data Clipped during writing"
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

    def spectrogram(self, wsize=512, tstep=256, order=2, log=False,
                    ax=None, cmap=None, cbar=True):
        """ Compute and plot the (absolute value) spectrogram of the signal 
        
        Based on a short-time fourier transform
        
        Parameters
        ----------
        wsize : int
            length of the STFT window in samples (must be a multiple of 4)
        tstep : int , opt
            step between successive windows in samples (must be a multiple of 2,
            a divider of wsize and smaller than wsize/2) (default: wsize/2)
        order : int, opt
            1 will have the Magnitude Spectrum, 2 the Power Spectrum
        log : bool, opt
            Set to True for the log-spectrogram
        """
        import matplotlib.pyplot as plt
        from scipy.fftpack import fft, ifft, fftfreq
        if ax is None:            
            ax = plt.gca()
                    
        
        ### Errors and warnings ###
        if wsize % 4:
            raise ValueError('The window length must be a multiple of 4.')
    
        if tstep is None:
            tstep = wsize / 2
    
        tstep = int(tstep)
    
        if (wsize % tstep) or (tstep % 2):
            raise ValueError('The step size is %d but it should be a multiple of 2 and a '
                             'divider of the window length %d'%(tstep,wsize))
    
        if tstep > wsize:
            raise ValueError('The step size is %d but it should be smaller than the '
                             'window length %d'%(tstep,wsize))
    
        n_step = int(math.ceil(self.length / float(tstep)))
        n_freq = wsize / 2 + 1
        
        _Logger.debug("Number of frequencies: %d" % n_freq)
        _Logger.debug("Number of time steps: %d" % n_step)
    
        X = np.zeros((self.channel_num, n_freq, n_step), dtype=np.complex)
    
        if self.channel_num == 0:
            return X
        x = self.data.T
        T = self.length
        # Defining sine window
        win = np.sin(np.arange(.5, wsize + .5) / wsize * np.pi)
        win2 = win ** 2
    
        swin = np.zeros((n_step - 1) * tstep + wsize)
        for t in range(n_step):
            swin[t * tstep:t * tstep + wsize] += win2
        swin = np.sqrt(wsize * swin)
    
        # Zero-padding and Pre-processing for edges
        xp = np.zeros((self.channel_num, wsize + (n_step - 1) * tstep),
                      dtype=x.dtype)
        xp[:, (wsize - tstep) / 2: (wsize - tstep) / 2 + T] = x
        x = xp
    
        for t in range(n_step):
            # Framing
            wwin = win / swin[t * tstep: t * tstep + wsize]
            frame = x[:, t * tstep: t * tstep + wsize] * wwin[None, :]
            # FFT
            fframe = fft(frame)
            X[:, :, t] = fframe[:, :n_freq]
        
        Spectro = np.abs(X)**order
        
        if log:
            Spectro = np.log10(Spectro)
        
        if Spectro.shape[0]>1:
            print "Taking mean of channels"
            Spectro = np.squeeze(np.mean(Spectro, axis=0))
        else:
            Spectro = np.squeeze(Spectro)
        
        (Fmax,Tmax) = Spectro.shape
        yticks =  np.arange(.0,Fmax, Fmax/10.0)
        yvalues = (yticks*0.5*float(self.fs)/float(Fmax)).astype(int)
        xticks =  np.arange(.0,Tmax, Tmax/10.0)
        xvalues = (xticks*float(tstep)/float(self.fs))
        if cmap is None:
            import matplotlib.cm as cm
            cmap = cm.coolwarm
        plt.imshow(Spectro,
                   origin='lower',
                   cmap=cmap)
        
        plt.xticks(xticks, ["%1.2f"%v for v in xvalues])
        plt.yticks(yticks, yvalues)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        if cbar:
            plt.colorbar()
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

        Attributes
        ----------
        `filepath`:    The path to the audio file
        `frame_size`:   In samples, defaut is 16384*3
        `frame_duration`:  Alternative to frameSize, specify directly a frame duration in seconds (Defult is None)
        `mono`:      Only load the left (first) channel (default is False)
        `Noverlap` :      overlap (as a ratio r such that :math:`0\leq r < 1`)

    Standard methods to manipulate the signal are:

        :func:`get_sub_signal` : Loads a subSignal

    """

    # more attr
#    filetype = ''       # file extension : .wav or .mp3
#    nframes = 0;        # number of
#    sample_width = 0;    # bit width of each frame
# segment_size = 0;      # in 16-bits samples : different from nframes which is
# internal wav representation
#    n_seg =0;     # numberof audio segments to be consideres
#    overlap = 0;         # overlapping rate between two consecutive segments

    # constructor
    def __init__(self, filepath, frame_size=16384 * 3, frame_duration=None, mono=False, Noverlap=0):
        self.location = filepath
        self.segment_size = frame_size

        # overlaps methods from SoundFile object
#        if (filepath[-4:] =='.wav'):
        wavfile = wave.open(filepath, 'r')
#        elif (filepath[-4:] =='.raw'):
#            wavfile = open()
        self.filetype = filepath[len(filepath) - 3:]
        self.channel_num = wavfile.getnchannels()
        self.fs = wavfile.getframerate()
        self.nframes = wavfile.getnframes()
        self.sample_width = wavfile.getsampwidth()

        self.overlap = Noverlap

        if frame_duration is not None:
            # optionally set the length in seconds, adapt to the sigbal
            # sampling frequency
            self.segment_size = int(math.floor(
                frame_duration * self.fs))

        if not (self.nframes % self.segment_size) == 0:
            _Logger.warning("Not a round number of segment: cropping ")
            self.nframes = (
                self.nframes / self.segment_size) * self.segment_size

        self.n_seg = int(math.floor(
            float(self.nframes) / (float(self.segment_size) * (1.0 - self.overlap))))

        if self.overlap >= 1:
            raise ValueError('Overlap must be in [0..1[ ')

#        print self.nframes, (float(self.segment_size) * (1.0 - self.overlap))
        self.n_seg -= int(math.ceil(self.overlap / (1.0 - self.overlap)))

        _Logger.info('Loaded ' + filepath + ' , ' + str(
            self.nframes) + ' frames of ' + str(self.sample_width) + ' bytes')
        _Logger.info('Type is ' + self.filetype + ' , ' + str(self.
                                                              channel_num) + ' channels at ' + str(self.fs))
        _Logger.info('Separated in ' + str(self.n_seg) + ' segments of size ' + str(self.segment_size) + ' samples overlap of ' + str(self.overlap * self.segment_size))
        self.length = self.n_seg * frame_size
        wavfile.close()

#    def readFrames(self , frameIndexes):
#        #self.data = array.array('h') #creates an array of ints
#        wavfile = wave.open(self.location, 'r')
#        str_bytestream = wavfile.readframes(self.nframes)
#        self.data = fromstring(str_bytestream,'h')
#        wavfile.close()

    def get_sub_signal(self, start_seg, seg_num, mono=False, normalize=False, channel=0, pad=0):
        """ Routine to actually read from the buffer and return a smaller signal instance

        :Returns:

            a :class:`Signal` object

        :Example:

            longSig = LongSignal(**myLongSigFilePath**, frameDuration=5) # Initialize long signal
            subSig = longSig.get_sub_signal(0,10) # Loads the first 50 seconds of data

        """

        # convert frame into bytes positions
        bFrame = int(start_seg * (self.segment_size * (1 - self.overlap)))
        nFrames = int(seg_num * self.segment_size)
        wavfile = wave.open(self.location, 'r')
        wavfile.setpos(bFrame)
#        print "Reading ",bFrame, nFrames, wavfile._framesize
        str_bytestream = wavfile.readframes(nFrames)
        data = np.fromstring(str_bytestream, 'h')
        wavfile.close()

        if self.channel_num > 1:
            reshapedData = data.reshape(nFrames, self.channel_num)
        else:
            if max(data.shape) > nFrames:
                nFrames = sum(data.shape)

            reshapedData = data.reshape(nFrames, )

        if mono:
            if len(reshapedData.shape) > 1:
                reshapedData = reshapedData[:, channel]

            reshapedData = reshapedData.reshape(nFrames, )

        SubSignal = Signal(reshapedData, self.fs, normalize=normalize)
        SubSignal.location = self.location

        if pad != 0:
            SubSignal.pad(pad)

# print "Created Signal of length " + str(SubSignal.length) +" samples " #of "
# + str(Signal.channel_num) + "channels"
        return SubSignal
