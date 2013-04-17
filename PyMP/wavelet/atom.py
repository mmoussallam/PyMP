'''
Module wavelet.atom
===================

This class inherits from :class:`.Atom` and is used to represent and manipulate wavelet based atoms.
:class:`WaveAtom` Objects can either be constructed or recovered from Xml using :func:`pymp_MDCTAtom.fromXml`

'''

from ..base import BaseAtom
from pywt import Wavelet


class WaveAtom(BaseAtom):

    nature = 'db8'
    level = 0
    frame = 0    
    time_position = 0

    # for time-shift invariant atoms
    time_shift = None
    proj_score = None

    def __init__(self, scale=0, amp=0, timePos=0, Fs=0,
                 nature='db8', level=2):
        '''
        Constructor
        '''
        self.length = scale
        self.amplitude = amp
        self.time_position = timePos
        self.fs = Fs
        self.nature = nature
        self.level = level

    def synthesize(self, value=None):
        """ synthesizes the waveform
            Specifies the amplitude, otherwise it will be initialiazed as unit-normed """
        wv = Wavelet(self.nature)
        scaling, wavelet, x = wv.wavefun(level=self.level)
        if value is None:
            value = self.amplitude

        self.waveform = value * wavelet
        print self.waveform.shape, self.level
        self.scaling = scaling
        self.x = x
#        return wavelet

    def __eq__(self, other):
        ''' overloaded equality operator, allows testing equality between atom objects '''
        if not isinstance(other, BaseAtom):
            return False
        return ((self.length == other.length) &
                (self.time_position == other.time_position) &
                (self.nature == other.nature))

    def __repr__(self):
        if self.time_shift is None:
            ts_str = 'No local adaptation'
        else:
            ts_str = str(self.time_shift)
        return '''
%s Wavelet Atom :
    length = %d
    frame = %d
    time Position = %d
    time shift = %s
    amplitude = %2.2f
    level = %d''' % (self.nature, self.length,
                             self.frame,
                             self.time_position,
                             ts_str,
                             self.amplitude,
                             self.level)

    def get_waveform(self):
        ''' Retrieve the atom waveform '''
        if self.waveform is None:
            return self.synthesize()

        return self.waveform

    def get_value(self):
        return self.amplitude
    
    
    def get_tf_info(self, Fs):
        """ returns energy centroids and spread in the time frequency plane """
        L = self.length
        K = L / 2
        freq = self.freq_bin
        f = float(freq) * float(Fs) / float(L)
        bw = (float(Fs) / float(K))  # / 2
        p = float(self.time_position + K) / float(Fs)
        l = float(L - K / 2) / float(Fs) 
        return f, bw, p, l