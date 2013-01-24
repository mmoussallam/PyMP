#
#
#                       Classes.mdct.pymp_MDCTAtom
#
#
#
# M. Moussallam                             Created on Nov 12, 2012
# -----------------------------------------------------------------------
#
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330,
#  Boston, MA  02111-1307, USA.
#

"""
Module mdct.atom
====================

This class inherits from :class:`.Atom` and is used to represent and manipulate MDCT atoms.
:class:`Atom` Objects can either be constructed or recovered from Xml using :func:`fromXml`

"""

import numpy as np

from ..base import BaseAtom
from .. import win_server
from ..tools.mdct import imdct

try:
    from xml.dom.minidom import Document, Element
except ImportError:
    print "Atoms can be saved and recovered in xml format, but you XML library seems not available"

#global _PyServer
#_Logger
# Initializing the waveform server as a global variable
_PyServer = win_server.get_server()


class Atom(BaseAtom):
    """ MDCT atom class : implement real domain MDCT atoms of scale defined by the atom length

    An MDCT atom is defined for a length L, a frequency localization k and a frame parameter p by:

    .. math::

        \phi_{L,p,k}[n]=w_{L}[u]\sqrt{\\frac{2}{L}} \cos [ \\frac{\pi}{L} \left(u+ \\frac{L+1}{2}\\right) (k+\\frac{1}{2}) ]

    Additionnal attributes are:

            - `freq_bin` : MDCT frequency bin index (default is 0)

            - `frame` : the frame index of the atom

            - `reduced_frequency`: corresponding sampling frequency (default is 0)

            - `mdct_value` : the atom mdct coefficient

    Additionnal parameters for time-shifted atoms:

            - `time_shift`: time shift in samples related to the closest MDCT grid index*

            - `proj_score`: useful when atom is reprojected using say.. LOMP algorithm

    """

    # MDCT attibutes
    nature = 'MDCT'
    freq_bin = 0
    frame = 0
    reduced_frequency = 0
    mdct_value = 0.0
    time_position = 0;

    # for time-shift invariant atoms
    time_shift = None
    proj_score = None

    # constructor
    def __init__(self, scale=0, amp=0, timePos=0, freqBin=0, Fs=0, mdctCoeff=0):
        ''' Basic constructor setting atom parameters '''
        self.length = scale
        self.amplitude = amp
        self.time_position = timePos
        self.freq_bin = freqBin
        self.fs = Fs
        if self.length != 0:
            self.reduced_frequency = (
                float(self.freq_bin + 0.5) / float(self.length))
        self.mdct_value = mdctCoeff

    # Synthesis routine - always prefer the PyWinServer unless you want a
    # specific window to ba applied
    def synthesize(self, value=None):
        """ synthesizes the waveform
            Specifies the amplitude, otherwise it will be initialiazed as unit-normed """

#        global _PyServer
        binIndex = np.math.floor(self.reduced_frequency * self.length)
        if value is None:
            self.waveform = self.mdct_value * _PyServer.get_waveform(
                self.length, binIndex)
        else:
            self.waveform = value * _PyServer.get_waveform(
                self.length, binIndex)

    def synthesize_ifft(self, newValue=None):
        ''' DEPRECATED synthesis using Python fftw3 wrapper but no waveform server... a lot slower '''
        mdctVec = np.zeros(3 * self.length)
        if newValue is None:
            mdctVec[self.length + self.freq_bin] = self.mdct_value
        else:
            mdctVec[self.length + self.freq_bin] = newValue
        self.waveform = imdct(
            mdctVec, self.length)[0.75 * self.length: 1.75 * self.length]
        return self.waveform

    def __eq__(self, other):
        ''' overloaded equality operator, allows testing equality between atom objects '''
        if not isinstance(other, BaseAtom):
            return False
        return ((self.length == other.length) & (self.time_position == other.time_position) & (self.freq_bin == other.freq_bin))

    def __repr__(self):
        if self.time_shift is None:
            ts_str = 'None'
        else:
            ts_str = str(self.time_shift)
        return '''
MDCT Atom : length = %d value = %2.2f
frame = %d  time Position = %d   time shift = %s
frequency Bin = %d   Frequency = %2.2f Hz''' % (self.length,
                            self.mdct_value,
                             self.frame,
                             self.time_position,
                             ts_str,
                             self.freq_bin,
                             self.reduced_frequency * self.fs,)


    def inner_prod(self, otherAtom):
        """ DEPRECATED returns the inner product between current atom and the other one
            This method should be as fast as possible"""
        waveform1 = self.get_waveform()
        waveform2 = otherAtom.get_waveform()
        startIdx = max(
            self.frame * self.length, otherAtom.frame * otherAtom.length)
        stopIdx = min((self.frame + 1) * self.length, (otherAtom.
                                                       frame + 1) * otherAtom.length)

        if startIdx >= stopIdx:
            return 0
            # disjoint support

        start1 = startIdx - self.frame * self.length
        start2 = startIdx - otherAtom.frame * otherAtom.length
        duration = stopIdx - startIdx
        norm1 = np.sqrt(sum(waveform1 ** 2))
        norm2 = np.sqrt(sum(waveform2 ** 2))

        return (self.mdct_value * otherAtom.mdct_value) * sum(waveform1[start1: start1 + duration] * waveform2[start2: start2 + duration]) / (norm1 * norm2)
#        return sum(self.get_waveform() * otherAtom.get_waveform())

    def get_waveform(self):
        ''' Retrieve the atom waveform '''
        if self.waveform is not None:
            return self.waveform
        else:
            return self.synthesize_ifft()

    def copy(self):
        '''copycat routine '''
        copyAtom = Atom(self.length,
                        self.amplitude,
                        self.time_position,
                        self.freq_bin,
                        self.fs,
                        self.mdct_value)
        copyAtom.frame = self.frame
        copyAtom.proj_score = self.proj_score
        copyAtom.time_shift = self.time_shift
        if self.waveform is not None:
            copyAtom.waveform = self.waveform
        else:
            copyAtom.synthesize()
        return copyAtom

    def get_value(self):
        return self.mdct_value


#def fromXml(xmlNode):
#    ''' Construct an Object from the corresponding XML Node '''
#    if not isinstance(xmlNode, Element):
#        raise TypeError('Xml element not provided')
#
#    atom = Atom(int(xmlNode.getAttribute('length')),
#                1,
#                int(xmlNode.getAttribute('tP')),
#                int(xmlNode.getAttribute('fB')),
#                int(xmlNode.getAttribute('Fs')),
#                float(xmlNode.getAttribute('value')))
#
#    atom.frame = int(xmlNode.getAttribute('frame'))
#    if not xmlNode.getAttribute('time_shift') in ('None', ''):
#        atom.time_shift = int(xmlNode.getAttribute('time_shift'))
#    if not xmlNode.getAttribute('score') in ('None', ''):
#        atom.proj_score = float(xmlNode.getAttribute('score'))
#    return atom
