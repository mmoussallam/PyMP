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
Module pymp_MDCTAtom
====================
                                                                          
This class inherits from :class:`.pymp_Atom` and is used to represent and manipulate MDCT atoms.
:class:`pymp_MDCTAtom` Objects can either be constructed or recovered from Xml using :func:`pymp_MDCTAtom.fromXml`
                                                                          
"""


from Classes.pymp_Atom import pymp_Atom
from Classes import  PyWinServer
from numpy import math , zeros
from Tools.mdct import imdct

try:
    from xml.dom.minidom import Document , Element
except ImportError:
    print "Atoms can be saved and recovered in xml format, but you XML library seems not available"
        
global _PyServer , _Logger
# Initializing the waveform server as a global variable
_PyServer = PyWinServer.PyServer()

class pymp_MDCTAtom(pymp_Atom):
    """ MDCT atom class : implement real domain MDCT atoms of scale defined by the atom length
        
    An MDCT atom is defined for a length L, a frequency localization k and a frame parameter p by:

    .. math::
    
        \phi_{L,p,k}[n]=w_{L}[u]\sqrt{\\frac{2}{L}} \cos [ \\frac{\pi}{L} \left(u+ \\frac{L+1}{2}\\right) (k+\\frac{1}{2}) ]
    
    Additionnal attributes are:
    
            - `frequencyBin` : MDCT frequency bin index (default is 0)
            
            - `frame` : the frame index of the atom
            
            - `reducedFrequency`: corresponding sampling frequency (default is 0)
            
            - `mdctValue` : the atom mdct coefficient 
            
    Additionnal parameters for time-shifted atoms:
    
            - `timeShift`: time shift in samples related to the closest MDCT grid index*
            
            - `projectionScore`: useful when atom is reprojected using say.. LOMP algorithm
            
    """

    # MDCT attibutes
    nature = 'MDCT'
    frequencyBin = 0;
    frame = 0;
    reducedFrequency = 0;
    mdctValue = 0; 

        
    # for time-shift invariant atoms
    timeShift = None
    projectionScore = None
    
    # constructor
    def __init__(self , len=0 , amp = 0 , timePos=0 , freqBin = 0 , Fs = 0 , mdctCoeff = 0):
        ''' Basic constructor setting atom parameters '''
        self.length = len
        self.amplitude = amp
        self.timePosition = timePos
        self.frequencyBin = freqBin                
        self.samplingFrequency = Fs
        if self.length != 0:
            self.reducedFrequency = (float(self.frequencyBin + 0.5) / float(self.length))
        self.mdct_value = mdctCoeff
        
    # Synthesis routine - always prefer the PyWinServer unless you want a specific window to ba applied
    def synthesize(self ,  value = None):
        """ synthesizes the waveform
            Specifies the amplitude, otherwise it will be initialiazed as unit-normed """   
        
        global _PyServer
        binIndex = math.floor(self.reducedFrequency * self.length);
        if  value is None:
            self.waveform =  self.mdct_value * _PyServer.getWaveForm(self.length , binIndex)
        else:
            self.waveform = value * _PyServer.getWaveForm(self.length , binIndex)

    
    def synthesizeIFFT(self , newValue=None):        
        ''' DEPRECATED synthesis using Python fftw3 wrapper but no waveform server... a lot slower '''
        mdctVec = zeros(3*self.length);
        if newValue is None:
            mdctVec[self.length +  self.frequencyBin] = self.mdct_value;
        else:
            mdctVec[self.length +  self.frequencyBin] = newValue;
        self.waveform =  imdct(mdctVec , self.length)[0.75*self.length : 1.75*self.length]
        return self.waveform

    
    def __eq__(self, other):
        ''' overloaded equality operator, allows testing equality between atom objects '''
        if not isinstance(other , pymp_Atom):
            return False
        return ( (self.length == other.length) & (self.timePosition == other.timePosition) & (self.frequencyBin == other.frequencyBin))
    
    def toXml(self , xmlDoc):
        ''' Useful routine to output the object as an XML node '''
        if not isinstance(xmlDoc, Document):
            raise TypeError('Xml document not provided')
        
        atomNode = xmlDoc.createElement('Atom')
        atomNode.setAttribute('nature',str(self.nature))
        atomNode.setAttribute('length',str(self.length))
        atomNode.setAttribute('tP',str(int(self.timePosition)))
        atomNode.setAttribute('fB',str(int(self.frequencyBin)))
        atomNode.setAttribute('frame',str(int(self.frame)))
        atomNode.setAttribute('value',str(self.mdct_value))
        atomNode.setAttribute('Fs',str(self.samplingFrequency))
        
        if self.projectionScore is not None:
            atomNode.setAttribute('timeShift',str(self.timeShift))
            atomNode.setAttribute('score',str(self.projectionScore))

        return atomNode
    
    def innerProd(self, otherAtom):
        """ DEPRECATED returns the inner product between current atom and the other one 
            This method should be as fast as possible"""
        waveform1 = self.getWaveform()
        waveform2 = otherAtom.getWaveform()
        startIdx = max(self.frame * self.length , otherAtom.frame * otherAtom.length)
        stopIdx =  min((self.frame+1) * self.length , (otherAtom.frame+1) * otherAtom.length)
        
        if startIdx >= stopIdx:
            return 0; # disjoint support
        
        start1 = startIdx- self.frame * self.length
        start2 = startIdx- otherAtom.frame * otherAtom.length
        duration = stopIdx - startIdx
        norm1 = math.sqrt(sum(waveform1**2))
        norm2 = math.sqrt(sum(waveform2**2))
        
        return (self.mdct_value * otherAtom.mdct_value)* sum( waveform1[start1: start1+duration] *  waveform2[start2: start2+duration]) / (norm1*norm2) 
#        return sum(self.getWaveform() * otherAtom.getWaveform())
    
    def getWaveform(self):
        ''' Retrieve the atom waveform '''
        if self.waveform is not None:
            return self.waveform
        else:
            return self.synthesizeIFFT()
  
    def copy(self):
        '''copycat routine '''
        copyAtom = pymp_MDCTAtom(self.length, 
                                   self.amplitude, 
                                   self.timePosition, 
                                   self.frequencyBin, 
                                   self.samplingFrequency,
                                   self.mdct_value)
        copyAtom.frame = self.frame
        copyAtom.projectionScore =  self.projectionScore
        copyAtom.timeShift =  self.timeShift
        if self.waveform is not None:
            copyAtom.waveform =  self.waveform
        else:
            copyAtom.synthesize();
        return copyAtom
    
    def getAmplitude(self):
        return self.mdct_value;
    
def fromXml(xmlNode):
    ''' Construct an Object from the corresponding XML Node '''
    if not isinstance(xmlNode, Element):
        raise TypeError('Xml element not provided')
    
    atom = pymp_MDCTAtom(int(xmlNode.getAttribute('length')), 
                              1, 
                              int(xmlNode.getAttribute('tP')), 
                              int(xmlNode.getAttribute('fB')), 
                              int(xmlNode.getAttribute('Fs')),  
                              float(xmlNode.getAttribute('value')))
    
    atom.frame = int(xmlNode.getAttribute('frame'))
    if not xmlNode.getAttribute('timeShift') in ('None' ,'') :
        atom.timeShift = int(xmlNode.getAttribute('timeShift'))
    if not xmlNode.getAttribute('score') in ('None' ,''):
        atom.projectionScore = float(xmlNode.getAttribute('score'))
    return atom