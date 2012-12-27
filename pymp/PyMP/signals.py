#
#                                                                            */
#                           Signal.py                                   */
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
Module signals
==================

The main class is :class:`Signal`, it can be instantiated from a numpy array using
the main constructor (mutlichannel is allowed).

It can also be created from a wav file using the :func:`InitFromFile` static routine

Longer Signals are handled with :class:`LongSignal` objects.
                                                                          
'''



from tools import SoundFile
from numpy import array , min , concatenate , zeros, ones , fromstring, sum , sin , pi , arange, dot , conj, floor,abs, exp, real , isnan
import matplotlib.pyplot as plt
from base import BaseAtom
import log
import math , wave , struct
global _Logger
_Logger = log.Log('Signal', imode=False)
#from operator import add, mul
class Signal(object):
    """This file defines the main class handling audio signal in the Pymp Framework.

    A Signal is fairly a numpy array (called dataVec) and a collection of attributes.
    Longer signals should not be loaded in memory. the `LongSignal` class allows
    to define this kind of signal, and slice it in frames overlapping or not.
    
    Attributes:
    
        `channelNumber`:The number of channel
        
        `length`:                The length in samples (integer) of the signal (total dimension of the numpy array is channelNumber x length)
         
        `samplingFrequency`:     The sampling frequency
        
        `location`:              (optional) Where the original file is located on the disk 
        
        `sampleWidth` :          Various bit format exist for wav file, this allows to handle it
        
        `isNormalized` :         A boolean telling is the numpy array has been normalized (which here means its values are between -1 and 1)
        
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
    dataVec = [];                # 
    channelNumber = 0;
    length = 0;
    samplingFrequency = 0;
    location = "";
    sampleWidth = 2;
    isNormalized = False;
    energy = 0
    
    # Constructor
    def __init__(self , data = [] , Fs = 0  , doNormalize= False , debugLevel = None):
        ''' Simple constructor from a numpy array (data) '''
        if debugLevel is not None:
            _Logger.setLevel(debugLevel)
        self.dataVec = array(data)
        self.length = len(data)
        self.samplingFrequency = Fs;
        
        if len(self.dataVec.shape) > 1:
            self.channelNumber = min(self.dataVec.shape);
        else:
            self.channelNumber = 1;
            
        if doNormalize & (self.length > 0):
            _Logger.info('Normalizing Signal');
            normaFact = self.dataVec.max()
            self.dataVec = self.dataVec.astype(float) / float(normaFact)
            self.isNormalized = True
        
        if len(data) > 0:
            self.energy = sum(data**2)
            _Logger.info('Signal created with energy: ' + str(self.energy));
    
    def normalize(self):
        ''' makes sure all values of the array are between -1 and 1 ''' 
        _Logger.info('Normalizing Signal');
        normaFact = abs(self.dataVec).max()
        self.dataVec = self.dataVec.astype(float) / float(normaFact)
        self.energy = sum(self.dataVec**2)
        self.isNormalized = True
       
    def plot(self , legend=None):
        ''' DEPRECATED plot the array using matplotlib ''' 
        plt.plot(self.dataVec)
        if legend !=None:
            plt.legend((legend))
        plt.show()

    # cropping routine
    def crop(self, startIndex = 0 , stopIndex = None):
        ''' cropping routine, usage is quite obvious '''
        if (startIndex > self.length) | (startIndex < 0):
#            raise ValueError("Starting Index beyond signal dimensions")
            _Logger.error('Starting Index '+str(startIndex)+' is beyond signal length of ' + str(startIndex));
        if (stopIndex > self.length) | (stopIndex < 0):
            _Logger.error('Stopping Index '+str(stopIndex)+' is beyond signal length of ' + str(self.length));
#            raise ValueError("Stopping Index beyond signal dimensions")
        if (len(self.dataVec.shape) >1 ):
            if (self.dataVec.shape[0] > self.dataVec.shape[1]) :
                self.dataVec = self.dataVec[startIndex:stopIndex , :]
            else:
                self.dataVec = self.dataVec[: , startIndex:stopIndex]
        else:
            self.dataVec = self.dataVec[startIndex:stopIndex]
        self.length = stopIndex - startIndex    
    
    def copy(self):
        copiedSignal = Signal(self.dataVec.copy() , self.samplingFrequency)
        copiedSignal.location = self.location
        copiedSignal.channelNumber = self.channelNumber
        copiedSignal.sampleWidth = self.sampleWidth
        copiedSignal.isNormalized = self.isNormalized
        return copiedSignal;
    
    def subtract(self , atom , debug=0 , preventEnergyIncrease = True):
        ''' Subtracts the atom waveform from the signal, at the position specified by the atom.timeLocalization property
            if preventEnergyIncrease is True, an error will be raised if subtracting the atom increases the signal's energy '''
        if not isinstance(atom , BaseAtom):
            raise TypeError("argument provided is not an atom")                    
        if atom.waveform is None:
#            print "Resynth"
            atom.synthesize()      
        # Keep track of the energy to make 
        oldEnergy = (self.dataVec[atom.timePosition : atom.timePosition + atom.length ]**2).sum()

        # In high debugging levels, show the process using matplotlib 
        if debug>2:
            plt.figure()
            plt.plot(self.dataVec[atom.timePosition : atom.timePosition + atom.length ])
            plt.plot(atom.waveform)
            plt.plot(self.dataVec[atom.timePosition : atom.timePosition + atom.length ] - atom.waveform , ':')
            plt.legend(('Signal' , 'Atom substracted', 'residual'))
            plt.show()
        
        # actual subtraction
        self.dataVec[atom.timePosition : atom.timePosition + atom.length ] -= atom.waveform
        
        newEnergy = (self.dataVec[atom.timePosition : atom.timePosition + atom.length ]**2).sum()
        
        # Test
        if (newEnergy > oldEnergy) and preventEnergyIncrease:
            # do not substract the atom
            self.dataVec[atom.timePosition : atom.timePosition + atom.length ] += atom.waveform
            # get out of here
            _Logger.error('Warning : Substracted Atom created energy : at pos: ' 
                             + str(atom.timePosition) + ' k: ' + ' Amp: ' + str(atom.getAmplitude()) +
                             str(atom.frequencyBin) + ' timeShift :' + 
                             str(atom.timeShift)+ ' scale : ' + 
                             str(atom.length) + ' from ' + str(oldEnergy) + ' to ' + str(newEnergy))
            raise ValueError('see Logger')
        
        self.energy = self.energy - oldEnergy + newEnergy
        
    
    def downsample(self , newFs):
        """ downsampling the signal by taking only a portion of the data """
       
        if newFs >= self.samplingFrequency:
            raise ValueError('new sampling frequency is bigger than actual, try upsampling instead')
       
        # ratio of selected points
        subRatio = float(self.samplingFrequency)/float(newFs);
        indexes = floor(arange(0, self.length, subRatio))        
#        print indexes
        self.dataVec = self.dataVec[indexes.tolist()]
        self.length = len(self.dataVec)
        self.samplingFrequency = newFs;
       
    def add(self , atom , window = None):
        ''' adds the contribution of the atom at the position specified by the atom.timeLocalization property '''
        if not isinstance(atom , BaseAtom):
            raise TypeError("argument provided is not an atom")
        
        if atom.waveform is None:
            print "Resynthesizing"
            atom.synthesize()
        
        localEnergy = sum(self.dataVec[atom.timePosition : atom.timePosition + atom.length ]**2)
        
        # do sum calculation
        try:
            self.dataVec[atom.timePosition : atom.timePosition + atom.length ] += atom.waveform        
        except:            
            _Logger.error('Mispositionned atom: ' + str(atom.timePosition) + ' and length ' + str(atom.length))
#        # update energy value
        self.energy += sum(self.dataVec[atom.timePosition : atom.timePosition + atom.length ]**2) - localEnergy        


    # Pad edges with zeroes
    def pad(self , zero_pad):
        ''' Pad edges with zeroes '''
        try:
            self.dataVec = concatenate((  concatenate((zeros(zero_pad) , self.dataVec)) , zeros(zero_pad) ))
            self.length = len(self.dataVec)
        except ValueError:
            print(self.dataVec.shape)
            _Logger.error("Wrong dimension number %s" % str(self.dataVec.shape))
    
    def depad(self , zero_pad):
        ''' Remove zeroes from the edges WARNING: no test on the deleted data: make sure these are zeroes! '''
        try:
            self.dataVec = self.dataVec[zero_pad:-zero_pad];
            self.length = len(self.dataVec)
        except ValueError:
            print(self.dataVec.shape)
            _Logger.error("Wrong dimension number %s" % str(self.dataVec.shape))
           
    
    def doWindow(self , K):
        ''' DEPRECATED '''
        self.dataVec[0:K] *= sin((arange(K).astype(float))*pi/(2*K));
        self.dataVec[-K:] *= sin((arange(K).astype(float))*pi/(2*K) + pi/2);
    
    def write(self, fileOutputPath , pad =0):
        ''' Write the current signal at the specified location in wav format
            
            This is done using the wave python library'''
        if pad>0:
            self.depad(pad)
        
        if self.energy == 0:
            _Logger.warning("Zero-energy signal")
            self.dataVec = zeros(self.length,);
#            print "Warning!!!! Zero-energy signal:"
        
        file = wave.open(fileOutputPath ,'wb')       
        file.setparams((self.channelNumber, 
                            self.sampleWidth, 
                            self.samplingFrequency, 
                            self.length, 
                            'NONE', 'not compressed'))
       
        # prepare binary output
        values = []
        if not self.isNormalized:
            self.normalize()
        
        
        for i in range(len(self.dataVec)):
            if isnan(self.dataVec[i]):
                _Logger.warning("NaN data found: replaced by 0")
                self.dataVec[i] = 0;
            if self.dataVec[i]<-1:
                _Logger.warning(str(i)+'th sample was below -1: cropping')
                self.dataVec[i] = -1;
            if self.dataVec[i]>1:
                _Logger.warning(str(i)+'th sample was over 1: cropping')
                self.dataVec[i] = 1;
            value = int(16384*self.dataVec[i])
            packed_value = struct.pack('h', value)
            values.append(packed_value)
            
        value_str = ''.join(values)
        file.writeframes(value_str)

#        else:
#            _Logger.warning("Warning BUGFIX not done yet, chances are writing failed");
#            file.writeframes((self.dataVec).astype(int).tostring())
        file.close()
 
    def WignerVPlot(self, window=True):
        """ Calculate the wigner ville distribution and plots it 
        WARNING: not sufficiently tested!"""
        from numpy.fft import ifft, fft, ifftshift, fftshift

        N = self.length;
        Ex = self.dataVec;
        
        # ensure the signal is mono (take left canal if stereo)
        if len(Ex.shape)>1 and not (Ex.shape[1] == 1):
            Ex = Ex[:,0];
        
        # reshape anyway as a column vector
        Ex = Ex.reshape((N,1));
        
        if window:
            from scipy.signal import hann
            Ex *= hann(N).reshape((N,1))
        
        x = ifftshift((arange(0,N)-N/2)*2*pi/(N-1));                            #%   Generate linear vector
        X = arange(0,N)-N/2;
        
        x = x.reshape((N,1))
        X = X.reshape((1,N))

        A = dot(1j*x,X/2)
        EX1 = ifft( dot(fft(Ex),ones((1,N)))*exp( A ));                    #%   +ve shift
        EX2 = ifft( dot(fft(Ex),ones((1,N)))*exp( dot(-1j*x,X/2) ));                    #%   -ve shift
        

        
        tmp0 = EX1*conj(EX2);
        
        print 

        
        print tmp0.shape
        tmp1 = fftshift(tmp0,(1,));
        tmp2 = fft(tmp1, axis=1)
        
        W = real(fftshift(tmp2, (1,)));        #%   Wigner function   


        return W;
    
def InitFromFile(filepath , forceMono = False , doNormalize= False , debugLevel=None):
    ''' Static method to create a Signal from a wav file on the disk 
        This is based on the wave Python library through the use of the Tools.SoundFile class
        '''        
    if debugLevel is not None:
        _Logger.setLevel(debugLevel)
    Sf = SoundFile.SoundFile(filepath)    
    #print Sf.GetAsMatrix().shape
    reshapedData = Sf.GetAsMatrix().reshape(Sf.nframes , Sf.nbChannel)
    if forceMono:
        reshapedData = reshapedData[: , 0]
    
    sig = Signal(reshapedData , Sf.sampleRate , doNormalize)    
    sig.sampleWidth = Sf.sampleWidth
    sig.location = filepath
    
    _Logger.info("Created Signal of length " + str(sig.length) +" samples of " + str(sig.channelNumber) + "channels");
    #print "Created Signal of length " + str(Signal.length) +" samples " #of " + str(Signal.channelNumber) + "channels"
    return sig



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
    
        :func:`getSubSignal` : Loads a subSignal
        
    """
    
    # more attr
#    filetype = ''       # file extension : .wav or .mp3
#    nframes = 0;        # number of 
#    sampleWidth = 0;    # bit width of each frame
#    segmentSize = 0;      # in 16-bits samples : different from nframes which is internal wav representation
#    segmentNumber =0;     # numberof audio segments to be consideres
#    overlap = 0;         # overlapping rate between two consecutive segments
    
    # constructor
    def __init__(self, filepath , frameSize= 16384*3, frameDuration = None, forceMono = False , Noverlap = 0 ):
        self.location = filepath
        self.segmentSize = frameSize
        
        
        # overlaps methods from SoundFile object
#        if (filepath[-4:] =='.wav'):
        file = wave.open(filepath, 'r')
#        elif (filepath[-4:] =='.raw'):
#            file = open()
        self.filetype = filepath[len(filepath)-3:]
        self.channelNumber = file.getnchannels()
        self.samplingFrequency = file.getframerate()
        self.nframes = file.getnframes()
        self.sampleWidth = file.getsampwidth()
        
        self.overlap = Noverlap
        
        if frameDuration is not None:
            # optionally set the length in seconds, adapt to the sigbal sampling frequency
            self.segmentSize = math.floor(frameDuration*self.samplingFrequency);
        
        
        self.segmentNumber = math.floor(self.nframes / (self.segmentSize * (1-self.overlap) ));
        
        if self.overlap >= 1:
            raise ValueError('Overlap must be in [0..1[ ')
        
        
        self.segmentNumber -= self.overlap/(1- self.overlap)
        
        _Logger.info('Loaded ' +  filepath + ' , ' + str(self.nframes) + ' frames of ' + str(self.sampleWidth) + ' bytes')
        _Logger.info( 'Type is ' + self.filetype + ' , ' + str(self.channelNumber) + ' channels at ' + str(self.samplingFrequency))
        _Logger.info( 'Separated in ' + str(self.segmentNumber) + ' segments of size ' + str(self.segmentSize) +  ' samples overlap of ' +  str(self.overlap * self.segmentSize))
        self.length = self.segmentNumber * frameSize
        file.close()
    
#    def readFrames(self , frameIndexes):
#        #self.data = array.array('h') #creates an array of ints            
#        file = wave.open(self.location, 'r')
#        str_bytestream = file.readframes(self.nframes)
#        self.data = fromstring(str_bytestream,'h')
#        file.close()
        
        
    def getSubSignal(self , startSegment , segmentNumber ,forceMono=False, doNormalize=False , channel=0 , padSignal = 0):
        """ Routine to actually read from the buffer and return a smaller signal instance 
        
        :Returns:
        
            a :class:`Signal` object
            
        :Example:
        
            longSig = LongSignal(**myLongSigFilePath**, frameDuration=5) # Initialize long signal
            subSig = longSig.getSubSignal(0,10) # Loads the first 50 seconds of data
        
        """
        
        # convert frame into bytes positions
        bFrame = startSegment*(self.segmentSize * (1 - self.overlap)) 
        nFrames = int(segmentNumber*self.segmentSize)
        file = wave.open(self.location, 'r')
        file.setpos(bFrame)
        str_bytestream = file.readframes(nFrames)
        data = fromstring(str_bytestream,'h')
        file.close()
        
        if self.channelNumber > 1:
            reshapedData = data.reshape(nFrames , self.channelNumber)
        else:
            reshapedData = data.reshape(nFrames , )
        if forceMono:
            if len(reshapedData.shape)>1:
                reshapedData = reshapedData[: , channel]
                
            reshapedData = reshapedData.reshape(nFrames , )
    
        SubSignal = Signal(reshapedData , self.samplingFrequency , doNormalize)    
        SubSignal.location = self.location
        
        if padSignal != 0:
            SubSignal.pad(padSignal)
        
#        print "Created Signal of length " + str(SubSignal.length) +" samples " #of " + str(Signal.channelNumber) + "channels"
        return SubSignal
        