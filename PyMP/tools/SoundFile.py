#*****************************************************************************/
#                                                                            */
#                           Tools.SoundFile.py                               */
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
#                                                                            */
#******************************************************************************/

# A handle for audio files, including routines to Read, Write, Resample, Concatenate and Crop

class SoundFile:
    
    def Help(self):
        print """ python class SOUNDFILE
        Handles for reading, writing, resampling audio files in WAV format
        This utility manipulates multi-channel audio PCM signals as matrices
        """
    # properties
    data = None; # The data as a numpy array will be contained here
    
    # Constructor
    def __init__(self, filename):
        import wave,numpy
        if not len(filename) >0:
            print "Invalid file name"
            
        elif filename[-3:] == 'raw':
            self.filename = filename[(filename.rfind('/'))+1:]
            self.filepath = filename[:(filename.rfind('/'))]
            file = open(filename , 'r');

            self.filetype = filename[len(filename)-3:]
            self.nbChannel = 1 # fixed value
            self.sampleRate = 11025 # fixed value
            #self.nframes = file.getnframes() # fixed value
            self.sampleWidth = 2 # format is signed int16
            #data = data/max([abs(min(data)) max(data)]);
            str_bytestream = file.read(-1)
            self.data = numpy.fromstring(str_bytestream,'h')
            self.nframes = len(self.data)
            file.close()
             
        elif filename[-3:] == 'wav' or filename[-3:] == 'WAV':
            self.filename = filename[(filename.rfind('/'))+1:]
            self.filepath = filename[:(filename.rfind('/'))]
            
            file = wave.open(filename, 'r')
            self.filetype = filename[len(filename)-3:]
            self.nbChannel = file.getnchannels()
            self.sampleRate = file.getframerate()
            self.nframes = file.getnframes()
            self.sampleWidth = file.getsampwidth()
            
            #self.data = array.array('h') #creates an array of ints            
            str_bytestream = file.readframes(self.nframes)
            
            #print filename,self.sampleWidth, self.nbChannel , self.sampleRate,self.nframes
            
            if self.sampleWidth == 1:
                typeStr = 'int8'
            elif self.sampleWidth == 2:
                typeStr = 'int16'
            elif self.sampleWidth == 3:
                typeStr ='int24' # WARNING NOT SUPPORTED BY NUMPY
            elif self.sampleWidth == 4:
                typeStr = 'uint32'
            self.data = numpy.fromstring(str_bytestream,dtype=typeStr)
            
#            self.data = numpy.fromstring(str_bytestream,'h')
            file.close()
        else:
            raise TypeError('Audio format not recognized')
    
    # ToString method
    def ToString(self):
        print self.filename , 'in ' , self.filepath
        #print sf.filetype
        print "Channels :" , self.nbChannel
        print "Sample Frequency :" , self.sampleRate , " Hz"        
        print "Duration :" , (self.nframes), " frames  or " , float(self.nframes)/float(self.sampleRate) , ' seconds'
        print "Sample width :" , self.sampleWidth , ' bytes'
    
    # returns a numerical matrix of the file
    def GetAsMatrix(self):
        # nomalize?
        return self.data
    
    # writes to an output file
    def Write(self, destination, filename=''):
        import wave
        if not len(filename) >0:
            outputname = self.filename
        else:
            outputname = filename
            
        if not len(destination) >0:
            outputpath = self.filepath
        else:
            outputpath = destination
            
        file = wave.open(outputpath + outputname, 'w')    
        file.setparams((self.nbChannel, self.sampleWidth , self.sampleRate , self.nframes , 'NONE', 'noncompressed'))                
        file.writeframes(self.data.tostring())
        file.close()
        
        print 'File written to ' , outputpath + outputname
        
