#*****************************************************************************/
#                                                                            */
#                           Tools.SoundFile.py                               */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */

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
        import wave, numpy
        print filename
        if not len(filename) >0:
            print "Invalid wavfile name"
        
        elif filename[-3:] == 'raw':
            self.filename = filename[(filename.rfind('/'))+1:]
            self.filepath = filename[:(filename.rfind('/'))]
            wavfile = open(filename , 'r');

            self.filetype = filename[len(filename)-3:]
            self.nbChannel = 1 # fixed value
            self.sampleRate = 11025 # fixed value
            #self.nframes = wavfile.getnframes() # fixed value
            self.sample_width = 2 # format is signed int16
            #data = data/max([abs(min(data)) max(data)]);
            str_bytestream = wavfile.read(-1)
            self.data = numpy.fromstring(str_bytestream,'h')
            self.nframes = len(self.data)
            wavfile.close()

        elif filename[-3:] == 'wav' or filename[-3:] == 'WAV':
            self.filename = filename[(filename.rfind('/'))+1:]
            self.filepath = filename[:(filename.rfind('/'))]

            wavfile = wave.open(filename, 'r')
            self.filetype = filename[len(filename)-3:]
            self.nbChannel = wavfile.getnchannels()
            self.sampleRate = wavfile.getframerate()
            self.nframes = wavfile.getnframes()
            self.sample_width = wavfile.getsampwidth()

            #self.data = array.array('h') #creates an array of ints
            str_bytestream = wavfile.readframes(self.nframes)

            #print filename,self.sampleWidth, self.nbChannel , self.sampleRate,self.nframes

            if self.sample_width == 1:
                typeStr = 'int8'
            elif self.sample_width == 2:
                typeStr = 'int16'
            elif self.sample_width == 3:
                typeStr ='int24' # WARNING NOT SUPPORTED BY NUMPY
            elif self.sample_width == 4:
                typeStr = 'uint32'
            self.data = numpy.fromstring(str_bytestream,dtype=typeStr)

#            self.data = numpy.fromstring(str_bytestream,'h')
            wavfile.close()
        else:
            raise TypeError('Audio format not recognized')

    # ToString method
    def ToString(self):
        print self.filename , 'in ' , self.filepath
        #print sf.filetype
        print "Channels :" , self.nbChannel
        print "Sample Frequency :" , self.sampleRate , " Hz"
        print "Duration :" , (self.nframes), " frames  or " , float(self.nframes)/float(self.sampleRate) , ' seconds'
        print "Sample width :" , self.sample_width , ' bytes'

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

        wavfile = wave.open(outputpath + outputname, 'w')
        wavfile.setparams((self.nbChannel, self.sample_width , self.sampleRate , self.nframes , 'NONE', 'noncompressed'))
        wavfile.writeframes(self.data.tostring())
        wavfile.close()

        print 'File written to ' , outputpath + outputname

