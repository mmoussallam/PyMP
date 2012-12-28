
#
#                       base classes
#
#
#
# M. Moussallam                             Created on Dec 27, 2012
# -----------------------------------------------------------------------
#
#


class BaseAtom(object):
    """ Abstract Atom class interface:

    To implement a new type of atom, you must derive from this class.

    Attributes:

        - `nature`: A string describing the atom type (e.g MDCT, MCLT , GaborReal) default is MDCT

        - `length: Sample length of the atom (default is 0)

        - `timePosition`: The index of the first atom sample in a signal

        - `waveform`: a numpy array that will contain the atom waveform

        - `amplitude`: the atom amplitude

        - `samplingFrequency`: the atom sampling frequency

    """

    # default values
    nature = 'Abstract'
    length = 0
    timePosition = 0
    waveform = None
    samplingFrequency = 0
    amplitude = 0
    phase = None

    def __init__(self):
        self.length = 0
        self.amplitude = 0
        self.timePosition = 0

    # mandatory functions
    def getWaveform(self):
        # A function to retrieve the atom waveform
        return self.waveform


class BaseBlock(object):
    ''' A block is an instance handling projections for Matching Pursuit.

        Mandatory fields:
            - type : the type of dictionary (e.g  Gabor , MDCT , Haar ...)
            - scale : the scale of the block
            - residualSignal : a py_pursuit_Signal instance that describes the current residual

        Mandatory methods
            - update :  updates the inner products table
            - getMaximum : retrieve the maximum absolute value of inner products
            - getMaxAtom : return a corresponding Atom instance'''

    #members
    scale = 0
    residualSignal = None

    # methods
    def __init__(self):
        """ empty constructor """


class BaseDico(object):
    """ This class creates an interface that any type of dictionary should reproduce
        in order to be used correclty by Pursuit algorithm in this framework:

            - `sizes`: a list of scales

            - `blocks`: a list of blocks that handles the projection of a residual signal along with
                        the selection of a projection maximum given a transform and a criteria
        """
    # attributes:
    sizes = None
    tolerances = None
    blocks = None
    overlap = 0.5
    nature = 'Abstract'

    def __init__(self):
        """ default constructor doesn't do anything"""

    def getN(self):
        return self.sizes[-1]  # last element of the list should be the biggest
