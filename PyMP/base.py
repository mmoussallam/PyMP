
#
#                       base classes
#
#
#
# M. Moussallam                             Created on Dec 27, 2012
# -----------------------------------------------------------------------
#
#
import numpy as np

class BaseSignal(object):
    """
    Attributes
    ----------
    *data* : array
        a numpy array containing the signal data
    *fs* : int
        The sampling frequency
    *location* :  str , optional
        Where the original file is located on the disk
    """
    
    data = np.array([])
    fs = 0
    location = ""

    def __init__(self):
        """ default constructor doesn't do anything """
    
    def add(self, atom):
        raise NotImplementedError("Subclass method not found")
        
    def subtract(self, atom):       
        raise NotImplementedError("Subclass method not found") 


class BaseAtom(object):
    """ Abstract Atom class interface:

    To implement a new type of atom, you must derive from this class.

    Attributes
    ----------
    nature: str
        A string describing the atom type (e.g MDCT, MCLT , GaborReal) default is MDCT
    waveform : array-like
        a numpy array that will contain the atom waveform
    amplitude : float
        the atom amplitude
    fs : int 
        the atom sampling frequency

    """

    # default values
    nature = 'Abstract'    
    waveform = None
    length = 0
    fs = 0
    amplitude = 0    

    def __init__(self):
        """" basic constructor doesn't do anything"""

    # mandatory functions
    def get_waveform(self):
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

    # members
    scale = 0
    residual_signal = BaseSignal()

    # methods
    def __init__(self):
        """ empty constructor """

    def update(self, residual):
        raise NotImplementedError("Subclass method not found")

    def find_max(self):
        raise NotImplementedError("Subclass method not found")

    def get_max_atom(self):
        raise NotImplementedError("Subclass method not found")

class BaseDico(object):
    """ This class creates an interface that any type of dictionary should reproduce
        in order to be used correclty by Pursuit algorithm in this framework:

            - `sizes`: a list of scales

            - `blocks`: a list of :class:`.BaseBlock` that handles the projection of a residual signal along with
                        the selection of a projection maximum given a transform and a criteria
        """
    # attributes:
    sizes = None    
    blocks = []    
    nature = 'Abstract'    

    def __init__(self):
        """ default constructor doesn't do anything"""

    def get_pad(self):
        """ the amount of zeroes that need to be added on the sides """
        return self.sizes[-1]  # last element of the list should be the biggest, 

    def update(self, res, it):
        raise NotImplementedError("Subclass method not found")

    def get_best_atom(self):
        raise NotImplementedError("Subclass method not found")
    
                        
class BaseApprox(object):
    """
    Attributes
    ----------
    *dico* : :class:`.BaseDico`
        the dictionary (as a :class:`.BaseDico` object) from which it has been constructed
    *atoms* : list
        a list of :class:`.BaseAtom` objets
    *original_signal* : :class:`.BaseSignal`
        a :class:`.Signal` object that is the original signal
    *recomposed_signal* :  :class:`.BaseSignal`
        a :class:`.BaseSignal` objet for the reconstructed signal (as the weighted sum of atoms specified in the atoms list)
    """
    dico = BaseDico()
    atoms = []
    original_signal = BaseSignal()
    recomposed_signal = BaseSignal()
    
    def __init__(self):
        """ default constructor doesn't do anything """
    
    def add(self, atom):
        raise NotImplementedError("Subclass method not found")
        
    def remove(self, atom):       
        raise NotImplementedError("Subclass method not found")   
    
 