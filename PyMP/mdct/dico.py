#
#
#                       Classes.mdct.Dico
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
Module mdct.dico
====================

This class inherits from :class:`.BaseDico` and is used to represent and manipulate multiscale MDCT dictionaries.
Dictionaries are mostly implemented as a collection of :class:`pymp_MDCTBlock` of various kind, according to the
type of pursuit that is seeked.

This module describes 3 kind of blocks:
    - :class:`Dico`  is a Dico based on the standard MDCT transform. Which means atoms localizations
                 are constrained by the scale of the transform
                 The atom selected is simple the one that maximizes the correlation with the residual
                 it is directly indexes by the max absolute value of the MDCT bin
                 No further optimization of the selected atom is performed

    - :class:`LODico`   is a dictionary that performs a local optimization of the time localization of the selected atom

    - :class:`FullDico`  this object simulates a dictionary where atoms at all time localizations are available
                  This kind of dictionary can be thought of as a Toeplitz matrix
                  BEWARE: memory consumption and CPU load will be very high! only use with very small signals
                  (e.g. 1024 samples or so) Fairly intractable at higher dimensions
"""


import math
from xml.dom.minidom import Document, Element

from numpy import zeros, abs, sum, array

from .. import log
from ..base import BaseDico, BaseAtom
from . import block

global _Logger
_Logger = log.Log('MDCTDico', level=0)


class Dico(BaseDico):
    """ class to handle multiscale MDCT dictionaries using MDCT blocks"""

    sizes = []
    tolerances = []
    nature = 'MDCT'
    blocks = None
    max_block_score = 0
    best_current_block = None
    starting_touched_index = 0
    ending_touched_index = -1

    # DEPRECATED
    forceHF = False
    use_c_optim = True

    def __init__(self, sizes=[], useC=True, forceHF=False, parallel=False, debug_level=None):
        if debug_level is not None:
            _Logger.setLevel(debug_level)

        self.sizes = sizes
        self.tolerances = [2 for i in self.sizes]
        self.use_c_optim = useC
        self.forceHF = forceHF
        self._pp = parallel
        _Logger.info('New dictionary created with sizes : ' + str(self.sizes))

    def find_block_by_scale(self, size):
        ''' Returns the index of the block corresponding to the given size or None if not found'''
        for i in range(len(self.sizes)):
            if size == self.sizes[i]:
                return i
        return None

    def initialize(self, residualSignal):
        ''' Create the collection of blocks specified by the MDCT sizes '''
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for mdctSize in self.sizes:
            self.blocks.append(block.Block(mdctSize,
                                           residualSignal, useC=self.use_c_optim, forceHF=self.forceHF))

    def init_proj_matrix(self, itNumbers):
        """ method used for monitoring the projection along iterations
            ONLY for DEV purposes"""
        projShape = self.blocks[0].projs_matrix.shape
        # Here all blocks have same number of projections
        return zeros((projShape[0] * len(self.sizes), itNumbers))

    def update_proj_matrix(self, projMatrix, iterationNumber, normedProj=False):
        """ method used for monitoring the projection along iterations
            ONLY for DEV purposes"""

        Lproj = self.blocks[0].projs_matrix.shape[0]
        for blockIdx in range(len(self.blocks)):
#            if normedProj:
# normCoeff = sqrt(sum(self.blocks[blockIdx].projectionMatrix**2))
#            else:
#            normCoeff = 1
            projMatrix[blockIdx * Lproj:(blockIdx + 1) * Lproj, iterationNumber] = self.blocks[
                blockIdx].projs_matrix  # /normCoeff

    def update(self, residualSignal, iteratioNumber=0, debug=0):
        ''' Update the projections in each block, only where it needs to be done as specified '''
        self.max_block_score = 0
        self.best_current_block = None

        if self.forceHF:
            self.maxHFBlockScore = 0
            self.bestCurrentHFBlock = None

        for block in self.blocks:
            startingTouchedFrame = int(
                math.floor(self.starting_touched_index / (block.scale / 2)))
            if self.ending_touched_index > 0:
                endingTouchedFrame = int(math.floor(self.ending_touched_index /
                                        (block.scale / 2))) + 1  # TODO check this
            else:
                endingTouchedFrame = -1
            _Logger.info("block: " + str(block.scale) + " : " +
                         str(startingTouchedFrame) + " " + str(endingTouchedFrame))

            block.update(
                residualSignal, startingTouchedFrame, endingTouchedFrame)

            if abs(block.max_value) > self.max_block_score:
                self.max_block_score = abs(block.max_value)
                self.best_current_block = block

    def get_best_atom(self, debug):
        if self.best_current_block == None:
            raise ValueError("no best block constructed, make sure inner product have been updated")

        if debug > 2:
            self.best_current_block.plot_proj_matrix()

        return self.best_current_block.get_max_atom()

    def compute_touched_zone(self, previousBestAtom):
        ''' update zone computed from the previously selected atom '''
        self.starting_touched_index = previousBestAtom.time_position  # - previousBestAtom.length/2
        self.ending_touched_index = self.starting_touched_index + \
            1.5 * previousBestAtom.length

    def to_xml(self, doc):
        ''' A routine to convert the dictionary to an XML node '''
        if not isinstance(doc, Document):
            raise TypeError('Xml document not provided')

        DicoNode = doc.createElement('Dictionary')
        DicoNode.setAttribute('nature', str(self.nature))
        DicoNode.setAttribute('class', self.__class__.__name__)
        SizesNode = doc.createElement('Sizes')
        SizesNode.setAttribute('number', str(len(self.sizes)))
        for size in self.sizes:
            sizeNode = doc.createElement('Size')
            sizeNode.setAttribute('scale', str(size))
            SizesNode.appendChild(sizeNode)

        DicoNode.appendChild(SizesNode)
        return DicoNode

    def get_atom_key(self, atom, sigLength):
        ''' Get the atom index in the dictionary '''
        if not isinstance(atom, BaseAtom):
            return None

        block = [i for i in range(
            len(self.sizes)) if self.sizes[i] == atom.length][0]
        n = atom.time_position + 1
        frame = math.floor(float(n) / float(atom.length / 2)) + 1
        return int(block * sigLength + frame * float(atom.length / 2) + atom.freq_bin)

    #
    def get_projections(self, indexes, sigLength):
        """ additional method provided for Gradient Pursuits
            indexes formalism: key as in the py_pursuit_Approx Object :
            int(block*self.length +  frame*float(atom.length /2) + atom.frequencyBin)"""

        projections = []
        for index in indexes:
            block = int(math.floor(index / sigLength))
# frame = math.floor( (index - block*sigLength)   /  (self.sizes[block]  /2))
            projections.append(self.blocks[block].
                               projs_matrix[int(index - block * sigLength)])

        return projections


class LODico(Dico):
    """ Shift invariant MDCT dictionary
        Only difference is in the constructor and initialization: need to use LOBlocks """

    # properties
    HRsizes = []  # the sizes for which High resolution is seeked
    nature = 'LOMDCT'

    # constructor
    def __init__(self, sizes=[], hrsizes=None, useC=True, debug_level=None):
        ''' Basic contructor. By default all the block will have the locally optimized behvior but you
        can specify only a subset with the hrsizes variable '''
        if debug_level is not None:
            _Logger.setLevel(debug_level)

        self.sizes = sizes
        self.use_c_optim = useC
        if hrsizes is not None:
            self.HRsizes = hrsizes
        else:
            self.HRsizes = sizes  # default behavior: compute high resolution for all scales

    def initialize(self, residualSignal):
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for mdctSize in self.sizes:
            # check whether this block should optimize time localization or not
            if mdctSize in self.HRsizes:
                self.blocks.append(block.LOBlock(mdctSize,
                                                 residualSignal, useC=self.use_c_optim))
            else:
                self.blocks.append(
                    block.Block(mdctSize, residualSignal, useC=self.use_c_optim))

    def get_projections(self, indexes, sigLength):
        """ additional method provided for Gradient Pursuits
            indexes formalism: key as in the py_pursuit_Approx Object :
            int(block*self.length +  frame*float(atom.length /2) + atom.frequencyBin)"""

        projections = []
        for index in indexes:
            block = int(math.floor(index / sigLength))
# frame = math.floor( (index - block*sigLength)   /  (self.sizes[block]  /2))
            value = self.blocks[block].projs_matrix[int(
                index - block * sigLength)]
            if value.real < 0:
                projections.append(-abs(value))
            else:
                projections.append(abs(value))

        return projections


class FullDico(Dico):
    """ This class handles blocks were MDCT products are computed for
        each and every possible time localization. Therefore Iterations
        are very slow but convergence should be optimum """
        # constructor
    nature = 'FullMDCT'

    def __init__(self, sizes=[]):
        self.sizes = sizes

    def initialize(self, residualSignal):
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for mdctSize in self.sizes:
            self.blocks.append(block.FullBlock(mdctSize, residualSignal))

    def init_proj_matrix(self, itNumbers):
        """ method used for monitoring the projection along iterations"""
        projShape = zeros(len(self.sizes))
        nbprojs = zeros(len(self.sizes))
        for bI in range(len(self.sizes)):
            projShape[bI] = (self.blocks[bI].projs_matrix[0].shape[0])
            nbprojs[bI] = (len(self.blocks[bI].projs_matrix))
            print bI, projShape[bI], nbprojs[bI]

        return zeros((sum(projShape * nbprojs), itNumbers))

    def update_proj_matrix(self, projMatrix, iterationNumber, normedProj=False):
        """ method used for monitoring the projection along iterations"""
        ide = 0
        for blockIdx in range(len(self.blocks)):
            projMat = array(
                self.blocks[blockIdx].projs_matrix.values()).flatten()
#            if normedProj:
#                normCoeff = sqrt(sum(projMat**2))
#            else:
#                normCoeff = 1
# print projMat.shape , projMatrix[id:id+projMat.shape[0],
# iterationNumber].shape
            projMatrix[ide:ide + projMat.shape[0],
                       iterationNumber] = projMat  # /normCoeff
            ide += projMat.shape[0]

'''class pymp_SpreadDico(py_pursuit_MDCTDico):
     UNDER DEVELOPPMENT DO NOT USE
    def __init__(self , sizes=[] ,type = 'SpreadMDCT' ,debugLevel=None , useC = True,
                 allBases = True , Spreadbases = [],penalty=0.5, maskSize = 2):
        if debugLevel is not None:
            _Logger.setLevel(debugLevel)

        self.useC = useC
        self.sizes = sizes
        self.type = type
        self.penalty = penalty
        self.maskSize =maskSize
        if allBases:
            self.spreadScales = self.sizes
        else:
            self.spreadScales = Spreadbases

        _Logger.info('New dictionary created with sizes : ' +str(self.sizes))

    def initialize(self , residualSignal):
        """ Initialize As many blocks as number of signals x number of window sizes
        """
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for mdctSize in self.sizes:
            if mdctSize in self.spreadScales:
                self.blocks.append(Block.py_pursuit_SpreadBlock(
                    mdctSize , residualSignal, useC=self.useC))
            else:
                self.blocks.append(Block.py_pursuit_MDCTBlock(
                    mdctSize , residualSignal, useC=self.useC))
    '''


def fromXml(xmlNode):
    ''' Export routine NOT FULLY TESTED '''
    if not isinstance(xmlNode, Element):
        raise TypeError('Xml element not provided')

    # retrieve sizes
    for e in xmlNode.childNodes:
        if e.localName == 'Sizes':
            sizesNode = e
            break

#    sizesNode = xmlNode.childNodes[0]
    sizes = []
    for node in sizesNode.childNodes:
        if node.localName == 'Size':
            sizes.append(int(node.getAttribute('scale')))

    if xmlNode.getAttribute('class') == 'Dico':
        return Dico(sizes)

    elif xmlNode.getAttribute('class') == 'LODico':
        return LODico(sizes)

    elif xmlNode.getAttribute('class') == 'FullDico':
        return FullDico(sizes)
