#
#
#                       Classes.mdct.random.pymp_RandomDicos
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
Module pymp_RandomDicos
=======================

This file handle dctionaries that are used in Randomized Matching Pursuits
see [1] for details.

[1] M. Moussallam, L. Daudet, et G. Richard,
"Matching Pursuits with Random Sequential Subdictionaries"
Signal Processing, vol. 92, pp. 2532-2544 2012.

"""

import math
from numpy import abs

from ..dico import Dico
from ... import log
from . import block


global _Logger
_Logger = log.Log('RandomMDCTDico', level=0)


class RandomDico(Dico):
    """ This dictionary implements a sequence of subdictionaries that are shifted in time at each iteration in a pre-defined manner
        the shifts are controlled by the different blocks.

        Attributes:

            `randomType`: The type of time-shift sequence, available choices are *scale*,*random*,*gaussian*,*binom*,*dicho*,*jump*,*binary* default is *random* which use a uniform pseudo-random generator

            `nbSim`: Number of consecutive iterations with the same time-shift (default is 1)

    """

    # properties
    randomType = 'none'  # type of sequence , Scale , Random or Dicho
    iterationNumber = 0  # memorizes the position in the sequence
    nbSim = 1
    # number of consecutive similar position
    nature = 'RandomMDCT'

    # constructor
    def __init__(self, sizes=[], randomType='random', nbSame=1, windowType=None):
        self.randomType = randomType
        self.sizes = sizes
        self.nbSim = nbSame

        self.windowType = windowType

    def initialize(self, residualSignal):
        self.blocks = []
        self.bestCurrentBlock = None
        self.startingTouchedIndex = 0
        self.endingTouchedIndex = -1

        for mdctSize in self.sizes:
            # check whether this block should optimize time localization or not
            self.blocks.append(block.RandomBlock(mdctSize, residualSignal, randomType=self.
                randomType, nbSim=self.nbSim, windowType=self.windowType))

    def computeTouchZone(self, previousBestAtom):
        # if the current time shift is about to change: need to recompute all
        # the scores
        if (self.nbSim > 0):
            if ((self.iterationNumber + 1) % self.nbSim == 0):
                self.startingTouchedIndex = 0
                self.endingTouchedIndex = -1
            else:
                self.startingTouchedIndex = previousBestAtom.timePosition - previousBestAtom.length / 2
                self.endingTouchedIndex = self.startingTouchedIndex + 1.5 * previousBestAtom.length
        else:
            self.startingTouchedIndex = previousBestAtom.timePosition - previousBestAtom.length / 2
            self.endingTouchedIndex = self.startingTouchedIndex + 1.5 * previousBestAtom.length

    def update(self, residualSignal, iterationNumber=0, debug=0):
        self.maxBlockScore = 0
        self.bestCurrentBlock = None
        self.iterationNumber = iterationNumber
        # BUGFIX STABILITY
#        self.endingTouchedIndex = -1
#        self.startingTouchedIndex = 0

        for block in self.blocks:
            startingTouchedFrame = int(
                math.floor(self.startingTouchedIndex / (block.scale / 2)))
            if self.endingTouchedIndex > 0:
                endingTouchedFrame = int(math.floor(self.
                    endingTouchedIndex / (block.scale / 2))) + 1
                # TODO check this
            else:
                endingTouchedFrame = -1

            block.update(residualSignal,
                 startingTouchedFrame, endingTouchedFrame, iterationNumber)

            if abs(block.maxValue) > self.maxBlockScore:
#                self.maxBlockScore = block.getMaximum()
                self.maxBlockScore = abs(block.maxValue)
                self.bestCurrentBlock = block

    def getSequences(self, length):
        sequences = []
        for block in self.blocks:
            sequences.append(block.TSsequence[0:length])
        return sequences
