#*****************************************************************************/
#                                                                            */
#                               Approx.py                                    */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
#                                                                            */
#                                                                            */
#  This program is free software you can redistribute it and/or             */
#  modify it under the terms of the GNU General Public License               */
#  as published by the Free Software Foundation either version 2            */
#  of the License, or (at your option) any later version.                    */
#                                                                            */
#  This program is distributed in the hope that it will be useful,           */
#  but WITHOUT ANY WARRANTY without even the implied warranty of            */
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
#  GNU General Public License for more details.                              */
#                                                                            */
#  You should have received a copy of the GNU General Public License         */
#  along with this program if not, write to the Free Software               */
#  Foundation, Inc., 59 Temple Place - Suite 330,                            */
#  Boston, MA  02111-1307, USA.                                              */
#                                                                            */
#*****************************************************************************

'''
Module approx
==================

The main class is :class:`approx`

'''
import math
import numpy as np

import matplotlib.patches as mpatches
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.colors as cc

from PyMP import signals
from PyMP import log
from PyMP.base import BaseAtom
from PyMP.tools.mdct import imdct


global _Logger
_Logger = log.Log('Approx')


class Approx:
    """ The approx class encapsulate the approximation that is iteratively being constructed by a greed algorithm

    This object handles MP constructed approximations
    The approx object is quite similarr in nature to the MPTK [1] Book object but allows
    further manipulations in python such as plotting the time frequency distribution of atoms

    The object has following attributes:
    Attributes:

        * `dico`     : the dictionary (as a :class:`.BaseDico` object) from which it has been constructed

        * `atoms`    : a list of :class:`.Atom` objets

        * `atom_number`    : the length of the atoms list

        * `srr`           : the Signal to Residual Ratio achieved by the approximation

        * `original_signal`    : a :class:`.Signal` object that is the original signal

        * `length`        : Length in samples of the time signal, same as the one of `original_signal`

        * `recomposed_signal`   : a :class:`.Signal` objet for the reconstructed signal (as the weighted sum of atoms specified in the atoms list)


    An approximant can be manipulated in various ways. Obviously atoms can be edited by several methods among which
    :func:`add` ,  :func:`remove`  and :func:`filterAtom`

    Measure of distorsion can be estimated by the method :func:`compute_srr`

    Approx objets can be exported in various formats using
    :func:`to_dico` ,
    :func:`to_sparse_array` ,
    :func:`to_array` ,
    :func:`write_to_xml` ,
    :func:`dumpToDisk` ,

    and reversely can be recovered from these formats

    A useful plotting routine, :func:`plot_tf` is provided to visualize atom distribution in the time frequency plane
    Also an experimental 3D plot taking the atom iteration number as a depth parameter
    :func:`plot_3d`


    """

    dico = []
    atoms = []
    atom_number = 0
    srr = 0
    original_signal = []
    frame_num = 0
    frame_len = 0
    length = 0
    recomposed_signal = None
    fs = 0

    def __init__(self, dico=None, atoms=[], originalSignal=None, length=0, Fs=0, debug_level=None):
        """ The approx class encapsulate the approximation that is iteratively being constructed by a greed algorithm """
        if debug_level is not None:
            _Logger.setLevel(debug_level)

        self.dico = dico
        self.atoms = atoms
        self.original_signal = originalSignal
        if originalSignal != None:
            self.length = originalSignal.length
            self.fs = originalSignal.fs
            isNormalized = originalSignal.is_normalized
        else:
            self.length = length
            self.fs = Fs
            isNormalized = False
        if dico != None:
            self.frame_num = self.length / max(self.dico.sizes)
        if self.frame_num > 0:
            self.frame_len = self.length / self.frame_num

        self.recomposed_signal = signals.Signal(
            np.zeros(self.length), self.fs)
        self.recomposed_signal.is_normalized = isNormalized

    

    def synthesize(self, method=0, forceReSynthesis=True):
        """ function that will synthesise the approximant using the list of atoms
            this is mostly DEPRECATED"""
        if self.original_signal == None:
            _Logger.warning("No original Signal provided")
#            return None

        if (self.recomposed_signal == None) | forceReSynthesis:
            synthesizedSignal = np.zeros(self.length)

            if len(self.atoms) == 0:
                _Logger.info("No Atoms")
                return None

            # first method by inverse MDCT
            if method == 0:
                for mdctSize in self.dico.sizes:
                    mdctVec = np.zeros(self.length)
                    for atom in self.atoms:
                        if atom.length == mdctSize:
                            # bugFIx
                            n = atom.timePosition + 1
                            frame = math.floor(
                                 float(n) / float(atom.length / 2)) + 1
# mdctVec[frame*float(atom.length /2) + atom.frequencyBin] += atom.amplitude
                            mdctVec[frame * float(atom.length /
                                2) + atom.frequencyBin] += atom.mdct_value
                    synthesizedSignal += imdct(mdctVec, mdctSize)
# synthesizedSignal += concatenate((zeros(mdctSize/4) , imdct(mdctVec ,
# mdctSize)) )[1:-mdctSize/4+1]

            # second method by recursive atom synthesis - NOT WORKING
            elif method == 1:
                for atom in self.atoms:
                    atom.synthesizeIFFT()
                    synthesizedSignal[atom.timePosition:
                         atom.timePosition + atom.length] += atom.waveform

            # HACK here to resynthesize using LOMP atoms
            elif method == 2:
                for atom in self.atoms:
                    atom.waveForm = atom.synthesizeIFFT()
                    if (atom.projectionScore is not None):
                        if (atom.projectionScore < 0):
                            atom.waveform = (-math.sqrt(-atom.projectionScore /
                                sum(atom.waveform ** 2))) * atom.waveform
                        else:
                            atom.waveform = (math.sqrt(atom.projectionScore /
                                sum(atom.waveform ** 2))) * atom.waveform

                    synthesizedSignal[atom.timePosition:
                         atom.timePosition + atom.length] += atom.waveform

            self.recomposed_signal = signals.Signal(synthesizedSignal, self.fs)
            #return self.recomposed_signal
        # other case: we just give the existing synthesized Signal.
        return self.recomposed_signal


    def __getitem__(self, item):
        ''' Get the waveform built by only the subAtomNumber first atoms 
            Outputs:
            
                *  a py_pursuit_Signal array '''
        if isinstance(item, slice):
            start, stop, step = item.start, item.stop , item.step
        elif isinstance(item, int):
            start = item
            stop = item
            
        else:
            raise TypeError("not recognized")
    
        if step is None:
            step=1
        output_approx = Approx(self.dico, [], self.original_signal, self.length, self.fs)
        if stop > self.atom_number:
            raise ValueError('Dude you asked fore more than I can give you..')
        
        for atomIdx in range(start,stop,step):
            output_approx.add(self.atoms[atomIdx])
        
        return output_approx

    # Filter the atom list by the given criterion
    def filter_atoms(self, mdctSize=0, posInterv=None, freqInterv=None):
        '''Filter the atom list by the given criterion, returns an new approximant object'''
        filteredApprox = Approx(self.dico, [], self.original_signal,
             self.length, self.fs)
        doAppend = True
        for atom in self.atoms:
#            if atom.length == mdctSize:
##                print  atom.length , " appended "
#                filteredApprox.add(atom)

            if atom.length == mdctSize:
                doAppend &= True
            else:
                doAppend &= False

            if(posInterv != None):
                if (min(posInterv) < atom.timePosition <= max(posInterv)):
                    doAppend &= True
                else:
                    doAppend &= False

            if(freqInterv != None):
                if (min(freqInterv) < atom.reducedFrequency <= max(freqInterv)):
                    doAppend &= True
                else:
                    doAppend &= False
            if doAppend:
#                print  atom.length , " appended "
                filteredApprox.add(atom)
#

        return filteredApprox

    # this function adds an atom to the collection and updates the internal
    # signal approximant
    def add(self, newAtom, clean=False, noWf=False):
        '''this function adds an atom to the collection and updates the internal signal approximant'''
        if not isinstance(newAtom, BaseAtom):
            raise TypeError("add need a py_pursuit_Atom as parameter ")

        self.atoms.append(newAtom)
        self.atom_number += 1
        # add atom waveform to the internal signal if exists
        if not noWf:
            if self.recomposed_signal != None:
                self.recomposed_signal.add(newAtom)
            else:
                self.synthesize(0, True)
    #            print "approx resynthesized"

            if clean:
                del newAtom.waveform

    #
    
    
    def remove(self, atom):
        ''' We need a routine to remove an atom , by default the last atom is removed '''
        if not isinstance(atom, BaseAtom):
            raise TypeError("add need a py_pursuit_Atom as parameter ")
        self.atoms.remove(atom)
        self.atom_number -= 1

        if self.recomposed_signal != None:
            self.recomposed_signal.subtract(atom, preventEnergyIncrease=False)

    # routine to compute approximation Signal To Residual ratio so far
    def compute_srr(self, residual=None):
        ''' routine to compute approximation Signal To Residual ratio so far using:

            .. math::srr = 10 \log_10 \frac{\| \tilde{x} \|^2}{\| \tilde{x} - x \|^2}

            where :math:`\tilde{x}` is the reconstructed signal and :math:`x` the original
        '''
        if not isinstance(self.recomposed_signal, signals.Signal):
            return np.NINF

#        recomposedEnergy = sum((self.recomposed_signal.dataVec)**2)
        recomposedEnergy = self.recomposed_signal.energy

        if recomposedEnergy <= 0:
            return np.NINF

        if residual is None:            
            resEnergy = sum((self.original_signal.data - 
                 self.recomposed_signal.data) ** 2)
        else:
            resEnergy = residual.energy
# resEnergy = sum((self.original_signal.dataVec -
# self.recomposed_signal.dataVec)**2)
        if resEnergy == 0:
            return np.PINF

        return 10 * math.log10(recomposedEnergy / resEnergy)

    # ploting routine 2d in a time-frequency plane
    def plot_tf(self, labelX=True, labelY=True, fontsize=12, ylim=None, patchColor=None, labelColor=None,
               multicolor=False, axes=None, maxAtoms=None, recenter=None, keepValues=False,
               french=False, Alpha=False, logF=False):
        """ A time Frequency plot routine using Matplotlib

            each atom of the approx is plotted in a time-frequency plane with a gray scale for amplitudes
            Many options are available:

            - labelX     : whether or not to add the Time label on the X axis

            - labelY     : whether or not to add the Frequency label on the Y axis

            - fontsize    : the label fontSize

            - ylim        : the frequency range of the plot

            - patchColor    : specify a single color for all atoms

            - maxAtoms    : specify a maximum number of atom to take into account. Usually atoms are ordered by decreasing amplitude due to MP rules, meaning only the most salient atoms will be plotted

            - recenter    : a couple of values to offset the time and frequency localization of atoms

            - keepValues    : use atom Amplitudes for atom colorations

            - french    : all labels in french

            - Alpha     : use transparency

            - logF        : frequency axis in logarithmic scale

        """
#        plt.figure()
        if french:
            labx = "Temps (s)"
            laby = "Frequence (Hz)"
        else:
            labx = "Time (s)"
            laby = "Frequency (Hz)"

        Fs = self.fs
        # first loop to recover max value
        maxValue = 0
        maxFreq = 1.0
        minFreq = 22000.0
        if not keepValues:
            valueArray = np.array(
                [math.log10(abs(atom.getAmplitude())) for atom in self.atoms])
        else:
            valueArray = np.array(
                [abs(atom.getAmplitude()) for atom in self.atoms])
        if multicolor:
            cmap = cc.LinearSegmentedColormap('jet', abs(valueArray))
            for atom in self.atoms:
                if abs(atom.mdct_value) > maxValue:
                    maxValue = abs(atom.mdct_value)

        # normalize to 0 -> 1
#        if min(valueArray) < 0:
        if len(valueArray) > 0 and not keepValues:
            valueArray = valueArray - min(valueArray)
            valueArray = valueArray / max(valueArray)

        # Get target axes
        if axes is None:
            axes = plt.gca()

        # normalize the colors
#        if multicolor:
#            normedValues = cc.Normalize(valueArray)

        if maxAtoms is not None:
            maxTom = min(maxAtoms, len(self.atoms))
        else:
            maxTom = len(self.atoms)

        # Deal with static offsets
        if recenter is not None:
            yOffset = recenter[0]
            xOffset = recenter[1]
        else:
            yOffset, xOffset = 0, 0

#        print yOffset,xOffset

        if Alpha:
            alphaCoeff = 0.6
        else:
            alphaCoeff = 1
        patches = []
        colors = []
        for i in range(maxTom):
            atom = self.atoms[i]
            L = atom.length
            K = L / 2
            freq = atom.frequencyBin
            f = float(freq) * float(Fs) / float(L)
            bw = (float(Fs) / float(K))  # / 2
            p = float(atom.timePosition + K) / float(Fs)
            l = float(L - K / 2) / float(Fs)
#            print "f =  " ,f , " Hz"
            if f > maxFreq:
                maxFreq = f + bw + bw
            if f < minFreq:
                minFreq = f - bw - 10 * bw

#            xy = p, f-bw,
            xy = p - xOffset, f - yOffset,
            width, height = l, bw
#            xy = p - xOffset, log10(f) - yOffset,

            if logF:
                xy = p - xOffset, (12 * np.log2(f - yOffset))

                height = 1.0  # BUGFIX

#            print xy , f
            colorvalue = math.sqrt(valueArray[i])

            if multicolor:
                patchtuplecolor = (math.sqrt(colorvalue),
                                   math.exp(
                                       -((colorvalue - 0.35) ** 2) / (0.15)),
                                   1 - math.sqrt(colorvalue))
                colors.append(colorvalue)
            else:
                if patchColor is None:
                    patchtuplecolor = (1 - math.sqrt(colorvalue),
                         1 - math.sqrt(colorvalue), 1 - math.sqrt(colorvalue))
                    colors.append(colorvalue)
                else:
                    patchtuplecolor = patchColor
                    colors.append(patchColor)
#            patchtuplecolor = (colorvalue, colorvalue, colorvalue)
# patch = mpatches.Rectangle(xy, width, height, facecolor=patchtuplecolor,
# edgecolor=patchtuplecolor , zorder=colorvalue )
#            xy = p + L/2 , f-bw,
            patch = mpatches.Ellipse(xy, width, height, facecolor=patchtuplecolor,
                 edgecolor=patchtuplecolor, zorder=colorvalue)
            if keepValues:
                patches.append(patch)
            else:
                axes.add_patch(patch)

        if keepValues:

            # first sort the patches by values
            sortedIndexes = valueArray.argsort()
            PatchArray = np.array(patches)
            p = PatchCollection(PatchArray[sortedIndexes].tolist(), linewidths=0., cmap=matplotlib.cm.copper_r, match_original=False, alpha=alphaCoeff)
    #        print array(colors).shape
            p.set_array(valueArray[sortedIndexes])

            axes.add_collection(p)

        if labelColor is None:
            labelColor = 'k'

        # finalize graphic options
        axes.set_xlim(0 - xOffset, (self.length) / float(Fs) - xOffset)
        if ylim is None:
            axes.set_ylim(max(0, minFreq) - yOffset, maxFreq - yOffset)
        else:
            axes.set_ylim(ylim)
        if labelY:
            axes.set_ylabel(laby, fontsize=fontsize, color=labelColor)
#            plt.ylabel.set_color(labelColor)
        else:
            plt.setp(axes, 'yticklabels', [])

        if labelX:
            axes.set_xlabel(labx, fontsize=fontsize)
        else:
            plt.setp(axes, 'xticklabels', [])
        if keepValues:
            plt.colorbar(p)

    def plot_3d(self, itStep, fig=None):
        """ Creates a 3D Time-Frequency plot with various steps NOT WORKING below 0.99
        EXPERIMENTAL"""
        from mpl_toolkits.mplot3d import Axes3D
        if fig is None:
            fig = plt.figure()

        # beware of compatibility issues here
        ax = Axes3D(fig)
        #for atom, idx in zip(self.atoms, range(self.atom_number)):
        ts = [atom.timePosition / float(self.fs)
             for atom in self.atoms]
        fs = [atom.reducedFrequency * self.
            fs for atom in self.atoms]
        zs = range(self.atom_number)
# ss = [float(atom.length)/float(self.fs) for atom in
# self.atoms]

        # More complicated but not working
#        ts = []
#        fs = []
#        zs=  []
#        Fs = float(self.fs)
#        for atom,idx in zip(self.atoms, range(self.atom_number)):
#            L = atom.length
#            K = L/2
#
#            disp = log2(L)
#
#            freq = atom.frequencyBin
#            f = freq*Fs/float(L)
#            bw = ( Fs / float(K) ) / 2
#
#            p  = float(atom.timePosition + K) / Fs
#            l  = float(L - K/2) / Fs
#
#            # hdispersion is disp - 7
#            hdisp = int(math.floor(max(disp - 7,1)))
#            vdisp = int(math.floor(max(15/(disp),1)))
#            # vdispersion is 15/(disp - 7)
#            for hdispIdx in range(hdisp):
#                for vdispIdx in range(vdisp):
#                    ts.append(p+10*float(hdispIdx)/Fs)
#                    fs.append(f+vdispIdx)
#                    zs.append(idx)

        ax.scatter(ts, zs, fs, color="k", s=10, marker='o')
        ax.set_xlabel('Temps (s)')
        ax.set_ylabel('Iteration ')
        ax.set_zlabel('Frequence (Hz) ')

#        allAtoms = len(self.atoms)
#        nbplots = allAtoms/itStep
#        if nbplots >10:
#            raise ValueError('Seriously? WTF dude?')
#
#        for plotIdx in range(nbplots):
#            subAx = Axes3D(fig)
#            self.plot_tf(axes=subAx, maxAtoms=plotIdx*itStep)

#    def dumpToDisk(self, dumpFileName, dumpFilePath='./'):
#        import cPickle
#
#        output = open(dumpFilePath + dumpFileName, 'wb')
#        _Logger.info("Dumping approx to  " + dumpFilePath + dumpFileName)
#        cPickle.dump(self, output, protocol=0)
#        _Logger.info("Done ")

#        plt.show()
    def write_to_xml(self, fname, output_path="./"):
        """ Write the atoms using an XML formalism to the designated output file """

        #from xml.dom.minidom import Document
        import xml.dom
                # creates the xml document
        doc = xml.dom.minidom.Document()

        # populate root node information
        _Logger.info("Retrieving Root node attributes")
        ApproxNode = doc.createElement("Approx")
        ApproxNode.setAttribute('length', str(self.length))
        ApproxNode.setAttribute(
            'originalLocation', self.original_signal.location)
        ApproxNode.setAttribute('frame_num', str(self.frame_num))
        ApproxNode.setAttribute('frame_len', str(self.frame_len))
        ApproxNode.setAttribute(
            'Fs', str(self.original_signal.fs))
        # add dictionary node
        ApproxNode.appendChild(self.dico.toXml(doc))

        # now add as many nodes as atoms
        AtomsNode = doc.createElement("Atoms")
        AtomsNode.setAttribute('number', str(self.atom_number))
        _Logger.info(
            "Getting Child info for " + str(self.atom_number) + " existing atoms")
        for atom in self.atoms:
            AtomsNode.appendChild(atom.toXml(doc))

        ApproxNode.appendChild(AtomsNode)
        doc.appendChild(ApproxNode)

        # check output file existence
#        if not os.path.exists(output_path + fname):
#            os.create

        # flush to external file

#        xml.dom.ext.PrettyPrint(doc, open(output_path + fname, "w"))
        _Logger.info("Written Xml to : " + output_path + fname)
        return doc

    def to_dico(self):
        """ Returns the approximant as a sparse dictionary object ,
        key is the index of the atom and values are atom objects"""
        dico = {}

        for atom in self.atoms:
            block = [i for i in range(
                len(self.dico.sizes)) if self.dico.sizes[i] == atom.length][0]
            n = atom.timePosition + 1
            frame = math.floor(float(n) / float(atom.length / 2)) + 1
# sparseArray[int(block*self.length +  frame*float(atom.length /2) +
# atom.frequencyBin)] = (atom.mdct_value , frame*float(atom.length /2) -
# atom.timePosition)
# dico[int(block*self.length +  frame*float(atom.length /2) +
# atom.frequencyBin)] = atom
            key = int(block * self.length + frame * float(atom.
                length / 2) + atom.frequencyBin)
            if key in dico:
                dico[key].append(atom)
            else:
                dico[key] = [atom]

        return dico

    def to_sparse_array(self):
        """ Returns the approximant as a sparse dictionary object , key is the index of the atom and value is its amplitude"""
        sparseArray = {}
        # quickly creates a dictionary for block numbering
        blockIndexes = {}
        for i in range(len(self.dico.sizes)):
            blockIndexes[self.dico.sizes[i]] = i

        for atom in self.atoms:
            block = blockIndexes[atom.length]
            n = atom.timePosition + 1
            frame = math.floor(float(n) / float(atom.length / 2)) + 1
# sparseArray[int(block*self.length +  frame*float(atom.length /2) +
# atom.frequencyBin)] = (atom.mdct_value , frame*float(atom.length /2) -
# atom.timePosition)
            sparseArray[int(block * self.length + frame * float(atom.length / 2)
                 + atom.frequencyBin)] = (atom.mdct_value, atom.timePosition)
        return sparseArray

#    def toSparseMatrix(self):
# """ Returns the approximant as a sparse dictionary object , key is the index
# of the atom and value is its amplitude"""
#        sparseArray = {}
#        # quickly creates a dictionary for block numbering
#        blockIndexes = {}
#        for i in range(len(self.dico.sizes)):
#            blockIndexes[self.dico.sizes[i]] = i
#
#        for atom in self.atoms:
#            block = blockIndexes[atom.length]
#            n = atom.timePosition +1
#            frame = math.floor( float(n) / float(atom.length /2) ) +1
# sparseArray[int(block*self.length +  frame*float(atom.length /2) +
# atom.frequencyBin)] = (atom.mdct_value , frame*float(atom.length /2) -
# atom.timePosition)
# sparseArray[int(block*self.length +  frame*float(atom.length /2) +
# atom.frequencyBin)] = (atom.mdct_value , atom.timePosition)
#        return sparseArray

    def to_array(self):
        """ Returns the approximant as an array object , key is the index of the atom and value is its amplitude"""
        sparseArray = np.zeros(len(self.dico.sizes) * self.length)
        timeShifts = np.zeros(len(self.dico.sizes) * self.length)
        # quickly creates a dictionary for block numbering
        blockIndexes = {}
        for i in range(len(self.dico.sizes)):
            blockIndexes[self.dico.sizes[i]] = i

        for atom in self.atoms:
            block = blockIndexes[atom.length]
            n = atom.timePosition + 1
            frame = math.floor(float(n) / float(atom.length / 2)) + 1
            sparseArray[int(block * self.length + frame * float(
                atom.length / 2) + atom.frequencyBin)] = atom.mdct_value
            timeShifts[int(block * self.length + frame * float(atom.length / 2) + atom.frequencyBin)] = frame * float(atom.length / 2) - atom.timePosition - atom.length / 4
        return sparseArray, timeShifts

#    def __del__(self):
#        """ Full Cleaning is sometimes useful """
#        for atom in self.atoms:
#            del atom.waveform
#            del atom
#        del self.recomposed_signal.dataVec


def load_from_disk(inputFilePath):
    import cPickle
    return cPickle.load(inputFilePath)


def read_from_xml(InputXmlFilePath, xmlDoc=None, buildSignal=True):
    """ reads an xml document and create the corresponding Approx object
    WARNING this method is only designed for MDCT pursuits for now"""

    from xml.dom.minidom import Document
    import xml.dom
    from .mdct import dico, atom
    # check if reading from file is needed
    if xmlDoc == None:
        xmlDoc = xml.dom.minidom.parse(InputXmlFilePath)

    if not isinstance(xmlDoc, Document):
        raise TypeError('Xml document is wrong')

    # retrieve the main approx node
    ApproxNode = xmlDoc.getElementsByTagName('Approx')[0]

    # retrieve the dictionary
    Dico = dico.fromXml(xmlDoc.getElementsByTagName('Dictionary')[0])

    # retrieve atom list
    AtomsNode = xmlDoc.getElementsByTagName('Atoms')[0]
    atoms = []
    approx = Approx(Dico, atoms, None, int(ApproxNode.getAttribute('length')),
                                   int(ApproxNode.getAttribute('Fs')))
    for node in AtomsNode.childNodes:
        if node.localName == 'Atom':
            if buildSignal:
                approx.add(atom.fromXml(node))
            else:
                approx.atoms.append(atom.fromXml(node))
#            atoms.append(py_pursuit_Atom.fromXml(node))
#    if not buildSignal:
#        approx.atom_number = len(approx.atoms)
    # optional: if load original signal - TODO

# approx =  py_pursuit_Approx(Dico, atoms, None,
# int(ApproxNode.getAttribute('length')),
#                                   int(ApproxNode.getAttribute('Fs')))
    approx.atom_number = len(atoms)
    return approx


def read_from_mat_struct(matl_struct):
    """ retrieve the python object from a saved version in matlab format
    This is useful if you saved a py_pursuit_approx object using the scipy.io.savemat
    routine and you loaded it back using scipy.io.loadmat"""

    # Convert to object struct
    appObject = matl_struct[0, 0]
    dico = appObject.dico[0, 0]
    # TODO proper lecture of dictionary objects

    # So far we only read the parameters

    approx = Approx(None, [], None, appObject.length[0, 0], appObject.
        fs[0, 0])
    return approx


def fusion_approxs(approxCollection, unPad=True):
    """ fusion a collection of frame by frame approximation.
    The collection is assumed to be temporally ordered """

    if not len(approxCollection) > 0:
        raise ValueError('Approx collection appears to be actually empty')

    dico = approxCollection[0].dico
    approxLength = approxCollection[0].length * len(approxCollection)
    if unPad:
        approxLength -= 2 * dico.sizes[-1] * len(approxCollection)
    Fs = approxCollection[0].fs

    approx = Approx(dico, [], None, approxLength, Fs)
    for segIdx in range(len(approxCollection)):
        currentApprox = approxCollection[segIdx]
        offset = segIdx * currentApprox.length
        if unPad:
            offset -= (segIdx + 1) * dico.sizes[-1]

        for atom in currentApprox.atoms:
            atom.timePosition += offset
            approx.atoms.append(atom)

    return approx
