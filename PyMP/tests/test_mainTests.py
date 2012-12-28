"""
Python test file for PyMP engine

testing normal behavior of most MP functions and classes

M.Moussallam
"""

import os
import sys
mainClassesPath = os.path.abspath('..')
sys.path.append(mainClassesPath)


import unittest
from PyMP.tools import mdct
import matplotlib.pyplot as plt
from numpy import random, zeros
import time

import cProfile
import math
global _Logger

#from Classes.gabor import *
from PyMP.mdct import atom as mdct_atom
from PyMP.mdct import dico as mdct_dico
from PyMP.mdct import block as mdct_block
from PyMP.base import BaseAtom, BaseDico
from PyMP.mdct.random import dico as random_dico
from PyMP import approx
from PyMP import signals
from PyMP import log
from PyMP import mp

audioFilePath = '../../data/'


class AtomTest(unittest.TestCase):
    def setUp(self):
        pass

    def runTest(self):
        # empty creation
        pyAtom = BaseAtom()
        self.assertEqual(pyAtom.length, 0)
        self.assertEqual(pyAtom.amplitude, 0)
        self.assertEqual(pyAtom.nature, 'Abstract')

        del pyAtom

        # full creation
        pyAtom2 = mdct_atom.Atom(1024, 1, 12432, 128, 44100, 0.57)
        self.assertEqual(pyAtom2.length, 1024)
        self.assertEqual(pyAtom2.amplitude, 1)
        self.assertEqual(pyAtom2.mdct_value, 0.57)
        self.assertEqual(pyAtom2.samplingFrequency, 44100)
        self.assertEqual(
            pyAtom2.reducedFrequency, float(128 + 0.5) / float(1024))
        self.assertEqual(pyAtom2.timePosition, 12432)
        self.assertEqual(pyAtom2.nature, 'MDCT')

#        synthesizedAtom2 = pyAtom2.synthesize()

        synthAtom3 = pyAtom2.synthesizeIFFT()

#        energy1 = sum(synthesizedAtom2.waveform**2)
        energy2 = sum(synthAtom3.real ** 2)

        print energy2

#        plt.plot(synthesizedAtom2.real)
        plt.plot(synthAtom3.real)
        del pyAtom2

        print " testing LOmp atoms synthesis "
        mdctValue = 0.57
        timeShift = 144
        projectionScore = -0.59
        pyAtomLOmp = mdct_atom.Atom(1024, 1, 12432, 128, 44100, 0.57)
        pyAtomLOmp.timeShift = timeShift
        pyAtomLOmp.projectionScore = projectionScore

        # test 1 synthesis
        pyAtomLOmp.synthesizeIFFT()
        wf1 = pyAtomLOmp.waveform.copy()

        wf2 = -(math.sqrt(abs(projectionScore) / sum(wf1 ** 2))) * wf1

        mdctVec = zeros(3 * 1024)
        mdctVec[1024 + 128] = projectionScore
        wf3 = mdct.imdct(mdctVec, 1024)[0.75 * 1024: 1.75 * 1024]

        plt.figure()
        plt.plot(wf1)
        plt.plot(wf2)
        plt.plot(wf3)
        plt.legend(('1', '2', '3'))
#        plt.show()

    def tearDown(self):
        pass


class DicoTest(unittest.TestCase):
    def runTest(self):
        # test dictionary class
        pyDico = BaseDico()
        self.assertEqual(pyDico.nature, 'Abstract')

        del pyDico

        # test dictionary class
        pyDico = mdct_dico.Dico(
            [2 ** l for l in range(7, 15, 1)], debugLevel=3)
        self.assertEqual(pyDico.sizes, [128, 256, 512, 1024,
             2048, 4096, 8192, 16384])
        self.assertEqual(pyDico.nature, 'MDCT')

        del pyDico


class Signaltest(unittest.TestCase):
    def runTest(self):

        pySig = signals.Signal(debugLevel=3)
        self.assertEqual(pySig.length, 0)
        self.assertEqual(len(pySig.data), 0)

        del pySig

        pySig = signals.Signal(audioFilePath + "ClocheB.wav")
        self.assertNotEqual(pySig.length, 0)
        self.assertNotEqual(pySig.data, [])
        self.assertEqual(pySig.channelNumber, 2)
        self.assertEqual(pySig.location, audioFilePath + "ClocheB.wav")
        self.assertEqual(pySig.samplingFrequency, 8000)

        # Last test, the wigner ville plot

        pySig.crop(8000, 8256)

        pySig.wigner_plot()

        del pySig

        pySig = signals.Signal(
            audioFilePath + "ClocheB.wav", doNormalize=True, forceMono=True)
        data1 = pySig.data.copy()
        # try adding and subtracting an atom
        pyAtom = mdct_atom.Atom(1024, 0.57, 4500, 128, 8000, 0.57)
        pySig.add(pyAtom)

        data2 = pySig.data.copy()
        
     
        self.assertNotEqual(sum((data1 - data2) ** 2), 0)
        pySig.subtract(pyAtom)

#        plt.plot(data1)
#        plt.plot(data2)
#        plt.plot(pySig.data)
#        plt.legend(("original", "added" , "subtracted"))
#
        self.assertAlmostEqual(sum((pySig.data - data1) ** 2), 0)

        # test on a long signals
        L = 4 * 16384
        longSignal = signals.LongSignal(
            audioFilePath + "Bach_prelude_40s.wav", L)
        # suppose we want to retrieve the middle of the signals , namely from
        # frame 5  to 12
        startSeg = 2
        segNumber = 3
        shortSignal = longSignal.getSubSignal(startSeg, segNumber, False)

        # witness signals
        witSignal = signals.Signal(
            audioFilePath + "Bach_prelude_40s.wav", doNormalize=True, forceMono=False)

#        shortSignal.plot()
#        witSignal.plot()
        plt.figure()
        plt.plot(shortSignal.data)
        plt.plot(
            witSignal.data[startSeg * L:startSeg * L + segNumber * L], ':')

#        plt.show()

        # test writing signals
        outputPath = 'subsignal.wav'
        if os.path.exists(outputPath):
            os.remove(outputPath)

        shortSignal.write(outputPath)

        # test long signals with overlap 50 %
        longSignal = signals.LongSignal(
            audioFilePath + "Bach_prelude_40s.wav", L, True, 0.5)
        # suppose we want to retrieve the middle of the signals , namely from
        # frame 5  to 12
        startSeg = 2
        segNumber = 3
        shortSignal = longSignal.getSubSignal(startSeg, segNumber, False)

        # witness signals
        witSignal = signals.Signal(
            audioFilePath + "Bach_prelude_40s.wav", doNormalize=True, forceMono=False)

#        shortSignal.plot()
#        witSignal.plot()
        plt.figure()
        plt.plot(shortSignal.data)
        plt.plot(
            witSignal.data[startSeg * L / 2:startSeg * L / 2 + segNumber * L], ':')

        # Testing the downsampliong utility
        plt.figure()
        plt.subplot(121)
        plt.specgram(shortSignal.data, 256, shortSignal.
            samplingFrequency, noverlap=128)
        self.assertEqual(shortSignal.samplingFrequency, 44100)
        shortSignal.write('normal.wav')
        shortSignal.downsample(8000)
        self.assertEqual(shortSignal.samplingFrequency, 8000)
        plt.subplot(122)
        plt.specgram(shortSignal.data, 256, shortSignal.
            samplingFrequency, noverlap=128)

        shortSignal.write('downsampled.wav')


#        plt.show()
class BlockTest(unittest.TestCase):
    def runTest(self):

        pySigOriginal = signals.Signal(
            audioFilePath + "ClocheB.wav", doNormalize=True, forceMono=True)
        pySigOriginal.crop(0, 5 * 8192)
        pySigOriginal.pad(2048)

        block = mdct_block.Block(
            1024, pySigOriginal, debugLevel=3, useC=True)
        # testing the automated enframing of the data
        self.assertEqual(block.frameLength, 512)
        print block.frameNumber
#        import parallelProjections
#        parallelProjections.initialize_plans(array([1024,]))

        try:
            import fftw3
        except ImportError:
            print " FFTW3 python wrapper not installed, abandonning test"
            return
        block.update(pySigOriginal)

        plt.plot(block.projectionMatrix.flatten(1))
        plt.plot(mdct.mdct(pySigOriginal.data, block.scale))
#        plt.show()

        self.assertAlmostEqual(sum((block.projectionMatrix - mdct.
            mdct(pySigOriginal.data, block.scale)) ** 2), 0)

#        parallelProjections.clean_plans(array([1024,]))
        del pySigOriginal


class py_mpTest(unittest.TestCase):
    def runTest(self):
# pyDico = Dico.Dico([2**l for l in range(7,15,1)] , Atom.transformType.MDCT)

        pyDico = mdct_dico.Dico([256, 2048, 8192])
        pySigOriginal = signals.Signal(
            audioFilePath + "ClocheB.wav", doNormalize=True, forceMono=True)
        pySigOriginal.crop(0, 5 * 16384)

        pySigOriginal.data += 0.01 * random.random(5 * 16384)

#        pySigOriginal.plot()

        # first try with a single-atom signals
        pyAtom = mdct_atom.Atom(2048, 1, 11775, 128, 8000, 0.57)
        pyApprox_oneAtom = approx.Approx(pyDico, [], pySigOriginal)
        pyApprox_oneAtom.add(pyAtom)

#        plt.plot(pyApprox_oneAtom.synthesize(0).data[12287:12287+2048])
#        plt.plot(pyAtom.synthesize(0))
#        plt.legend(("IMDCT","Atom"))
#        plt.show()

        pySignal_oneAtom = signals.Signal(pyApprox_oneAtom.
            synthesize(0).data, pySigOriginal.samplingFrequency, False)

        # add some noise
#        pySignal_oneAtom.data += 0.0001*random.random(5*16384)

#        approximant = mp.mp_proto1(pySignal_oneAtom, pyDico, 10, 10)
##        approximant.plot_tf()
#        plt.plot(pySignal_oneAtom.data)
#        plt.plot(approximant.synthesize(0).data)
#        plt.legend(("original","approximant"))
#        plt.show()
#        del approximant

        # test two atoms
# pyAtom2 = Atom.Atom(16384 , 1, 6*8192-1-4096 , 128 , 8000 ,
# Atom.transformType.MDCT , -0.42)
#        pySignal_oneAtom.add(pyAtom2)
##        pySignal_oneAtom.plot()
#        approximant = mp.mp(pySignal_oneAtom, pyDico, 20, 10 , True, False)[0]
#
#        plt.plot(approximant.synthesize(0).data)
#        plt.title("Reconstituted signals")
#        plt.show()
#        plt.plot(pySignal_oneAtom.data)
#        plt.plot(approximant.synthesize(0).data)
#        plt.legend(("original","approximant"))
#        plt.show()

#        cProfile.run('mp.mp_proto1')

        # second test
        pySigOriginal = signals.Signal(
            audioFilePath + "ClocheB.wav", doNormalize=True, forceMono=True)
#        approximant = mp.mp_proto1(pySigOriginal, pyDico, 10, 10)
#        self.assertTrue( isinstance(approximant , Approx.Approx));

        # last test - decomposition profonde
        pyDico2 = mdct_dico.Dico([128, 256, 512, 1024, 2048,
             4096, 8192, 16384], parallel=False)
        pyDico1 = mdct_dico.Dico([16384])
        # profiling test
        print "Plain"
        cProfile.runctx('mp.mp(pySigOriginal, pyDico1, 20, 1000 ,debug=0 , itClean=True)', globals(), locals())

        cProfile.runctx('mp.mp(pySigOriginal, pyDico2, 20, 1000 ,debug=0 , itClean=True)', globals(), locals())

#        print "Parallel"
# cProfile.runctx('mp.mp(pySigOriginal, pyDico_parallel, 40, 100 ,debug=0)' ,
# globals() , locals())
#
#        print pySigOriginal.length
#        t0 = time.clock()
#        approximant = mp.mp(pySigOriginal, pyDico2, 20, 4000 ,False)[0]
#        t1 = time.clock()
# print "SRR of ", approximant.compute_srr() , " dB achieved in ", t1 - t0 , "
# sec and ", approximant.atomNumber,"iteration with C"
#
#        del approximant , pyDico2
#
#        t2 = time.clock()
# pyDico2 = Dico.LODico([128 , 256 , 512 , 1024 , 2048 , 4096, 8192 , 16384])
# approximant = mp.mp(pySigOriginal, pyDico2, 20, 4000
# ,debug=0,padSignal=True)[0]
#        t3 = time.clock()
# print "SRR of ", approximant.compute_srr() , " dB achieved in ", t3 - t2 , "
# sec with C and ", approximant.atomNumber,"iteration and LOmp"
#
#        del approximant , pyDico2
##

# pyDico1 = Dico.LODico([2**j for j in range(7,15) ] , Atom.transformType.MDCT
# )
# pyDico2 = Dico.LODico([2**j for j in range(7,15) ] , Atom.transformType.MDCT
# )
#
#        t = time.clock()
# approximant_High , decays_high = mp.mp(pySigOriginal, pyDico2, 40, 1000
# ,debug=0 , forceHighFreqs=True , HFitNum = 800)
#        print "elapsed : " , time.clock() - t;
#        t = time.clock()
# approximant , decay = mp.mp(pySigOriginal, pyDico1, 40, 1000
# ,debug=0,padSignal=False)
#        print "elapsed : " , time.clock() - t;
#        approximant_High.recomposedSignal.write('recomposedHF.wav');
#        approximant.recomposedSignal.write('recomposed.wav');
#
# print "SRR of ", approximant.compute_srr() , " dB achieved in ", t3 - t2 , "
# sec without C"
#        plt.figure()
#        plt.subplot(211)
#        approximant.plot_tf()
#        plt.subplot(212)
#        approximant_High.plot_tf()
#
#        plt.figure()
#        plt.plot(decay)
#        plt.plot(decays_high , 'r')
#        plt.legend(('Without HF forcing' , 'With HF forcing'));

#        plt.show()

#        del pySigOriginal


class ApproxTest(unittest.TestCase):

    def runTest(self):
#        self.writeXmlTest()

        # test dictionary class

        pyApprox = approx.Approx(debugLevel=3)
        self.assertEqual(pyApprox.originalSignal, None)
        self.assertEqual(pyApprox.atomNumber, 0)
        self.assertEqual(pyApprox.SRR, 0)
        self.assertEqual(pyApprox.atoms, [])

        del pyApprox

        pyDico = mdct_dico.Dico([2 ** l for l in range(7, 15, 1)])
        pySigOriginal = signals.Signal(
            audioFilePath + "ClocheB.wav", forceMono=True)
        pySigOriginal.crop(0, 5 * max(pyDico.sizes))

        pyApprox = approx.Approx(pyDico, [], pySigOriginal)

        pyAtom = mdct_atom.Atom(1024, 1, 12288 - 256, 128, 44100, 0.57)
        pyApprox.add(pyAtom)
#
#        approxSignal = pyApprox.synthesize(0)
#        approxSignal.plot()
#
#        del approxSignal

        pyApprox.add(
            mdct_atom.Atom(8192, 1, 4096 - 2048, 32, 44100, -0.24))
        approxSignal1 = pyApprox.synthesize(0)

        plt.figure()
        plt.subplot(121)
        pyApprox.plot_tf()
        plt.subplot(122)
        pyApprox.plot_tf(multicolor=True, keepValues=True)
#        approxSignal1.plot()

#        del approxSignal
        approxSignal2 = pyApprox.synthesize(1)
#        approxSignal2.plot()
        plt.figure()
        plt.plot(approxSignal1.data)
        plt.plot(approxSignal2.data)
        plt.plot(approxSignal1.data - approxSignal2.data)
        plt.legend(("MDCT", "AtomSynth", "Diff"))
#        plt.show() #TODO here correct mistakes

        # assert two methods are equivalent
        self.assertAlmostEqual(
            sum((approxSignal1.data - approxSignal2.data) ** 2), 0)

        # testing filtering
        self.assertEqual(
            pyAtom, pyApprox.filter(1024, None, None).atoms[0])
        self.assertEqual(pyAtom, pyApprox.filter(1024, [
            12000, 15000], None).atoms[0])

        print pyApprox.compute_srr()
        # TODO testing du SRR

        #testing the write_to_xml and read_from_xml methods

        del pySigOriginal

    def writeXmlTest(self):
        pySigOriginal = signals.Signal(
            audioFilePath + "ClocheB.wav", doNormalize=True, forceMono=True, debugLevel=0)
        pySigOriginal.crop(0, 1 * 16384)
        pyDico = mdct_dico.Dico([256, 2048, 8192], debugLevel=2)
        # first compute an approximant using mp
        approximant = mp.mp(pySigOriginal, pyDico, 10, 10, debug=2)[0]

        outputXmlPath = "approx_test.xml"
        doc = approximant.write_to_xml(outputXmlPath)

        # Test reading from the xml flow
        newApprox = approx.read_from_xml('', doc)
        self.assertEqual(newApprox.dico.sizes, approximant.dico.sizes)
        self.assertEqual(newApprox.atomNumber, approximant.atomNumber)
        self.assertEqual(newApprox.length, approximant.length)

        # now the hardest test
        newApprox.synthesize()
        newApprox.recomposedSignal.plot()
        self.assertAlmostEquals(sum(approximant.recomposedSignal.
            data - newApprox.recomposedSignal.data), 0)

        # test reading from the xml file
        del newApprox
        newApprox = approx.read_from_xml(outputXmlPath)
        self.assertEqual(newApprox.dico.sizes, approximant.dico.sizes)
        self.assertEqual(newApprox.atomNumber, approximant.atomNumber)
        self.assertEqual(newApprox.length, approximant.length)

        # now the hardest test
        newApprox.synthesize()
        newApprox.recomposedSignal.plot()
        self.assertAlmostEquals(sum(approximant.recomposedSignal.
            data - newApprox.recomposedSignal.data), 0)

        mdctOrig = approximant.to_array()[0]
        mdctRead = newApprox.to_array()[0]

        del doc, newApprox
        # test writing with LOmp atoms
        pyCCDico = mdct_dico.LODico([256, 2048, 8192])
        approx_LOmp = mp.mp(pySigOriginal, pyCCDico, 10, 10, False)[0]

        outputXmlPath = "approxLOmp_test.xml"
        doc = approx_LOmp.write_to_xml(outputXmlPath)

        # Test reading from the xml flow
        newApprox = approx.read_from_xml('', doc)
        self.assertEqual(newApprox.dico.sizes, approx_LOmp.dico.sizes)
        self.assertEqual(newApprox.atomNumber, approx_LOmp.atomNumber)
        self.assertEqual(newApprox.length, approx_LOmp.length)
        self.assertAlmostEqual(
            sum(newApprox.to_array()[0] - approx_LOmp.to_array()[0]), 0)

        plt.figure()
        plt.plot(newApprox.to_array()[0])
        plt.plot(approx_LOmp.to_array()[0], 'r:')
#        plt.show()

        # test reading from the xml file
        del newApprox
        newApprox = approx.read_from_xml(outputXmlPath)
        self.assertEqual(newApprox.dico.sizes, approx_LOmp.dico.sizes)
        self.assertEqual(newApprox.atomNumber, approx_LOmp.atomNumber)
        self.assertEqual(newApprox.length, approx_LOmp.length)
        self.assertAlmostEqual(
            sum(newApprox.to_array()[0] - approx_LOmp.to_array()[0]), 0)


class py_mpTest2(unittest.TestCase):
    def runTest(self):

        pyCCDico = mdct_dico.LODico([256, 2048, 8192])
        pyDico = mdct_dico.Dico([256, 2048, 8192])

        pySigOriginal = signals.Signal(
            audioFilePath + "ClocheB.wav", doNormalize=True, forceMono=True)
        pySigOriginal.crop(0, 1 * 16384)

        # intentionally non aligned atom -> objective is to fit it completely
        # through time shift calculation
        pyAtom = mdct_atom.Atom(2048, 1, 8192 - 512, 128, 8000, 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = signals.Signal(0.0001 * random.random(
            pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)

        fig_1Atom = plt.figure()
        print "Testing one Aligned Atom with LOmp - should be perfect"
        approximant = mp.mp(pySignal_oneAtom, pyCCDico, 10, 10, debug=1)[0]
#        approximant.plot_tf()
#        plt.subplot(211)
        plt.plot(pySignal_oneAtom.data)
        plt.plot(approximant.recomposedSignal.data)
        plt.plot(
            pySignal_oneAtom.data - approximant.recomposedSignal.data)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Aligned Atom with LOmp")
#        plt.show()
        plt.xlabel("samples")
#        plt.savefig(figPath +"LOmp_aligned" +ext)

#        approximant.write_to_xml(ApproxPath+"LOmp_aligned")

        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del pyAtom, approximant, pySignal_oneAtom

        print "Testing one Non-Aligned Atom with mp - should not be perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = signals.Signal(0.0001 * random.random(
            pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)

        approximant = mp.mp(pySignal_oneAtom, pyDico, 10, 10, debug=1)[0]
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySignal_oneAtom.data)
        plt.plot(approximant.recomposedSignal.data)
        plt.plot(
            pySignal_oneAtom.data - approximant.recomposedSignal.data)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Non aligned atom with mp : SRR of " + str(int(approximant.compute_srr())) + " dB in " + str(approximant.atomNumber) + " iteration")
#        plt.show()
        print " Approx Reached : ", int(approximant.compute_srr()
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant

        print "Testing one Non-Aligned Atom with LOmp - should be almost perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = signals.Signal(0.0001 * random.random(
            pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)

        approximant = mp.mp(
            pySignal_oneAtom, pyCCDico, 10, 10, False, False)[0]
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(pySignal_oneAtom.data)
        plt.plot(approximant.recomposedSignal.data)
        plt.plot(
            pySignal_oneAtom.data - approximant.recomposedSignal.data)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Non aligned atom with LOmp : SRR of " + str(int(approximant.compute_srr())) + " dB in " + str(approximant.atomNumber) + " iteration")
        plt.xlabel("samples")
#        plt.savefig(figPath+"nonAlignedAtom_mp_vs_LOmp" + ext)
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant

        print "Testing Real signals with mp"
        approximant = mp.mp(pySigOriginal, pyDico, 10, 10, False)[0]
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.data[4096:-4096])
        plt.plot(approximant.recomposedSignal.data[4096:-4096])
        plt.plot(pySigOriginal.data[4096:-4096] - approximant.
            recomposedSignal.data[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with mp : SRR of " + str(int(approximant.compute_srr(
            ))) + " dB in " + str(approximant.atomNumber) + " iteration")
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant

        print "Testing Real signals with LOmp"
#        pyCCDico =
        approximant = mp.mp(pySigOriginal, pyCCDico, 10, 10, False, False)[0]
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(pySigOriginal.data[4096:-4096])
        plt.plot(approximant.recomposedSignal.data[4096:-4096])
        plt.plot(pySigOriginal.data[4096:-4096] - approximant.
            recomposedSignal.data[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with LOmp : SRR of " + str(int(approximant.compute_srr(
            ))) + " dB in " + str(approximant.atomNumber) + " iteration")
        plt.xlabel("samples")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_nbIt10"+ext)
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant
        print "comparing results and processing times for long decompositions"
        t0 = time.clock()
        approx1 = mp.mp(pySigOriginal, pyDico, 20, 500, False)[0]
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.data[12288:-12288])
        plt.plot(approx1.recomposedSignal.data[12288:-12288])
        plt.plot(pySigOriginal.data[12288:-12288] - approx1.
            recomposedSignal.data[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : ", approx1.compute_srr(), " dB in ", approx1.atomNumber, " iteration and: ", t1 - t0, " seconds"

        t2 = time.clock()
        approx2 = mp.mp(pySigOriginal, pyCCDico, 20, 500, False, False)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pySigOriginal.data[12288:-12288])
        plt.plot(approx2.recomposedSignal.data[12288:-12288])
        plt.plot(pySigOriginal.data[12288:-12288] - approx2.
            recomposedSignal.data[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3 - t2) + "s")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atomNumber, " iteration and: ", t3 - t2, " seconds"

        del approx1, approx2
        del pySigOriginal
        print "comparing results and processing times for long decompositions of white gaussian noise"
        pyNoiseSignal = signals.Signal(
            0.5 * random.random(5 * 16384), 44100, False)
        pyNoiseSignal.pad(16384)
        t0 = time.clock()
        approx1 = mp.mp(pyNoiseSignal, pyDico, 10, 500, False, True)[0]
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pyNoiseSignal.data[16384:-16384])
        plt.plot(approx1.recomposedSignal.data[16384:-16384])
        plt.plot(pyNoiseSignal.data[16384:-16384] - approx1.
            recomposedSignal.data[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("White Noise signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : ", approx1.compute_srr(), " dB in ", approx1.atomNumber, " iteration and: ", t1 - t0, " seconds"

        t2 = time.clock()
        approx2 = mp.mp(pyNoiseSignal, pyCCDico, 10, 500, False, True)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pyNoiseSignal.data[16384:-16384])
        plt.plot(approx2.recomposedSignal.data[16384:-16384])
        plt.plot(pyNoiseSignal.data[16384:-16384] - approx2.
            recomposedSignal.data[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Noise signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3 - t2) + "s")
        plt.xlabel("samples")
#        plt.savefig(figPath+"NoiseSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atomNumber, " iteration and: ", t3 - t2, " seconds"


class py_mpTest2Bis(unittest.TestCase):
    def runTest(self):
        ApproxPath = "../Approxs/"
#        ext = ".png"

        pyRandomDico = random_dico.RandomDico([256, 2048, 8192], 'scale')
        pyDico = mdct_dico.Dico([256, 2048, 8192])

        pySigOriginal = signals.Signal(
            audioFilePath + "ClocheB.wav", doNormalize=True, forceMono=True)
        pySigOriginal.crop(0, 1 * 16384)

        # intentionally non aligned atom -> objective is to fit it completely
        # through time shift calculation
        pyAtom = mdct_atom.Atom(2048, 1, 8192 - 512, 128, 8000, 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = signals.Signal(0.0001 * random.random(
            pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)

        fig_1Atom = plt.figure()
        print "Testing one Aligned Atom with Random"
        approximant = mp.mp(
            pySignal_oneAtom, pyRandomDico, 10, 10, False, False)[0]
#        approximant.plot_tf()
#        plt.subplot(211)
        plt.plot(pySignal_oneAtom.data)
        plt.plot(approximant.recomposedSignal.data)
        plt.plot(
            pySignal_oneAtom.data - approximant.recomposedSignal.data)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Aligned Atom with LOmp")
#        plt.show()
        plt.xlabel("samples")
#        plt.savefig(figPath +"LOmp_aligned" +ext)

        approximant.write_to_xml(ApproxPath + "LOmp_aligned")

        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del pyAtom, approximant, pySignal_oneAtom

        print "Testing one Non-Aligned Atom with mp - should not be perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = signals.Signal(0.0001 * random.random(
            pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)

        approximant, decay = mp.mp(
            pySignal_oneAtom, pyDico, 10, 10, False, False)
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySignal_oneAtom.data)
        plt.plot(approximant.recomposedSignal.data)
        plt.plot(
            pySignal_oneAtom.data - approximant.recomposedSignal.data)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Non aligned atom with mp : SRR of " + str(int(approximant.compute_srr())) + " dB in " + str(approximant.atomNumber) + " iteration")
#        plt.show()
        print " Approx Reached : ", int(approximant.compute_srr()
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant

        print "Testing one Non-Aligned Atom with Random"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = signals.Signal(0.0001 * random.random(
            pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)

        approximant, decay = mp.mp(
            pySignal_oneAtom, pyRandomDico, 10, 10, False, False)
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(pySignal_oneAtom.data)
        plt.plot(approximant.recomposedSignal.data)
        plt.plot(
            pySignal_oneAtom.data - approximant.recomposedSignal.data)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Non aligned atom with LOmp : SRR of " + str(int(approximant.compute_srr())) + " dB in " + str(approximant.atomNumber) + " iteration")
        plt.xlabel("samples")
#        plt.savefig(figPath+"nonAlignedAtom_mp_vs_LOmp" + ext)
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant

        print "Testing Real signals with mp"
        approximant, decay = mp.mp(pySigOriginal, pyDico, 10, 10, False)
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.data[4096:-4096])
        plt.plot(approximant.recomposedSignal.data[4096:-4096])
        plt.plot(pySigOriginal.data[4096:-4096] - approximant.
            recomposedSignal.data[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with mp : SRR of " + str(int(approximant.compute_srr(
            ))) + " dB in " + str(approximant.atomNumber) + " iteration")
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant

        print "Testing Real signals with Randommp"
#        pyCCDico =
        approximant = mp.mp(
            pySigOriginal, pyRandomDico, 10, 10, False, False)[0]
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(pySigOriginal.data[4096:-4096])
        plt.plot(approximant.recomposedSignal.data[4096:-4096])
        plt.plot(pySigOriginal.data[4096:-4096] - approximant.
            recomposedSignal.data[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with LOmp : SRR of " + str(int(approximant.compute_srr(
            ))) + " dB in " + str(approximant.atomNumber) + " iteration")
        plt.xlabel("samples")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_nbIt10"+ext)
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atomNumber, " iteration"

        del approximant
        print "comparing results and processing times for long decompositions"
        t0 = time.clock()
        approx1, decay = mp.mp(pySigOriginal, pyDico, 20, 500, False)
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.data[12288:-12288])
        plt.plot(approx1.recomposedSignal.data[12288:-12288])
        plt.plot(pySigOriginal.data[12288:-12288] - approx1.
            recomposedSignal.data[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : ", approx1.compute_srr(), " dB in ", approx1.atomNumber, " iteration and: ", t1 - t0, " seconds"

        t2 = time.clock()
        approx2 = mp.mp(pySigOriginal, pyRandomDico, 20, 500, False, False)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pySigOriginal.data[12288:-12288])
        plt.plot(approx2.recomposedSignal.data[12288:-12288])
        plt.plot(pySigOriginal.data[12288:-12288] - approx2.
            recomposedSignal.data[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3 - t2) + "s")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atomNumber, " iteration and: ", t3 - t2, " seconds"

        del approx1, approx2
        del pySigOriginal
        print "comparing results and processing times for long decompositions of white gaussian noise"
        pyNoiseSignal = signals.Signal(
            0.5 * random.random(5 * 16384), 44100, False)
        pyNoiseSignal.pad(16384)
        t0 = time.clock()
        approx1, decay = mp.mp(pyNoiseSignal, pyDico, 10, 500, False, True)
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pyNoiseSignal.data[16384:-16384])
        plt.plot(approx1.recomposedSignal.data[16384:-16384])
        plt.plot(pyNoiseSignal.data[16384:-16384] - approx1.
            recomposedSignal.data[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("White Noise signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : ", approx1.compute_srr(), " dB in ", approx1.atomNumber, " iteration and: ", t1 - t0, " seconds"

        t2 = time.clock()
        approx2 = mp.mp(pyNoiseSignal, pyRandomDico, 10, 500, False, True)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pyNoiseSignal.data[16384:-16384])
        plt.plot(approx2.recomposedSignal.data[16384:-16384])
        plt.plot(pyNoiseSignal.data[16384:-16384] - approx2.
            recomposedSignal.data[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Noise signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3 - t2) + "s")
        plt.xlabel("samples")
#        plt.savefig(figPath+"NoiseSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atomNumber, " iteration and: ", t3 - t2, " seconds"


class py_mpTest3(unittest.TestCase):
    """ this time we decompose a longer signals with the mp_LongSignal : result in an enframed signals """
    def runTest(self):
        filePath = audioFilePath + "Bach_prelude_4s.wav"

        # Let us load a long signals: not loaded in memory for now, only
        # segment by segment in the mp process
        mdctDico = [256, 1024, 8192]
        frameSize = 5 * 8192

        # signals buidling
        originalSignal = signals.LongSignal(filePath, frameSize, True)

        # dictionaries
        pyCCDico = mdct_dico.LODico(mdctDico)
        pyDico = mdct_dico.Dico(mdctDico)

        # Let's feed the proto 3 with these:
        # we should get a collection of approximants (one for each frame) in
        # return
        xmlOutPutDir = '../../Approxs/Bach_prelude/LOmp/'
        approximants, decays = mp.mp_long(
            originalSignal, pyCCDico, 10, 100, False, True, xmlOutPutDir)

#        del approximants
#        xmlOutPutDir = '../Approxs/Bach_prelude/mp/'
# approximants , decays = mp.mp_LongSignal(originalSignal, pyDico, 5, 100,
# False, True, xmlOutPutDir)
#
        # concatenate all segments to retrieve the global approximation
#        recomposedData = zeros(8192)
#        for segIdx in range(len(approximants)) :
#            recomposedData = concatenate()

        self.assertEqual(len(approximants), originalSignal.segmentNumber)

        fusionnedApprox = approx.fusion_approxs(approximants)
        self.assertEqual(fusionnedApprox.samplingFrequency,
             originalSignal.samplingFrequency)
#        self.assertEqual(fusionnedApprox.length, originalSignal.length )
        plt.figure
        fusionnedApprox.plot_tf()
#        plt.show()


if __name__ == '__main__':
    import matplotlib
    print matplotlib.__version__

    _Logger = log.Log('test', level=3, imode=False)
    _Logger.info('Starting Tests')
    suite = unittest.TestSuite()

    suite.addTest(py_mpTest3())
    suite.addTest(py_mpTest())
    suite.addTest(py_mpTest2())
    suite.addTest(ApproxTest())
    suite.addTest(AtomTest())
    suite.addTest(DicoTest())
    suite.addTest(BlockTest())
    suite.addTest(Signaltest())
##
    unittest.TextTestRunner(verbosity=2).run(suite)

    plt.show()
    _Logger.info('Tests stopped')
