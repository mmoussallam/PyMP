"""
Python test file for PyMP engine

testing normal behavior of most MP functions and classes

M.Moussallam
"""
import matplotlib
matplotlib.use('Agg')  # to avoid display while testing

import os
import os.path as op
import sys

import unittest
from PyMP.tools import mdct
import matplotlib.pyplot as plt
import numpy as np
import time

import cProfile
import math
global _Logger

#from Classes.gabor import *
from PyMP.mdct import atom as mdct_atom
from PyMP.mdct import dico as mdct_dico
from PyMP.mdct import block as mdct_block
from PyMP.base import BaseAtom, BaseDico
from PyMP.mdct.rand import dico as random_dico
from PyMP import approx
from PyMP import signals
from PyMP import log
from PyMP import mp
from PyMP import parallelProjections

audioFilePath = op.join(op.dirname(__file__), '..', '..', 'data')


class AtomTest(unittest.TestCase):
    def setUp(self):
        pass

    def runTest(self):
        # empty creation
        atom = BaseAtom()
        self.assertEqual(atom.length, 0)
        self.assertEqual(atom.amplitude, 0)
        self.assertEqual(atom.nature, 'Abstract')

        del atom

        # full creation
        atom2 = mdct_atom.Atom(1024, 1, 12432, 128, 44100, 0.57)
        self.assertEqual(atom2.length, 1024)
        self.assertEqual(atom2.amplitude, 1)
        self.assertEqual(atom2.mdct_value, 0.57)
        self.assertEqual(atom2.fs, 44100)
        self.assertEqual(atom2.reduced_frequency, (128 + 0.5) / 1024.)
        self.assertEqual(atom2.time_position, 12432)
        self.assertEqual(atom2.nature, 'MDCT')

        print atom2
#        synthesizedAtom2 = atom2.synthesize()

        synthAtom3 = atom2.synthesize_ifft()

#        energy1 = sum(synthesizedAtom2.waveform**2)
        energy2 = sum(synthAtom3.real ** 2)

        print energy2

#        plt.plot(synthesizedAtom2.real)
        plt.plot(synthAtom3.real)
        del atom2

        print " testing LOmp atoms synthesis "
        mdct_value = 0.57
        timeShift = 144
        projection_score = -0.59
        atom_LOmp = mdct_atom.Atom(1024, 1, 12432, 128, 44100, 0.57)
        atom_LOmp.time_shift = timeShift
        atom_LOmp.proj_score = projection_score

        # test 1 synthesis
        atom_LOmp.synthesize_ifft()
        wf1 = atom_LOmp.waveform.copy()

        wf2 = -(math.sqrt(abs(projection_score) / sum(wf1 ** 2))) * wf1

        mdctVec = np.zeros(3 * 1024)
        mdctVec[1024 + 128] = projection_score
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
        dico = BaseDico()
        self.assertEqual(dico.nature, 'Abstract')

        # test dictionary class
        dico = mdct_dico.Dico(
            [2 ** l for l in range(7, 15, 1)], debug_level=3)
        self.assertEqual(dico.sizes, [128, 256, 512, 1024,
             2048, 4096, 8192, 16384])
        self.assertEqual(dico.nature, 'MDCT')


class Signaltest(unittest.TestCase):
    def runTest(self):

        signal = signals.Signal(debug_level=3)
        self.assertEqual(signal.length, 0)
        self.assertEqual(len(signal.data), 0)

        del signal

        signal = signals.Signal(op.join(audioFilePath, "ClocheB.wav"))
        self.assertNotEqual(signal.length, 0)
        self.assertNotEqual(signal.data, [])
        self.assertEqual(signal.channel_num, 2)
        self.assertEqual(signal.location, op.join(audioFilePath, "ClocheB.wav"))
        self.assertEqual(signal.fs, 8000)

        # Last test, the wigner ville plot

        signal.crop(8000, 8256)

        signal.wigner_plot()

        del signal

        signal = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                               normalize=True, mono=True)
        data1 = signal.data.copy()
        # try adding and subtracting an atom
        pyAtom = mdct_atom.Atom(1024, 0.57, 4500, 128, 8000, 0.57)
        signal.add(pyAtom)

        data2 = signal.data.copy()

        self.assertNotEqual(sum((data1 - data2) ** 2), 0)
        signal.subtract(pyAtom)

#        plt.plot(data1)
#        plt.plot(data2)
#        plt.plot(signal.data)
#        plt.legend(("original", "added" , "subtracted"))
#
        self.assertAlmostEqual(sum((signal.data - data1) ** 2), 0)

        # test on a long signals
        L = 4 * 16384
        longSignal = signals.LongSignal(op.join(audioFilePath,
                                                "Bach_prelude_40s.wav"), L)
        # suppose we want to retrieve the middle of the signals , namely from
        # frame 5  to 12
        startSeg = 2
        segNumber = 3
        shortSignal = longSignal.get_sub_signal(
            startSeg, segNumber, normalize=True)

        # witness signals
        witSignal = signals.Signal(op.join(audioFilePath,
                                           "Bach_prelude_40s.wav"),
                                   normalize=True, mono=False)

#        shortSignal.plot()
#        witSignal.plot()
        plt.figure()
        plt.plot(shortSignal.data)
        witSignal[startSeg * L: (startSeg + segNumber) * L].plot(pltStr='r:')

        # test writing signals
        outputPath = 'subsignal.wav'
        if os.path.exists(outputPath):
            os.remove(outputPath)

        shortSignal.write(outputPath)

        # test long signals with overlap 50 %
        longSignal = signals.LongSignal(op.join(audioFilePath,
                                                "Bach_prelude_40s.wav"),
                                        L, True, 0.5)
        # suppose we want to retrieve the middle of the signals , namely from
        # frame 5  to 12
        startSeg = 2
        segNumber = 3
        shortSignal = longSignal.get_sub_signal(startSeg, segNumber, False)

        # witness signals
        witSignal = signals.Signal(op.join(audioFilePath,
                                           "Bach_prelude_40s.wav"),
                                   normalize=True, mono=False)

#        shortSignal.plot()
#        witSignal.plot()
        plt.figure()
        plt.plot(shortSignal.data)
        witSignal[startSeg * L / 2:startSeg * L / 2 + segNumber * L].plot()

        # Testing the downsampliong utility
        plt.figure()
        plt.subplot(121)
        plt.specgram(shortSignal.data, 256, shortSignal.
            fs, noverlap=128)
        self.assertEqual(shortSignal.fs, 44100)
        shortSignal.write('normal.wav')
        shortSignal.downsample(8000)
        self.assertEqual(shortSignal.fs, 8000)
        plt.subplot(122)
        plt.specgram(shortSignal.data, 256, shortSignal.
            fs, noverlap=128)

        shortSignal.write('downsampled.wav')


#        plt.show()
class BlockTest(unittest.TestCase):
    def runTest(self):

        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       normalize=True, mono=True)
        signal_original.crop(0, 5 * 8192)
        signal_original.pad(2048)

        block = mdct_block.Block(
            1024, signal_original, debug_level=2, useC=True)
        # testing the automated enframing of the data
        self.assertEqual(block.frame_len, 512)
        print block.frame_num
#        import parallelProjections
        parallelProjections.initialize_plans(
            np.array([1024, ]), np.array([2, ]))

        try:
            import fftw3
        except ImportError:
            print " FFTW3 python wrapper not installed, abandonning test"
#            return
        block.update(signal_original)

        plt.plot(block.projs_matrix.flatten(1))
        plt.plot(mdct.mdct(signal_original.data, block.scale))
#        plt.show()

        self.assertAlmostEqual(np.sum((block.projs_matrix - mdct.
            mdct(signal_original.data, block.scale)) ** 2), 0)

        parallelProjections.clean_plans(np.array([1024, ]))
        print "Cleaning done"
        del signal_original


class py_mpTest(unittest.TestCase):
    def runTest(self):
# dico = Dico.Dico([2**l for l in range(7,15,1)] , Atom.transformType.MDCT)

        dico = mdct_dico.Dico([256, 2048, 8192])
        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       normalize=True, mono=True)
        signal_original.crop(0, 5 * 16384)

        signal_original.data += 0.01 * np.random.random(5 * 16384)

        # first try with a single-atom signals
        pyAtom = mdct_atom.Atom(2048, 1, 11775, 128, 8000, 0.57)
        pyApprox_oneAtom = approx.Approx(dico, [], signal_original)
        pyApprox_oneAtom.add(pyAtom)

        signal_one_atom = signals.Signal(pyApprox_oneAtom.
            synthesize(0).data, signal_original.fs, False)

        # second test
        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       normalize=True, mono=True)

        # last test - decomposition profonde
        dico2 = mdct_dico.Dico([128, 256, 512, 1024, 2048,
             4096, 8192, 16384], parallel=False)
        dico1 = mdct_dico.Dico([16384])
        # profiling test
        print "Plain"
        cProfile.runctx('mp.mp(signal_original, dico1, 20, 1000 ,debug=0 , '
                        'clean=True)', globals(), locals())

        cProfile.runctx('mp.mp(signal_original, dico2, 20, 1000 ,debug=0 , '
                        'clean=True)', globals(), locals())


class SequenceDicoTest(unittest.TestCase):

    def runTest(self):
# dico = Dico.Dico([2**l for l in range(7,15,1)] , Atom.transformType.MDCT)

        dico = random_dico.SequenceDico([256, 2048, 8192], seq_type='random')
        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       normalize=True, mono=True)
        signal_original.crop(0, 5 * 16384)

        signal_original.data += 0.01 * np.random.random(5 * 16384)

        cProfile.runctx('mp.mp(signal_original, dico, 20, 20, debug=1, '
                        'clean=True)', globals(), locals())


class ApproxTest(unittest.TestCase):

    def runTest(self):
#        self.writeXmlTest()

        # test dictionary class

        pyApprox = approx.Approx(debug_level=3)
        self.assertEqual(pyApprox.original_signal, None)
        self.assertEqual(pyApprox.atom_number, 0)
        self.assertEqual(pyApprox.srr, 0)
        self.assertEqual(pyApprox.atoms, [])

        print pyApprox
        del pyApprox

        dico = mdct_dico.Dico([2 ** l for l in range(7, 15, 1)])
        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       mono=True)
        signal_original.crop(0, 5 * max(dico.sizes))

        pyApprox = approx.Approx(dico, [], signal_original)

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
            pyAtom, pyApprox.filter_atoms(1024, None, None).atoms[0])
        self.assertEqual(pyAtom, pyApprox.filter_atoms(1024, [
            12000, 15000], None).atoms[0])

        print pyApprox.compute_srr()
        # TODO testing du SRR

        #testing the write_to_xml and read_from_xml methods
        pyCCDico = mdct_dico.LODico([256, 2048, 8192])
        approx_LOmp = mp.mp(signal_original, pyCCDico, 10, 100, False)[0]

        sliceApprox = approx_LOmp[:10]

        print approx_LOmp
        print sliceApprox
        sliceApprox.compute_srr()
        print sliceApprox
        plt.figure()
        plt.subplot(121)
        approx_LOmp.plot_tf()
        plt.subplot(122)
        sliceApprox.plot_tf()
        plt.plot()

        del signal_original

    def writeXmlTest(self):
        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       normalize=True, mono=True,
                                       debug_level=0)
        signal_original.crop(0, 1 * 16384)
        dico = mdct_dico.Dico([256, 2048, 8192], debug_level=2)
        # first compute an approximant using mp
        approximant = mp.mp(signal_original, dico, 10, 10, debug=2)[0]

        outputXmlPath = "approx_test.xml"
        doc = approximant.write_to_xml(outputXmlPath)

        # Test reading from the xml flow
        newApprox = approx.read_from_xml('', doc)
        self.assertEqual(newApprox.dico.sizes, approximant.dico.sizes)
        self.assertEqual(newApprox.atom_number, approximant.atom_number)
        self.assertEqual(newApprox.length, approximant.length)

        # now the hardest test
        newApprox.synthesize()
        newApprox.recomposed_signal.plot()
        self.assertAlmostEquals(sum(approximant.recomposed_signal.
            data - newApprox.recomposed_signal.data), 0)

        # test reading from the xml file
        del newApprox
        newApprox = approx.read_from_xml(outputXmlPath)
        self.assertEqual(newApprox.dico.sizes, approximant.dico.sizes)
        self.assertEqual(newApprox.atom_number, approximant.atom_number)
        self.assertEqual(newApprox.length, approximant.length)

        # now the hardest test
        newApprox.synthesize()
        newApprox.recomposed_signal.plot()
        self.assertAlmostEquals(sum(approximant.recomposed_signal.
            data - newApprox.recomposed_signal.data), 0)

        mdctOrig = approximant.to_array()[0]
        mdctRead = newApprox.to_array()[0]

        del doc, newApprox
        # test writing with LOmp atoms
        pyCCDico = mdct_dico.LODico([256, 2048, 8192])
        approx_LOmp = mp.mp(signal_original, pyCCDico, 10, 100, False)[0]

        outputXmlPath = "approxLOmp_test.xml"
        doc = approx_LOmp.write_to_xml(outputXmlPath)

        # Test reading from the xml flow
        newApprox = approx.read_from_xml('', doc)
        self.assertEqual(newApprox.dico.sizes, approx_LOmp.dico.sizes)
        self.assertEqual(newApprox.atom_number, approx_LOmp.atom_number)
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
        self.assertEqual(newApprox.atom_number, approx_LOmp.atom_number)
        self.assertEqual(newApprox.length, approx_LOmp.length)
        self.assertAlmostEqual(
            sum(newApprox.to_array()[0] - approx_LOmp.to_array()[0]), 0)

        del newApprox


class py_mpTest2(unittest.TestCase):
    def runTest(self):

        pyCCDico = mdct_dico.LODico([256, 2048, 8192])
        dico = mdct_dico.Dico([256, 2048, 8192])

        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       normalize=True, mono=True)
        signal_original.crop(0, 1 * 16384)

        # intentionally non aligned atom -> objective is to fit it completely
        # through time shift calculation
        pyAtom = mdct_atom.Atom(2048, 1, 8192 - 512, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        fig_1Atom = plt.figure()
        print "Testing one Aligned Atom with LOmp - should be perfect"
        approximant = mp.mp(signal_one_atom, pyCCDico, 10, 10, debug=2)[0]
#        approximant.plot_tf()
#        plt.subplot(211)
        plt.plot(signal_one_atom.data)
        plt.plot(approximant.recomposed_signal.data)
        plt.plot(
            signal_one_atom.data - approximant.recomposed_signal.data)
        plt.legend(("original", "approximant", "residual"))
        plt.title("Aligned Atom with LOmp")
#        plt.show()
        plt.xlabel("samples")
#        plt.savefig(figPath +"LOmp_aligned" +ext)

#        approximant.write_to_xml(approx_path+"LOmp_aligned")

        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atom_number, " iteration"

        del pyAtom, approximant, signal_one_atom

        print "Testing one Non-Aligned Atom with mp - should not be perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        approximant = mp.mp(signal_one_atom, dico, 10, 10, debug=1)[0]
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(signal_one_atom.data)
        plt.plot(approximant.recomposed_signal.data)
        plt.plot(
            signal_one_atom.data - approximant.recomposed_signal.data)
        plt.legend(("original", "approximant", "residual"))
        plt.title("Non aligned atom with mp : SRR of "
                  + str(int(approximant.compute_srr())) + " dB in "
                  + str(approximant.atom_number) + " iteration")
#        plt.show()
        print " Approx Reached : ", int(approximant.compute_srr()
            ), " dB in ", approximant.atom_number, " iteration"

        del approximant

        print "Testing one Non-Aligned Atom with LOmp - should be almost perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        approximant = mp.mp(
            signal_one_atom, pyCCDico, 10, 10, False, False)[0]
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(signal_one_atom.data)
        plt.plot(approximant.recomposed_signal.data)
        plt.plot(
            signal_one_atom.data - approximant.recomposed_signal.data)
        plt.legend(("original", "approximant", "residual"))
        plt.title("Non aligned atom with LOmp : SRR of "
                  + str(int(approximant.compute_srr())) + " dB in "
                  + str(approximant.atom_number) + " iteration")
        plt.xlabel("samples")
#        plt.savefig(figPath+"nonAlignedAtom_mp_vs_LOmp" + ext)
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atom_number, " iteration"

        del approximant

        print "Testing Real signals with mp"
        approximant = mp.mp(signal_original, dico, 10, 10, False)[0]
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(signal_original.data[4096:-4096])
        plt.plot(approximant.recomposed_signal.data[4096:-4096])
        plt.plot(signal_original.data[4096:-4096] - approximant.
            recomposed_signal.data[4096:-4096])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with mp : SRR of %d dB in %d iteration" %
                  (approximant.compute_srr(), approximant.atom_number))
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atom_number, " iteration"

        del approximant

        print "Testing Real signals with LOmp"
#        pyCCDico =
        approximant = mp.mp(signal_original, pyCCDico, 10, 10, False, False)[0]
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(signal_original.data[4096:-4096])
        plt.plot(approximant.recomposed_signal.data[4096:-4096])
        plt.plot(signal_original.data[4096:-4096] - approximant.
            recomposed_signal.data[4096:-4096])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with LOmp : SRR of %d dB in %d iteration" %
                  (approximant.compute_srr(), approximant.atom_number))
        plt.xlabel("samples")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_nbIt10"+ext)
        print(" Approx Reached : ", approximant.compute_srr(),
              " dB in ", approximant.atom_number, " iteration")

        del approximant
        print "comparing results and processing times for long decompositions"
        t0 = time.clock()
        approx1 = mp.mp(signal_original, dico, 20, 500, False)[0]
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(signal_original.data[12288:-12288])
        plt.plot(approx1.recomposed_signal.data[12288:-12288])
        plt.plot(signal_original.data[12288:-12288] - approx1.
            recomposed_signal.data[12288:-12288])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with mp : SRR of %d dB in %d iteration and %s s" %
                  (approx1.compute_srr(), approx1.atom_number, t1 - t0))
#        plt.show()
        print(" Approx Reached : ", approx1.compute_srr(), " dB in ",
              approx1.atom_number, " iteration and: ", t1 - t0, " seconds")

        t2 = time.clock()
        approx2 = mp.mp(signal_original, pyCCDico, 20, 500, False, False)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(signal_original.data[12288:-12288])
        plt.plot(approx2.recomposed_signal.data[12288:-12288])
        plt.plot(signal_original.data[12288:-12288] - approx2.
            recomposed_signal.data[12288:-12288])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atom_number) + " iteration and " + str(t3 - t2) + "s")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atom_number, " iteration and: ", t3 - t2, " seconds"

        del approx1, approx2
        del signal_original
        print "comparing results and processing times for long decompositions of white gaussian noise"
        noise_signal = signals.Signal(
            0.5 * np.random.random(5 * 16384), 44100, False)
        noise_signal.pad(16384)
        t0 = time.clock()
        approx1 = mp.mp(noise_signal, dico, 10, 500, False, True)[0]
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(noise_signal.data[16384:-16384])
        plt.plot(approx1.recomposed_signal.data[16384:-16384])
        plt.plot(noise_signal.data[16384:-16384] - approx1.
            recomposed_signal.data[16384:-16384])
        plt.legend(("original", "approximant", "residual"))
        plt.title("White Noise signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atom_number) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : ", approx1.compute_srr(), " dB in ", approx1.atom_number, " iteration and: ", t1 - t0, " seconds"

        t2 = time.clock()
        approx2 = mp.mp(noise_signal, pyCCDico, 10, 500, False, True)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(noise_signal.data[16384:-16384])
        plt.plot(approx2.recomposed_signal.data[16384:-16384])
        plt.plot(noise_signal.data[16384:-16384] - approx2.
            recomposed_signal.data[16384:-16384])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Noise signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atom_number) + " iteration and " + str(t3 - t2) + "s")
        plt.xlabel("samples")
#        plt.savefig(figPath+"NoiseSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atom_number, " iteration and: ", t3 - t2, " seconds"


class py_mpTest2Bis(unittest.TestCase):
    def runTest(self):
        approx_path = "../Approxs/"
#        ext = ".png"

        rand_dico = random_dico.SequenceDico([256, 2048, 8192], 'scale')
        dico = mdct_dico.Dico([256, 2048, 8192])

        signal_original = signals.Signal(op.join(audioFilePath, "ClocheB.wav"),
                                       normalize=True, mono=True)
        signal_original.crop(0, 1 * 16384)

        # intentionally non aligned atom -> objective is to fit it completely
        # through time shift calculation
        pyAtom = mdct_atom.Atom(2048, 1, 8192 - 512, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        fig_1Atom = plt.figure()
        print "Testing one Aligned Atom with Random"
        approximant = mp.mp(
            signal_one_atom, rand_dico, 10, 10, False, False)[0]
#        approximant.plot_tf()
#        plt.subplot(211)
        plt.plot(signal_one_atom.data)
        plt.plot(approximant.recomposed_signal.data)
        plt.plot(
            signal_one_atom.data - approximant.recomposed_signal.data)
        plt.legend(("original", "approximant", "residual"))
        plt.title("Aligned Atom with LOmp")
#        plt.show()
        plt.xlabel("samples")
#        plt.savefig(figPath +"LOmp_aligned" +ext)

        approximant.write_to_xml(approx_path + "LOmp_aligned")

        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atom_number, " iteration"

        del pyAtom, approximant, signal_one_atom

        print "Testing one Non-Aligned Atom with mp - should not be perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        approximant, decay = mp.mp(
            signal_one_atom, dico, 10, 10, False, False)
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(signal_one_atom.data)
        plt.plot(approximant.recomposed_signal.data)
        plt.plot(
            signal_one_atom.data - approximant.recomposed_signal.data)
        plt.legend(("original", "approximant", "residual"))
        plt.title("Non aligned atom with mp : SRR of " + str(int(approximant.compute_srr())) + " dB in " + str(approximant.atom_number) + " iteration")
#        plt.show()
        print " Approx Reached : ", int(approximant.compute_srr()
            ), " dB in ", approximant.atom_number, " iteration"

        del approximant

        print "Testing one Non-Aligned Atom with Random"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        approximant, decay = mp.mp(
            signal_one_atom, rand_dico, 10, 10, False, False)
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(signal_one_atom.data)
        plt.plot(approximant.recomposed_signal.data)
        plt.plot(signal_one_atom.data - approximant.recomposed_signal.data)
        plt.legend(("original", "approximant", "residual"))
        plt.title("Non aligned atom with LOmp : SRR of " + str(int(approximant.compute_srr())) + " dB in " + str(approximant.atom_number) + " iteration")
        plt.xlabel("samples")
#        plt.savefig(figPath+"nonAlignedAtom_mp_vs_LOmp" + ext)
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atom_number, " iteration"

        del approximant

        print "Testing Real signals with mp"
        approximant, decay = mp.mp(signal_original, dico, 10, 10, False)
#        approximant.plot_tf()
        plt.figure()
        plt.subplot(211)
        plt.plot(signal_original.data[4096:-4096])
        plt.plot(approximant.recomposed_signal.data[4096:-4096])
        plt.plot(signal_original.data[4096:-4096] - approximant.
            recomposed_signal.data[4096:-4096])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with mp : SRR of " + str(int(approximant.compute_srr(
            ))) + " dB in " + str(approximant.atom_number) + " iteration")
#        plt.show()
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atom_number, " iteration"

        del approximant

        print "Testing Real signals with Randommp"
#        pyCCDico =
        approximant = mp.mp(
            signal_original, rand_dico, 10, 10, False, False)[0]
#        approximant.plot_tf()
        plt.subplot(212)
        plt.plot(signal_original.data[4096:-4096])
        plt.plot(approximant.recomposed_signal.data[4096:-4096])
        plt.plot(signal_original.data[4096:-4096] - approximant.
            recomposed_signal.data[4096:-4096])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with LOmp : SRR of " +
                  str(int(approximant.compute_srr())) +
                   " dB in " + str(approximant.atom_number) + " iteration")
        plt.xlabel("samples")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_nbIt10"+ext)
        print " Approx Reached : ", approximant.compute_srr(
            ), " dB in ", approximant.atom_number, " iteration"

        del approximant
        print "comparing results and processing times for long decompositions"
        t0 = time.clock()
        approx1, decay = mp.mp(signal_original, dico, 20, 500, False)
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(signal_original.data[12288:-12288])
        plt.plot(approx1.recomposed_signal.data[12288:-12288])
        plt.plot(signal_original.data[12288:-12288] - approx1.
            recomposed_signal.data[12288:-12288])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atom_number) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : ", approx1.compute_srr(), " dB in ", approx1.atom_number, " iteration and: ", t1 - t0, " seconds"

        t2 = time.clock()
        approx2 = mp.mp(signal_original, rand_dico, 20, 500, False, False)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(signal_original.data[12288:-12288])
        plt.plot(approx2.recomposed_signal.data[12288:-12288])
        plt.plot(signal_original.data[12288:-12288] - approx2.
            recomposed_signal.data[12288:-12288])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atom_number) + " iteration and " + str(t3 - t2) + "s")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atom_number, " iteration and: ", t3 - t2, " seconds"

        del approx1, approx2
        del signal_original
        print "comparing results and processing times for long decompositions of white gaussian noise"
        noise_signal = signals.Signal(
            0.5 * np.random.random(5 * 16384), 44100, False)
        noise_signal.pad(16384)
        t0 = time.clock()
        approx1, decay = mp.mp(noise_signal, dico, 10, 500, False, True)
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(noise_signal.data[16384:-16384])
        plt.plot(approx1.recomposed_signal.data[16384:-16384])
        plt.plot(noise_signal.data[16384:-16384]
                 - approx1.recomposed_signal.data[16384:-16384])
        plt.legend(("original", "approximant", "residual"))
        plt.title("White Noise signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atom_number) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : ", approx1.compute_srr(), " dB in ", approx1.atom_number, " iteration and: ", t1 - t0, " seconds"

        t2 = time.clock()
        approx2 = mp.mp(noise_signal, rand_dico, 10, 500, False, True)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(noise_signal.data[16384:-16384])
        plt.plot(approx2.recomposed_signal.data[16384:-16384])
        plt.plot(noise_signal.data[16384:-16384] - approx2.
            recomposed_signal.data[16384:-16384])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Noise signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atom_number) + " iteration and " + str(t3 - t2) + "s")
        plt.xlabel("samples")
#        plt.savefig(figPath+"NoiseSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : ", approx2.compute_srr(), " dB in ", approx2.atom_number, " iteration and: ", t3 - t2, " seconds"


class py_mpTest3(unittest.TestCase):
    """ this time we decompose a longer signals with the mp_LongSignal : result in an enframed signals """
    def runTest(self):
        filePath = op.join(audioFilePath, "Bach_prelude_4s.wav")

        # Let us load a long signals: not loaded in memory for now, only
        # segment by segment in the mp process
        mdctDico = [256, 1024, 8192]
        frameSize = 5 * 8192

        # signals buidling
        original_signal = signals.LongSignal(filePath, frameSize, True)

        # dictionaries
        pyCCDico = mdct_dico.LODico(mdctDico)

        # Let's feed the proto 3 with these:
        # we should get a collection of approximants (one for each frame) in
        # return
        # xmlOutPutDir = op.join(op.dirname(__file__), '..', '..',
        # '/Approxs/Bach_prelude/LOmp/')
        xmlOutPutDir = '.'
        approximants = mp.mp_long(
            original_signal, pyCCDico, 10, 100, False, True, xmlOutPutDir)[0]

        self.assertEqual(len(approximants), original_signal.segmentNumber)

        fusionned_approx = approx.fusion_approxs(approximants)

        print fusionned_approx
        print approximants

        self.assertEqual(fusionned_approx.fs, original_signal.fs)
#        self.assertEqual(fusionned_approx.length, original_signal.length )
        plt.figure
        fusionned_approx.plot_tf()
#        plt.show()


if __name__ == '__main__':
    import matplotlib
    print matplotlib.__version__

    _Logger = log.Log('test', level=3, imode=False)
    _Logger.info('Starting Tests')
    suite = unittest.TestSuite()

#    suite.addTest(py_mpTest3())
#    suite.addTest(py_mpTest())
#    suite.addTest(py_mpTest2())
#    suite.addTest(ApproxTest())
#    suite.addTest(AtomTest())
#    suite.addTest(DicoTest())
#    suite.addTest(BlockTest())
    suite.addTest(Signaltest())
#
    unittest.TextTestRunner(verbosity=2).run(suite)

    plt.show()
    _Logger.info('Tests stopped')
