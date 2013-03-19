"""
Python test file for PyMP engine

testing normal behavior of most MP functions and classes

M.Moussallam
"""
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # to avoid display while testing

from PyMP.mdct.block import Block

import os
import os.path as op

import unittest
from PyMP.tools import mdct
import numpy as np
import time

import cProfile
import math
global _Logger

# from Classes.gabor import *
from PyMP.mdct import atom as mdct_atom
from PyMP.wavelet import atom as wavelet_atom
from PyMP.mdct import dico as mp_mdct_dico
from PyMP.mdct import block as mdct_block
from PyMP.base import BaseAtom, BaseDico
from PyMP.mdct.rand import dico as random_dico
from PyMP import approx
from PyMP import signals
from PyMP import log
from PyMP import mp
from PyMP import win_server
from PyMP import parallelProjections

audio_filepath = op.join(op.dirname(__file__), '..', '..', 'data')


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
        atom_LOmp = mdct_atom.Atom(1024, 1, 12432, 128, 44100, 1)
        atom_LOmp.time_shift = timeShift

        atom_LOmp.proj_score = projection_score

        # test 1 synthesis
        atom_LOmp.synthesize(-(math.sqrt(abs(projection_score))))
        wf1 = atom_LOmp.waveform.copy()

        wf2 = (math.sqrt(abs(projection_score) / sum(wf1 ** 2))) * wf1

        mdctVec = np.zeros(3 * 1024)
        mdctVec[1024 + 128] = -(math.sqrt(abs(projection_score)))
        wf3 = mdct.imdct(mdctVec, 1024)[0.75 * 1024: 1.75 * 1024]

#        plt.figure()
#        plt.plot(wf1)
#        plt.plot(wf2)
#        plt.plot(wf3)
#        plt.legend(('1', '2', '3'))
#
#        plt.show()

        np.testing.assert_array_almost_equal(wf1, wf2)
        np.testing.assert_array_almost_equal(wf2, wf3)

        


class WaveletAtomTest(unittest.TestCase):
    def runTest(self):

        # full creation
        atom = wavelet_atom.WaveAtom(scale=1024, amp=0.59, timePos=12432,
                                     Fs=1000, nature='db8', level=5)
        self.assertEqual(atom.length, 1024)
        self.assertEqual(atom.amplitude, 0.59)
        self.assertEqual(atom.level, 5)
        self.assertEqual(atom.fs, 1000)
        self.assertEqual(atom.time_position, 12432)
        self.assertEqual(atom.nature, 'db8')

        print atom
        print atom.get_waveform()
        plt.figure()
        plt.plot(atom.x, atom.waveform)
#        plt.show()


class DicoTest(unittest.TestCase):
    def runTest(self):
        # test dictionary class
        dico = BaseDico()
        self.assertEqual(dico.nature, 'Abstract')

        # test dictionary class
        dico = mp_mdct_dico.Dico(
            [2 ** l for l in range(7, 15, 1)], debug_level=2)
        self.assertEqual(dico.sizes, [128, 256, 512, 1024,
                                      2048, 4096, 8192, 16384])
        self.assertEqual(dico.nature, 'MDCT')


class Signaltest(unittest.TestCase):

    def runTest(self):

        signal = signals.Signal(debug_level=3)
        self.assertEqual(signal.length, 0)
        self.assertEqual(len(signal.data), 0)

        del signal

        signal = signals.Signal(op.join(audio_filepath, "ClocheB.wav"))
        self.assertNotEqual(signal.length, 0)
        self.assertNotEqual(signal.data, [])
        self.assertEqual(signal.channel_num, 2)
        self.assertEqual(
            signal.location, op.join(audio_filepath, "ClocheB.wav"))
        self.assertEqual(signal.fs, 8000)

        # Last test, the wigner ville plot

        signal.crop(8000, 8256)

        signal.wigner_plot()

        del signal

        signal = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                normalize=True, mono=True)
        data1 = signal.data.copy()
        # try adding and subtracting an atom
        # TEST bad calls
        badargs = (None,)
        self.assertRaises(TypeError, signal.add, *badargs)

        pyAtom = mdct_atom.Atom(1024, 0.57, 4500, 128, 8000, 0.57)
        signal.add(pyAtom)

        data2 = signal.data.copy()

        self.assertNotEqual(sum((data1 - data2) ** 2), 0)
        signal.subtract(pyAtom)

        self.assertAlmostEqual(sum((signal.data - data1) ** 2), 0)

        # Second time: it should not be happy!
        badargs = (pyAtom,)
        self.assertRaises(ValueError, signal.subtract, *badargs)

        # Unless we force him to accept
        signal.subtract(pyAtom, prevent_energy_increase=False)

        # testing the windowing function
        sig = signals.Signal(np.ones(128,))
        sig.window(32)
        self.assertTrue((sig.data[0:32] == np.sin(
            (np.arange(32).astype(float)) * np.pi / (2 * 32))).all())
        self.assertTrue((sig.data[-32:] == np.sin(
            (np.arange(32).astype(float)) * np.pi / (2 * 32) + np.pi / 2)).all())

#        plt.plot(data1)
#        plt.plot(data2)
#        plt.plot(signal.data)
#        plt.legend(("original", "added" , "subtracted"))
#
        # test on a long signals
        l_sig = signals.LongSignal(op.join(audio_filepath,
                                                "Bach_prelude_40s.wav"), 
                                    8192,
                                    mono=True,
                                    Noverlap=0)
        sig = signals.Signal(op.join(audio_filepath,
                                                "Bach_prelude_40s.wav"), mono=True)
        n_seg_normal =  math.floor(sig.length / 8192);
        self.assertEqual(l_sig.n_seg, n_seg_normal)

        # Now with a 50% overlap
        l_sig = signals.LongSignal(op.join(audio_filepath,
                                                "Bach_prelude_40s.wav"), 
                                    8192,
                                    mono=True,
                                    Noverlap=0.5)
        self.assertEqual(l_sig.n_seg, 2*n_seg_normal -1)       
        
        
        # Now with a 75% overlap
        l_sig = signals.LongSignal(op.join(audio_filepath,
                                                "Bach_prelude_40s.wav"), 
                                    8192,
                                    mono=True,
                                    Noverlap=0.75)
        print sig.length, n_seg_normal
        self.assertEqual(l_sig.n_seg, 4*n_seg_normal - 3)
        
        L = 4 * 16384
        longSignal = signals.LongSignal(op.join(audio_filepath,
                                                "Bach_prelude_40s.wav"), L)
        # suppose we want to retrieve the middle of the signals , namely from
        # frame 5  to 12
        startSeg = 2
        segNumber = 3
        shortSignal = longSignal.get_sub_signal(
            startSeg, segNumber, normalize=True)

        # witness signals
        witSignal = signals.Signal(op.join(audio_filepath,
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
        longSignal = signals.LongSignal(op.join(audio_filepath,
                                                "Bach_prelude_40s.wav"),
                                        L, True, 0.5)
        # suppose we want to retrieve the middle of the signals , namely from
        # frame 5  to 12
        startSeg = 2
        segNumber = 3
        shortSignal = longSignal.get_sub_signal(startSeg, segNumber, False)

        # witness signals
        witSignal = signals.Signal(op.join(audio_filepath,
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

        # Final test: play utility
        print "Playing Short excerpt from Bach Prelude"
        real_signal = signals.Signal(op.join(audio_filepath, "Bach_prelude_4s.wav"),
                                normalize=False, mono=True)
        real_signal.play()

        print "Testing too long excerpt: SHOULD NOT work"
        l_sig = signals.Signal(op.join(audio_filepath, "Bach_prelude_40s.wav"),
                                normalize=False, mono=True)
        self.assertRaises(ValueError, l_sig.play)
        
        
        print "However This should work"
        l_sig[16*l_sig.fs:19*l_sig.fs].play()
        
        
        
#        plt.show()
class BlockTest(unittest.TestCase):
    def runTest(self):

        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
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

#            return
        block.update(signal_original)

        plt.plot(block.projs_matrix.flatten(1))
        plt.plot(mdct.mdct(signal_original.data, block.scale))
#        plt.show()

        self.assertAlmostEqual(np.sum((block.projs_matrix - mdct.
                                       mdct(signal_original.data,
                                            block.scale)) ** 2), 0)

        # testing other blocks
        lomp_block = mdct_block.LOBlock(1024, signal_original)
        lomp_block.initialize()
        lomp_block.update(signal_original, 0, -1)
        lomp_best_atom = lomp_block.get_max_atom(1)

        lomp_block.plot_proj_matrix()

        full_block = mdct_block.FullBlock(1024, signal_original)
        full_block.initialize()
        full_block.update(signal_original, 0, -1)
        best_full_atom = full_block.get_max_atom()

        full_block.plot_proj_matrix()

        spread_block = mdct_block.SpreadBlock(
            1024, signal_original, penalty=0.5, maskSize=2)
        spread_block.initialize()
        spread_block.update(signal_original, 0, -1)
        spread_block = full_block.get_max_atom()

        parallelProjections.clean_plans(np.array([1024, ]))
        print "Cleaning done"
        del signal_original


class MPTest(unittest.TestCase):

    def badArgsTest(self):
        print "TESTING BAD CALLS"
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        dico = mp_mdct_dico.Dico([128, 256, 512, 1024, 2048,
                                  4096, 8192, 16384])

        # testing mp with empty signal
        badargs = (None, dico, 50, 10)
        self.assertRaises(TypeError, mp.mp, *badargs)

        # testing mp with empty dictionary
        badargs = (signal_original, None, 50, 10)
        self.assertRaises(TypeError, mp.mp, *badargs)

        # testing mp
        # asburd call : should raise a ValueError
        badargs = (signals.Signal(np.zeros(signal_original.data.shape)),
                   dico, 50, 10)

        self.assertRaises(ValueError, mp.mp, *badargs)

        mp.mp(signal_original,
              dico, 50, 1, debug_iteration=0)

    def runTest(self):

        self.badArgsTest()
# dico = Dico.Dico([2**l for l in range(7,15,1)] , Atom.transformType.MDCT)

        dico = mp_mdct_dico.Dico([256, 2048, 8192])
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
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
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)

        # last test - decomposition profonde
        dico2 = mp_mdct_dico.Dico([128, 256, 512, 1024, 2048,
                                   4096, 8192, 16384])
        dico1 = mp_mdct_dico.Dico([16384])

        # testing mp
        # asburd call : should raise a ValueError
        badargs = (signals.Signal(np.zeros(signal_original.data.shape)),
                   dico2, 50, 10)

        self.assertRaises(ValueError, mp.mp, *badargs)

        full_num = 100
        stop_num = 80
        full_approx = mp.mp(signal_original, dico2, 50, full_num, debug=0,
                            pad=False)[0]

        self.assertEqual(full_approx.atom_number, full_num)
        # testing mp_continue
        stopped_approx = mp.mp(signal_original, dico2, 50, stop_num, debug=0,
                               pad=False)[0]

        # asburd call : should raise a ValueError
        badargs = (approx.Approx(dico1, [],
                                 signal_original),
                   signals.Signal(np.zeros(signal_original.data.shape)),
                   dico2, 50,
                   full_num - stop_num)
        self.assertRaises(ValueError, mp.mp_continue, *badargs)

        # good call: should work and give same results
        completed_approx = mp.mp_continue(stopped_approx, signal_original,
                                          dico2, 50, full_num - stop_num, debug=0, pad=False)[0]
        

        self.assertEqual(full_approx.length, completed_approx.length)
        self.assertEqual(full_approx.atom_number, completed_approx.atom_number)

        # now assert all atoms are the same in both decompositions
        for i in range(full_num):
            self.assertEqual(full_approx[i], completed_approx[i])

#        self.assertAlmostEqual(full_approx.srr, completed_approx.srr)

        # profiling test
        print "Plain"
        cProfile.runctx('mp.mp(signal_original, dico1, 20, 1000 ,debug=0 , '
                        'clean=True)', globals(), locals())

        cProfile.runctx('mp.mp(signal_original, dico2, 20, 1000 ,debug=0 , '
                        'clean=True)', globals(), locals())

        print "Comparing with legacy implementation"
        n_atoms = 300
        app_1 , dec1 = mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=True, update='mp')
        app_2 , dec2 = mp.mp(signal_original, dico, 100, n_atoms ,debug=0, pad=False)
        np.testing.assert_almost_equal(dec1, dec2)
        for i in range(n_atoms):
            self.assertEquals(app_1[i],app_2[i])

        print "Comparing MP and MP_continue"
        
        cProfile.runctx('mp.mp(signal_original, dico1, 20, 100 ,debug=0 , pad=False)', globals(), locals())
        
        curr_approx = mp.mp(signal_original, dico1, 20, 100,
                            debug=0 ,clean=False, pad=False)[0]
        
        cProfile.runctx('mp.mp_continue(curr_approx, signal_original, dico1, 20, 100 ,debug=0 , pad=False)', globals(), locals())

class GreedyTest(unittest.TestCase):
    
    
    def badArgsTest(self):
        print "TESTING BAD CALLS"
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        dico = mp_mdct_dico.Dico([128, 256, 512, 1024, 2048,
                                  4096, 8192, 16384])

        # testing mp with empty signal
        badargs = (None, dico, 50, 10)
        self.assertRaises(TypeError, mp.greedy, *badargs)

        # testing mp with empty dictionary
        badargs = (signal_original, None, 50, 10)
        self.assertRaises(TypeError, mp.greedy, *badargs)

        # testing mp with unknown update rule
        badargs = (signal_original, dico, 50, 10)
        badkwargs = { 'update':'nothing'}
        self.assertRaises(ValueError, mp.greedy, *badargs, **badkwargs)
        
        # testing mp
        # asburd call : should raise a ValueError
        badargs = (signals.Signal(np.zeros(signal_original.data.shape)),
                   dico, 50, 10)

        self.assertRaises(ValueError, mp.greedy, *badargs)
        print "------ OK -------"
    
    def runTest(self):
        
        self.badArgsTest()
        
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        
        dico = mp_mdct_dico.Dico([128, 1024, 8192], debug_level=0)
        print "Comparing with legacy implementation"
        n_atoms = 300
        app_1 , dec1 = mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=True, update='mp')
        app_2 , dec2 = mp.mp(signal_original, dico, 100, n_atoms ,debug=0, pad=False)
        np.testing.assert_almost_equal(dec1, dec2)
        for i in range(n_atoms):
            self.assertEquals(app_1[i],app_2[i])
        
        print "Running MP, OMP and local versions on synthetic k-sparse"
        dico = mp_mdct_dico.LODico([16, 64])
        L = 16*300
        data = np.zeros(L,)
        # create a 2*k-sparse signal
        K = 50
        for k in range(K):
            
            at = mdct_atom.Atom(16, 1,int((L-64)*np.random.rand(1)[0]),
                                       int(8*np.random.rand(1)[0]),
                                       Fs=8000, mdctCoeff=np.random.rand(1)[0])
            at.synthesize()
            data[at.time_position: at.time_position + at.length] += at.waveform
            at = mdct_atom.Atom(64, 1, int((L-64)*np.random.rand(1)[0]),
                                       freqBin=32*np.random.rand(1)[0],
                                       Fs=8000, mdctCoeff=np.random.rand(1)[0])
            at.synthesize()
            data[at.time_position: at.time_position + at.length] += at.waveform
        
        
        
        signal_original = signals.Signal(data,Fs=8000, mono=True, normalize=True)
        signal_original.data += 0.01*np.random.random(L,)

        
        n_atoms = 2*K
        cProfile.runctx('mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=True, update=\'mp\')', globals(), locals())
        cProfile.runctx('mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=False, update=\'locgp\')', globals(), locals())
        cProfile.runctx('mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=False, update=\'locomp\')', globals(), locals())
        cProfile.runctx('mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=False, update=\'omp\')', globals(), locals())
        
        app_1 , dec1 = mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=True, update='mp')
        app_2 , dec2 = mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=False, update='locgp')
        app_3 , dec3 = mp.greedy(signal_original, dico, 100, n_atoms ,debug=0, pad=False, update='omp')
        
        plt.figure()
        plt.plot(10.0*np.log10(dec1/dec1[0]))
        plt.plot(10.0*np.log10(dec2/dec2[0]))
        plt.plot(10.0*np.log10(dec3/dec3[0]))
        plt.legend(('MP','LocGP','OMP'))
        plt.show()
        

class OMPTest(unittest.TestCase):
    def runTest(self):
# dico = Dico.Dico([2**l for l in range(7,15,1)] , Atom.transformType.MDCT)

        dico = mp_mdct_dico.LODico([256, 2048, 8192])
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
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
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        

        app_mp, dec_mp = mp.mp(signal_original, dico, 10, 200 ,debug=0 ,pad=True, clean=True)
        app_locomp, dec_locomp = mp.locomp(signal_original, dico, 10, 200 ,debug=0 ,pad=False, clean=True)
        
        #  Ok test also the local GP implementation
        app_locgp, dec_locgp = mp.locomp(signal_original, dico, 10, 200 ,debug=0
                                           ,pad=False, approximate=True)
        
        print app_mp
        print app_locomp
        print app_locgp        
        
        
        plt.figure()    
        plt.plot(10.0*np.log10(dec_mp/dec_mp[0]),'b')
        plt.plot(10.0*np.log10(dec_locomp/dec_locomp[0]),'r--')
        plt.plot(10.0*np.log10(dec_locgp/dec_locgp[0]),'k-.')     
        plt.legend(('MP','LocOMP','LocGP'))   
#        plt.show()
        
        
        self.assertGreater(dec_mp[-1], dec_locomp[-1])
        self.assertGreater(dec_mp[-1], dec_locgp[-1])
        self.assertEqual(int(dec_locgp[-1]),int(dec_locomp[-1]))
        
        # profiling test
#        print "Plain"
#        cProfile.runctx('', globals(), locals())
#
#        cProfile.runctx('mp.mp(signal_original, dico2, 20, 1000 ,debug=0 , '
#                        'clean=True)', globals(), locals())


class SequenceDicoTest(unittest.TestCase):

    def runTest(self):
# dico = Dico.Dico([2**l for l in range(7,15,1)] , Atom.transformType.MDCT)

        dico = random_dico.SequenceDico(
            [256, 2048, 8192], seq_type='random', seed=1001)
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        signal_original.crop(0, 5 * 16384)

        signal_original.data += 0.01 * np.random.random(5 * 16384)

        cProfile.runctx('mp.mp(signal_original, dico, 20, 20, debug=1, '
                        'clean=True)', globals(), locals())

        # Comparing with MP and a fixed sequence
        n_atoms = 10
        vary_dico = random_dico.SequenceDico(
            [256, 2048, 8192], seq_type='random')

        fake_fix_dico = random_dico.SequenceDico(
            [256, 2048, 8192], seq_type='random', nbSame=n_atoms)

        real_fix_dico = mp_mdct_dico.Dico([256, 2048, 8192])
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        signal_original.crop(0, 5 * 16384)

        signal_original.data += 0.01 * np.random.random(5 * 16384)

        app, dec = mp.mp(signal_original, vary_dico, 20, 300, debug=0)
        app_fake_fix, fake_fix_dec = mp.mp(
            signal_original, fake_fix_dico, 20, n_atoms, debug=0, pad=True)
        print [b.shift_list[0:5] for b in dico.blocks]
        print [b.shift_list[0:5] for b in fake_fix_dico.blocks]
        print [b.shift_list[0:5] for b in vary_dico.blocks]
        app_real_fix, dec_real_fix = mp.mp(
            signal_original, real_fix_dico, 20, n_atoms, debug=0, pad=True)

#        plt.figure()
#        plt.plot(dec)
#        plt.plot(fake_fix_dec,'-.')
#        plt.plot(dec_real_fix,':')
#        plt.show()


class ApproxTest(unittest.TestCase):

    def runTest(self):

        app = approx.Approx(debug_level=2)
        self.assertEqual(app.original_signal, None)
        self.assertEqual(app.atom_number, 0)
        self.assertEqual(app.srr, 0)
        self.assertEqual(app.atoms, [])

        print app
        del app

        dico = mp_mdct_dico.Dico([2 ** l for l in range(7, 15, 1)])
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         mono=True)
        signal_original.crop(0, 5 * max(dico.sizes))

        app = approx.Approx(dico, [], signal_original)

        pyAtom = mdct_atom.Atom(1024, 1, 12288 - 256, 128, 44100, 0.57)
        app.add(pyAtom)
        self.assertEqual(app.atom_number, 1)

        # testing the remove method
        app.remove(pyAtom)
        self.assertEqual(app.atom_number, 0)

        app.add(pyAtom)

        app.add(
            mdct_atom.Atom(8192, 1, 4096 - 2048, 32, 44100, -0.24))
        approxSignal1 = app.synthesize(0)

        # testing the update method
        app_update = approx.Approx(dico, [], signal_original)
        app_update.add(
            mdct_atom.Atom(8192, 1, 4096 - 2048, 32, 44100, -0.24))
        old_array = app_update.synthesize(0).data.copy()
        
        self.assertAlmostEqual(np.sqrt(np.sum(old_array**2)), 0.24)
        
        app_update.update([0], [0.80])
        new_array = app_update.synthesize(0).data.copy()
        
        self.assertAlmostEqual(np.sqrt(np.sum(new_array**2)), 0.80)
        
#        plt.figure()
#        plt.plot(old_array)
#        plt.plot(new_array)
#        plt.show()

        plt.figure()
        plt.subplot(121)
        app.plot_tf()
        plt.subplot(122)
        app.plot_tf(multicolor=True, keepValues=True)
#        approxSignal1.plot()

#        del approxSignal
        approxSignal2 = app.synthesize(1)
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

        # test last case, with LOMP atoms
        approxSignal3 = app.synthesize(2)

        # testing filtering
        self.assertEqual(
            pyAtom, app.filter_atoms(1024, None, None).atoms[0])
        self.assertEqual(pyAtom, app.filter_atoms(1024, [
            12000, 15000], None).atoms[0])

        self.assertAlmostEqual(app.compute_srr(), -114.057441323)
        # TODO testing du SRR

        # testing the write_to_xml and read_from_xml methods
        pyCCDico = mp_mdct_dico.LODico([256, 2048, 8192])
        approx_LOmp = mp.mp(signal_original, pyCCDico, 10, 100, False)[0]

        sliceApprox = approx_LOmp[:10]

        for i in range(10):
            self.assertEqual(approx_LOmp.atoms[i], sliceApprox.atoms[i])

        self.assertEqual(approx_LOmp[0], sliceApprox[0])

        print approx_LOmp
        print sliceApprox
        sliceApprox.compute_srr()
        print sliceApprox

        plt.figure()
        plt.subplot(121)
        approx_LOmp.plot_tf()
        plt.subplot(122)
        sliceApprox.plot_tf(french=True)
        plt.plot()

        # testing all the plot options
        plt.figure()
        sliceApprox.plot_3d(itStep=1)

        del signal_original

        self.neighborTesting()

        self.ioTesting()
    
    def neighborTesting(self):
        dico = mp_mdct_dico.Dico([2 ** l for l in range(7, 15, 1)])
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         mono=True)
        app = approx.Approx(dico, [], signal_original)

        # fill with two atoms in 10000 and 12000
        app.add(mdct_atom.Atom(1024, amp=1, timePos=10000,
                               freqBin=12, Fs=8000, mdctCoeff=0.57))
        
        app.add(mdct_atom.Atom(1024, amp=1, timePos=12000,
                               freqBin=12, Fs=8000, mdctCoeff=0.57))
        
        app.add(mdct_atom.Atom(1024, amp=1, timePos=9000,
                               freqBin=12, Fs=8000, mdctCoeff=0.57))
        
        # now look for neighbors:
        two_neigh = app.get_neighbors(mdct_atom.Atom(1024, amp=1, timePos=11000,
                                               freqBin=12, Fs=8000, mdctCoeff=0.57))
        
        
        self.assertEqual(app.get_neighbors(mdct_atom.Atom(1024, 1, 11000,12, 8000, 0.57)),
                         [0,1])
        self.assertEqual(app.get_neighbors(mdct_atom.Atom(1024, 1, 11025,12, 8000, 0.57)),
                         [1])        
        self.assertEqual(app.get_neighbors(mdct_atom.Atom(1024, 1, 8500,12, 8000, 0.57)),
                         [2])
        self.assertEqual(app.get_neighbors(mdct_atom.Atom(8192, 1, 7000,12, 8000, 0.57)),
                         [0,1,2])

    def ioTesting(self):
        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True,
                                         debug_level=0)
        signal_original.crop(0, 1 * 16384)
        dico = mp_mdct_dico.Dico([256, 2048, 8192], debug_level=2)
        # first compute an approximant using mp
        approximant = mp.mp(signal_original, dico, 10, 10, debug=2)[0]

        output_dump_path = os.path.abspath("approx_dump_test.pymp")

        print "TESTING dump and load "
        approximant.dump(output_dump_path)

        # Test reading from the dumped file
        new_approx = approx.load(output_dump_path)
        self.assertEqual(new_approx.dico.sizes, approximant.dico.sizes)
        self.assertEqual(new_approx.atom_number, approximant.atom_number)
        self.assertEqual(new_approx.length, approximant.length)

        # now the hardest test
        new_approx.synthesize()
#        new_approx.recomposed_signal.plot()
        self.assertAlmostEquals(sum(approximant.recomposed_signal.
                                    data - new_approx.recomposed_signal.data), 0)

        del new_approx

        # test writing with LOmp atoms
        lomp_dico = mp_mdct_dico.LODico([256, 2048, 8192])
        approx_LOmp = mp.mp(signal_original, lomp_dico, 10, 100, False)[0]

        output_lodump_path = os.path.abspath("approx_lodump_test.pymp")
        approx_LOmp.dump(output_lodump_path)

        # Test reading from the xml flow
        new_approx = approx.load(output_lodump_path)
        self.assertEqual(new_approx.dico.sizes, approx_LOmp.dico.sizes)
        self.assertEqual(new_approx.atom_number, approx_LOmp.atom_number)
        self.assertEqual(new_approx.length, approx_LOmp.length)
        self.assertAlmostEqual(
            sum(new_approx.to_array()[0] - approx_LOmp.to_array()[0]), 0)

        plt.figure()
        plt.plot(new_approx.to_array()[0])
        plt.plot(approx_LOmp.to_array()[0], 'r:')
#        plt.show()

        print "TESTING: exports "
        print new_approx.to_array()
        print new_approx.to_dico()
        print new_approx.to_sparse_array()

        del new_approx


class LOMPTest(unittest.TestCase):
    def runTest(self):

        lodico = mp_mdct_dico.LODico([256, 2048, 8192])
        dico = mp_mdct_dico.Dico([256, 2048, 8192])

        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        signal_original.crop(0, 1 * 16384)

        # intentionally non aligned atom -> objective is to fit it completely
        # through time shift calculation
        pyAtom = mdct_atom.Atom(2048, 1, 8192 - 512, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        plt.figure()
        print "Testing one Aligned Atom with LOmp - should be perfect"
        approximant = mp.mp(signal_one_atom, lodico, 10, 10, debug=2)[0]
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

        print " Approx Reached : %1.3f dB in %d  it" % (approximant.compute_srr(),
                                                     approximant.atom_number)

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
        print " Approx Reached : %1.3f dB in %d  it" % (approximant.compute_srr(),
                                                     approximant.atom_number)

        del approximant

        print "Testing one Non-Aligned Atom with LOmp - should be almost perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        approximant = mp.mp(
            signal_one_atom, lodico, 10, 10, False, False)[0]
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
        print " Approx Reached : %1.3f dB in %d  it" % (approximant.compute_srr(),
                                                     approximant.atom_number)

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
        print " Approx Reached : %1.3f dB in %d it" % (approximant.compute_srr(),
                                                     approximant.atom_number)

        del approximant

        print "Testing Real signals with LOmp"
#        lodico =
        approximant = mp.mp(signal_original, lodico, 10, 10, False, False)[0]
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
        print " Approx Reached : %1.3f dB in %d it" % (approximant.compute_srr(),
                                                     approximant.atom_number)

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
        print " Approx Reached : %1.3f dB in %d iteration and %1.3f seconds" % (
                                                     approx1.compute_srr(),
                                                     approx1.atom_number,
                                                     t1-t0)

        t2 = time.clock()
        approx2 = mp.mp(signal_original, lodico, 20, 500, False, False)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(signal_original.data[12288:-12288])
        plt.plot(approx2.recomposed_signal.data[12288:-12288])
        plt.plot(signal_original.data[12288:-12288] - approx2.
                 recomposed_signal.data[12288:-12288])
        plt.legend(("original", "approximant", "residual"))
        plt.title("Bell signals with LOmp : SRR of " + str(int(approx2.compute_srr())) + " dB in " + str(approx2.atom_number) + " iteration and " + str(t3 - t2) + "s")
#        plt.savefig(figPath+"RealSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : %1.3f dB in %d iteration and %1.3f seconds" % (
                                                     approx2.compute_srr(),
                                                     approx2.atom_number,
                                                     t3-t2)

        del approx1, approx2
        del signal_original
        print "comparing results and processing times for long decompositions of white gaussian noise"
        noise_signal = signals.Signal(
            0.5 * np.random.random(5 * 16384), 44100, False)
        noise_signal.pad(16384)
        t0 = time.clock()
        approx1 = mp.mp(noise_signal, dico, 10, 500, False, True)[0]
        t1 = time.clock()
#        plt.figure()
#        plt.subplot(211)
#        plt.plot(noise_signal.data[16384:-16384])
#        plt.plot(approx1.recomposed_signal.data[16384:-16384])
#        plt.plot(noise_signal.data[16384:-16384] - approx1.
#                 recomposed_signal.data[16384:-16384])
#        plt.legend(("original", "approximant", "residual"))
#        plt.title("White Noise signals with mp : SRR of " + str(int(approx1.compute_srr())) + " dB in " + str(approx1.atom_number) + " iteration and " + str(t1 - t0) + "s")
#        plt.show()
        print " Approx Reached : %1.3f dB in %d iteration and %1.3f seconds" % (
                                                     approx1.compute_srr(),
                                                     approx1.atom_number,
                                                     t1-t0)

        t2 = time.clock()
        approx2 = mp.mp(noise_signal, lodico, 10, 500, False, True)[0]
        t3 = time.clock()
#        plt.subplot(212)
#        plt.plot(noise_signal.data[16384:-16384])
#        plt.plot(approx2.recomposed_signal.data[16384:-16384])
#        plt.plot(noise_signal.data[16384:-16384] - approx2.
#                 recomposed_signal.data[16384:-16384])
#        plt.legend(("original", "approximant", "residual"))
#        plt.title("Noise signals with LOmp : SRR of " +
#                  str(int(approx2.compute_srr())) + " dB in "
#                  + str(approx2.atom_number) + " iteration and "
#                  + str(t3 - t2) + "s")
#        plt.xlabel("samples")
#        plt.savefig(figPath+"NoiseSignal_mp_vs_LOmp_SRR20"+ext)
        print " Approx Reached : %1.3f dB in %d iteration and %1.3f seconds" % (
                                                     approx2.compute_srr(),
                                                     approx2.atom_number,
                                                     t3-t2)


class SSMPTest(unittest.TestCase):
    def runTest(self):

#        ext = ".png"

        seq_dico = random_dico.SequenceDico([256, 2048, 8192], 'scale')
        dico = mp_mdct_dico.Dico([256, 2048, 8192])

        signal_original = signals.Signal(op.join(audio_filepath, "ClocheB.wav"),
                                         normalize=True, mono=True)
        signal_original.crop(0, 1 * 16384)

        # intentionally non aligned atom -> objective is to fit it completely
        # through time shift calculation
        pyAtom = mdct_atom.Atom(2048, 1, 8192 - 512, 128, 8000, 0.57)
        pyAtom.synthesize()
        np.random.seed(42)
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        print "Testing one Aligned Atom with sequence dico"
        approximant = mp.mp(
            signal_one_atom, seq_dico, 10, 10, False, False)[0]

        self.assertAlmostEqual(approximant.compute_srr(), 37.78735640946547)
        del pyAtom, approximant, signal_one_atom

        print "Testing one Non-Aligned Atom with mp - should not be perfect"
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        approximant, decay = mp.mp(
            signal_one_atom, dico, 10, 10, False, False)

        self.assertAlmostEqual(approximant.compute_srr(), 11.0394466910923, 1)
        del approximant

        print "Testing one Non-Aligned Atom with Random (reproductible)"
        seed = 1001
        rand_dico = random_dico.SequenceDico(
            [256, 2048, 8192], 'random', seed=seed)
        pyAtom = mdct_atom.Atom(2048, 1, 7500, 128, 8000, 0.57)
        pyAtom.synthesize()
        np.random.seed(seed)
        signal_one_atom = signals.Signal(0.0001 * np.random.random(
            signal_original.length), signal_original.fs, False)
        signal_one_atom.add(pyAtom)

        approximant, decay = mp.mp(
            signal_one_atom, rand_dico, 10, 10, False, False)

        self.assertAlmostEqual(approximant.srr, 10.970629166418217)

        del approximant

        print "Testing Real signals with Randommp"
#        pyCCDico =
        seed = 42
        rand_dico = random_dico.SequenceDico(
            [2 ** j for j in range(7, 15)], 'random', seed=seed)
        approximant = mp.mp(
            signal_original, rand_dico, 10, 10, False, False)[0]

        self.assertEqual(approximant.atom_number, 10)
        self.assertAlmostEqual(approximant.srr, 0.6242965242229909)

        del approximant
        print "comparing results and processing times for long decompositions"
        t0 = time.clock()
        approx1 = mp.mp(signal_original, dico, 20, 500, False)[0]
        t1 = time.clock()

        stri = 'Approx Reached %1.3f dB in %d iterations and %1.3f sec ' % (
            approx1.compute_srr(), approx1.atom_number, t1 - t0)
        print stri
        self.assertAlmostEqual(approx1.srr, 19.779705511708805)
        self.assertEqual(approx1.atom_number, 500)

        t2 = time.clock()
        approx2 = mp.mp(signal_original, rand_dico, 20, 500, False, False)[0]
        t3 = time.clock()

        stri = " Approx Reached %1.3f dB in %d iterations and %1.3f sec " % (
            approx2.compute_srr(), approx2.atom_number, t3 - t2)
        print stri
        self.assertAlmostEqual(approx2.srr, 20.02389209197687)
        self.assertEqual(approx2.atom_number, 271)

        del approx1, approx2
        del signal_original
        # "comparing results and processing times for long decompositions 
        # of white gaussian noise"
        noise_signal = signals.Signal(
            0.5 * np.random.random(5 * 16384), 44100, False)
        noise_signal.pad(16384)
        t0 = time.clock()
        approx1 = mp.mp(noise_signal, dico, 10, 500, False, True)[0]
        t1 = time.clock()

        stri = " Approx Reached %1.3f dB in %d iterations and %1.3f sec " % (
            approx1.compute_srr(), approx1.atom_number, t1 - t0)
        print stri

        rand_dico = random_dico.SequenceDico(
            [256, 2048, 8192], 'random', seed=seed)
        t2 = time.clock()
        approx2 = mp.mp(noise_signal, rand_dico, 10, 500, False, True)[0]
        t3 = time.clock()

        stri = 'Approx Reached %1.3f dB in %d iterations and %1.3f sec' % (
            approx2.compute_srr(), approx2.atom_number, t3 - t2)
        print stri


class MPlongTest(unittest.TestCase):
    """ this time we decompose a longer signals with the mp_LongSignal :
    result in an enframed signals """
    def runTest(self):
        filePath = op.join(audio_filepath, "Bach_prelude_4s.wav")

        # Let us load a long signals: not loaded in memory for now, only
        # segment by segment in the mp process
        mdctDico = [256, 1024, 8192]
        frameSize = 5 * 8192

        # signals buidling
        original_signal = signals.LongSignal(filePath, frameSize, True)

        # dictionaries
        pyCCDico = mp_mdct_dico.LODico(mdctDico)

        # Let's feed the proto 3 with these:
        # we should get a collection of approximants (one for each frame) in
        # return
        # xmlOutPutDir = op.join(op.dirname(__file__), '..', '..',
        # '/Approxs/Bach_prelude/LOmp/')
        xmlOutPutDir = '.'
        approximants = mp.mp_long(
            original_signal, pyCCDico, 10, 100, False, True, xmlOutPutDir)[0]

        self.assertEqual(len(approximants), original_signal.n_seg)

        fusionned_approx = approx.fusion_approxs(approximants)

        print fusionned_approx
        print approximants

        self.assertEqual(fusionned_approx.fs, original_signal.fs)
#        self.assertEqual(fusionned_approx.length, original_signal.length )
        plt.figure
        fusionned_approx.plot_tf()
#        plt.show()


class WinServerTest(unittest.TestCase):

    def runTest(self):
        server1 = win_server.get_server()
        print "Server first call"
        cProfile.runctx('server1.get_waveform(4096, 2001)', globals(),
                        locals())
        print "Server second call should NOT call parallelProjections.get_atom"
        cProfile.runctx('server1.get_waveform(4096, 2001)', globals(),
                        locals())

        # try to instantiate a new one: should not call
        # PyMP.parallelProjections.get_atom
        server2 = win_server.get_server()
        print "Server third call should NOT call parallelProjections.get_atom "
        cProfile.runctx('server2.get_waveform(4096, 2001)', globals(),
                        locals())

        # final test, create a block a get the waveform from it (instantiation
        # from another scope
        block = Block(4096, np.random.random((8000,)))
        block.max_bin_idx = 2001
        print "Fourth third call should NOT call parallelProjections.get_atom "
        cProfile.runctx('block.synthesize_atom()', globals(), locals())


if __name__ == '__main__':
    import matplotlib
    print matplotlib.__version__

    _Logger = log.Log('test', level=3, imode=False)
    _Logger.info('Starting Tests')
    suite = unittest.TestSuite()

#    suite.addTest(MPlongTest())
#    suite.addTest(MPTest())
#    suite.addTest(OMPTest())
    suite.addTest(GreedyTest())
#    suite.addTest(SequenceDicoTest())
#    suite.addTest(SSMPTest())
#    suite.addTest(LOMPTest())
#    suite.addTest(ApproxTest())
#    suite.addTest(AtomTest())
#    suite.addTest(DicoTest())
#    suite.addTest(BlockTest())
#    suite.addTest(WinServerTest())
#    suite.addTest(Signaltest())
#    suite.addTest(WaveletAtomTest())
#
    unittest.TextTestRunner(verbosity=2).run(suite)

#    plt.show()
    _Logger.info('Tests stopped')
