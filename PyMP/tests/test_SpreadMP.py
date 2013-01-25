# TODO refactoring, until then its DEPRECATED


'''
Created on Sep 13, 2011

@author: moussall
'''
import unittest


import matplotlib.pyplot as plt
import numpy as np
import os.path as op
from PyMP import Signal, mp
from PyMP import parallelProjections
from PyMP.mdct import dico as mdct_dico
from PyMP.mdct import block as mdct_block

audio_filepath= op.join(op.dirname(__file__), '..', '..', 'data')
#audio_filepath = '/sons/sqam/'


class blocksTest(unittest.TestCase):

    def runTest(self):
        pySig = Signal(op.join(audio_filepath, 'glocs.wav'), mono=True)
        pySig.crop(0, 5 * pySig.fs)
        pySig.pad(2048)

        scale = 1024
        parallelProjections.initialize_plans(
            np.array([scale, ]), np.array([2, ]))

        classicBlock = mdct_block.Block(scale, pySig, 0,
                                        debug_level=3, useC=False)

        spreadBlock = mdct_block.SpreadBlock(scale, pySig, 0,
                                             debug_level=3, useC=False,
                                             penalty=0, maskSize=5)

        # compute the projections, should be equivalent
        classicBlock.update(pySig, 0, -1)
        spreadBlock.update(pySig, 0, -1)

        maxClassicAtom1 = classicBlock.get_max_atom()
        print maxClassicAtom1.length, maxClassicAtom1.frame, maxClassicAtom1.freq_bin, maxClassicAtom1.mdct_value
        maxSpreadcAtom1 = spreadBlock.get_max_atom()
        print maxSpreadcAtom1.length, maxSpreadcAtom1.frame, maxSpreadcAtom1.freq_bin, maxSpreadcAtom1.mdct_value
        # assert equality using the inner comparison method of MDCT atoms
        self.assertEqual(maxClassicAtom1, maxSpreadcAtom1)

        pySig.subtract(maxSpreadcAtom1)

        # recompute the projections
        classicBlock.update(pySig, 0, -1)
        spreadBlock.update(pySig, 0, -1)

#        plt.show()
        maxClassicAtom2 = classicBlock.get_max_atom()
        print maxClassicAtom2.length, maxClassicAtom2.frame, maxClassicAtom2.freq_bin, maxClassicAtom2.mdct_value
        maxSpreadcAtom2 = spreadBlock.get_max_atom()
        print maxSpreadcAtom2.length, maxSpreadcAtom2.frame, maxSpreadcAtom2.freq_bin, maxSpreadcAtom2.mdct_value
        self.assertNotEqual(maxClassicAtom2, maxSpreadcAtom2)

        parallelProjections.clean_plans()


class dicosTest(unittest.TestCase):

    def runTest(self):
        # create a SpreadDico
        pySig = Signal(op.join(audio_filepath, 'glocs.wav'), mono=True)
        pySig.crop(0, 5 * pySig.fs)
        pySig.pad(2048)

        dico = [128, 1024, 8192]

        parallelProjections.initialize_plans(
            np.array(dico), np.array([2] * len(dico)))

        classicDIco = mdct_dico.Dico(dico, useC=False)
        spreadDico = mdct_dico.SpreadDico(dico, useC=False,
                                          allBases=True,
                                          penalty=0, maskSize=3)

        classicDIco.initialize(pySig)
        spreadDico.initialize(pySig)

        classicDIco.update(pySig, 2)
        spreadDico.update(pySig, 2)

        classicAtom1 = classicDIco.get_best_atom(0)
        spreadAtom1 = spreadDico.get_best_atom(0)

        self.assertEqual(classicAtom1, spreadAtom1)

        pySig.subtract(classicAtom1)
        classicDIco.update(pySig, 2)
        spreadDico.update(pySig, 2)

        classicAtom2 = classicDIco.get_best_atom(0)
        spreadAtom2 = spreadDico.get_best_atom(0)

        self.assertNotEqual(classicAtom2, spreadAtom2)


class realTest(unittest.TestCase):

    def runTest(self):
        name = 'orchestra'
        pySig = Signal(op.join(audio_filepath, 'glocs.wav'), mono=True, normalize=True)
        pySig.crop(0, 5 * pySig.fs)
        pySig.pad(16384)
        sigEnergy = np.sum(pySig.data ** 2)
        dico = [128, 1024, 8192]
        nbAtoms = 200

        classicDIco = mdct_dico.Dico(dico, useC=False)
        spreadDico = mdct_dico.SpreadDico(
            dico, useC=False, allBases=False, Spreadbases=[1024, 8192],
            penalty=0.1, maskSize=3)

        approxClassic, decayClassic = mp.mp(pySig, classicDIco, 20, nbAtoms)
        approxSpread, decaySpread = mp.mp(
            pySig, spreadDico, 20, nbAtoms, pad=False)

        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        approxClassic.plot_tf(ylim=[0, 4000])
        plt.title('Classic decomposition : 200 atoms 3xMDCT')
        plt.subplot(122)
        approxSpread.plot_tf(ylim=[0, 4000])
        plt.title('Decomposition with TF masking: 200 atoms 3xMDCT')
#        plt.savefig(name + '_TestTFMasking.eps')

        plt.figure()
        plt.plot([10 * np.log10(i / sigEnergy) for i in decayClassic])
        plt.plot([10 * np.log10(i / sigEnergy) for i in decaySpread], 'r')
        plt.legend(('Classic decomposition', 'Spreading Atoms'))
        plt.ylabel('Residual energy decay(dB)')
        plt.xlabel('Iteration')
#        plt.savefig(name + '_decayTFMasking.eps')


class realTest2(unittest.TestCase):

    def runTest(self):
        name = 'orchestra'
        pySig = Signal(op.join(audio_filepath, 'Bach_prelude_40s.wav'), mono=True, normalize=True)
        pySig.crop(0, 5 * pySig.fs)
        pySig.pad(16384)
        sigEnergy = np.sum(pySig.data ** 2)
        dico = [128, 1024, 8192]
        nbAtoms = 200

        classicDIco = mdct_dico.Dico(dico, useC=False)
        spreadDico = mdct_dico.SpreadDico(
            dico, useC=False, allBases=True, penalty=0.1, maskSize=10)

        approxClassic, decayClassic = mp.mp(pySig, classicDIco, 20, nbAtoms)
        approxSpread, decaySpread = mp.mp(
            pySig, spreadDico, 20, nbAtoms, pad=False)

        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        approxClassic.plot_tf(ylim=[0, 4000])
        plt.title('Classic decomposition : 200 atoms 3xMDCT')
        plt.subplot(122)
        approxSpread.plot_tf(ylim=[0, 4000])
        plt.title('Decomposition with TF masking: 200 atoms 3xMDCT')
#        plt.savefig(name + '_TestTFMasking.eps')

        plt.figure()
        plt.plot([10 * np.log10(i / sigEnergy) for i in decayClassic])
        plt.plot([10 * np.log10(i / sigEnergy) for i in decaySpread], 'r')
        plt.legend(('Classic decomposition', 'Spreading Atoms'))
        plt.ylabel('Residual energy decay(dB)')
        plt.xlabel('Iteration')
#        plt.savefig(name + '_decayTFMasking.eps')
        
        
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    suite = unittest.TestSuite()

    suite.addTest(blocksTest())
    suite.addTest(dicosTest())
    suite.addTest(realTest())
    suite.addTest(realTest2())
#
    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()