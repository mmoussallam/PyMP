# TODO refactoring, until then its DEPRECATED


'''
Created on Sep 13, 2011

@author: moussall
'''
import unittest

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
from PyMP import Signal, mp
from PyMP import parallelProjections
from PyMP.mdct import dico as mdct_dico
from PyMP.mdct import block as mdct_block

audio_filepath = op.join(op.dirname(__file__), '..', '..', 'data')
# audio_filepath = '/sons/sqam/'


class blocksTest(unittest.TestCase):

    def runTest(self):
        pySig = Signal(op.join(audio_filepath, 'glocs.wav'), mono=True)
        pySig.crop(0, 5 * pySig.fs)
        pySig.pad(2048)

        scale = 1024
        parallelProjections.initialize_plans(
            np.array([scale, ]), np.array([2, ]))

        classicBlock = mdct_block.Block(scale, pySig, 0,
                                        debug_level=3)

        spreadBlock = mdct_block.SpreadBlock(scale, pySig, 0,
                                             debug_level=3,
                                             penalty=0, maskSize=5)

        # compute the projections, should be equivalent
        classicBlock.update(pySig, 0, -1)
        spreadBlock.update(pySig, 0, -1)

        maxClassicAtom1 = classicBlock.get_max_atom()
        print maxClassicAtom1.length, maxClassicAtom1.frame,
        print maxClassicAtom1.freq_bin, maxClassicAtom1.mdct_value
        maxSpreadcAtom1 = spreadBlock.get_max_atom()
        print maxSpreadcAtom1.length, maxSpreadcAtom1.frame,
        print maxSpreadcAtom1.freq_bin, maxSpreadcAtom1.mdct_value
        # assert equality using the inner comparison method of MDCT atoms
        self.assertEqual(maxClassicAtom1, maxSpreadcAtom1)

        # verifying the masking index construction
        mask_frame_width = 2
        mask_bin_width = 1
        spreadBlock.compute_mask(
            maxSpreadcAtom1, mask_bin_width, mask_frame_width, 0.5)

        c_frame = int(np.ceil(maxSpreadcAtom1.time_position / (scale / 2)))
        c_bin = int(maxSpreadcAtom1.reduced_frequency * scale)

        z1 = np.arange(int(
            c_frame - mask_frame_width), int(c_frame + mask_frame_width) + 1)
        z2 = np.arange(
            int(c_bin - mask_bin_width), int(c_bin + mask_bin_width) + 1)
#        x, y = np.meshgrid(z1, z2)
#        print spreadBlock.mask_index_x
#        np.testing.assert_array_equal(spreadBlock.mask_index_x, z1)
#        np.testing.assert_array_equal(spreadBlock.mask_index_y, z2)

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

        classicDIco = mdct_dico.Dico(dico)
        spreadDico = mdct_dico.SpreadDico(dico,
                                          all_scales=True,
                                          penalty=0, maskSize=3)

        self.assertEqual(spreadDico.mask_times, [3, 3, 3])

        classicDIco.initialize(pySig)
        spreadDico.initialize(pySig)

        classicDIco.update(pySig, 2)
        spreadDico.update(pySig, 2)

        classicAtom1 = classicDIco.get_best_atom(0)
        spreadAtom1 = spreadDico.get_best_atom(0)
#        print classicAtom1, spreadAtom1
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
        pySig = Signal(
            op.join(audio_filepath, 'glocs.wav'), mono=True, normalize=True)
        pySig.crop(0, 5 * pySig.fs)
        pySig.pad(16384)
        sigEnergy = np.sum(pySig.data ** 2)
        dico = [128, 1024, 8192]
        nbAtoms = 200

        classicDIco = mdct_dico.Dico(dico, useC=False)
        spreadDico = mdct_dico.SpreadDico(
            dico, all_scales=False, spread_scales=[1024, 8192],
            penalty=0.1, mask_time=2, mask_freq=2)

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

        plt.figure()
        for blockI in range(1, 3):
            block = spreadDico.blocks[blockI]
            plt.subplot(2, 2, blockI)
            print block.mask.shape, block.mask.shape[0] / (block.scale / 2), block.scale / 2
            plt.imshow(
                np.reshape(block.mask, (
                    block.mask.shape[0] / (block.scale / 2), block.scale / 2)),
                interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.subplot(2, 2, blockI + 2)
# print block.mask.shape, block.mask.shape[0] / (block.scale/2),
# block.scale/2
            block.im_proj_matrix()
            plt.colorbar()


class realTest2(unittest.TestCase):

    def runTest(self):
        name = 'orchestra'
        pySig = Signal(op.join(
            audio_filepath, 'Bach_prelude_40s.wav'), mono=True, normalize=True)
        pySig.crop(0, 5 * pySig.fs)
        pySig.pad(16384)
        sigEnergy = np.sum(pySig.data ** 2)
        dico = [128, 1024, 8192]
        nbAtoms = 200

        classicDIco = mdct_dico.Dico(dico)
        spreadDico = mdct_dico.SpreadDico(
            dico, all_scales=True, penalty=0.1, maskSize=10)

        approxClassic, decayClassic = mp.mp(pySig, classicDIco, 20, nbAtoms)
        approxSpread, decaySpread = mp.mp(
            pySig, spreadDico, 20, nbAtoms, pad=False)
        import matplotlib.pyplot as plt
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


class coherenceTest(unittest.TestCase):
    
    def runTest(self):
        dico = [128, 1024, 8192]
        nbAtoms = 100
        pySig = Signal(op.join(audio_filepath,
                               'Bach_prelude_40s.wav'),
                       mono=True, normalize=True)
        classicDIco = mdct_dico.Dico(dico)
        spreadDico = mdct_dico.SpreadDico(
            dico, all_scales=True, penalty=0, maskSize=10)
        
        import time
        t = time.time()
        app_mp , _ = mp.mp(pySig, classicDIco, 20, nbAtoms)
        print "Classic took %1.3f sec"%(time.time()-t)
        t = time.time()
        app_spreadmp , _ = mp.mp(pySig, spreadDico, 20, nbAtoms)
        print "Spread took %1.3f sec"%(time.time()-t)
        
        
        plt.figure()
        plt.subplot(121)
        app_mp.plot_tf()
        plt.subplot(122)
        app_spreadmp.plot_tf()
        plt.show()

class perfTest(unittest.TestCase):

    def runTest(self):
        dico = [128, 1024, 8192]
        nbAtoms = 100
        pySig = Signal(op.join(audio_filepath,
                               'Bach_prelude_40s.wav'),
                       mono=True, normalize=True)
        classicDIco = mdct_dico.Dico(dico)
        spreadDico = mdct_dico.SpreadDico(
            dico, all_scales=True, penalty=0.1, maskSize=10)

        import cProfile
        cProfile.runctx(
            'mp.mp(pySig, classicDIco, 20, nbAtoms)', globals(), locals())
        cProfile.runctx('mp.mp(pySig, spreadDico, 20, nbAtoms, pad=False)',
                        globals(), locals())


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    import matplotlib.pyplot as plt
    suite = unittest.TestSuite()

#    suite.addTest(blocksTest())
#    suite.addTest(dicosTest())
#    suite.addTest(realTest())
#    suite.addTest(realTest2())
    suite.addTest(coherenceTest())
#    suite.addTest(perfTest())
#
    unittest.TextTestRunner(verbosity=2).run(suite)
#    plt.show()
