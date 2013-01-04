'''
Created on Sep 13, 2011

@author: moussall
'''
import unittest
from MatchingPursuit.mdct import py_pursuit_MDCTBlock , py_pursuit_MDCTDico
from MatchingPursuit import py_pursuit_Signal , MP
import matplotlib.pyplot as plt
import numpy as np
class Test(unittest.TestCase):


    def testSpreadMP(self):
        ''' Spread MP is a technique in which you penalize the selection of atoms near existing ones
        in a  predefined number of iterations, or you specify a perceptual TF masking to enforce the selection of
        different features. The aim is to have maybe a loss in compressibility in the first iteration but
        to have the most characteristics / discriminant atoms be chosen in the first iterations '''

        # First test: create a SpreadMDCTBlock and a classic MDCTBlock and check that some
        # atom selections are penalized
        self.blocks();
        self.dicos();
        self.realTest();

    def blocks(self):
        pySig = py_pursuit_Signal.InitFromFile('../../../data/glocs.wav',forceMono=True);
        pySig.crop(0, 5*pySig.samplingFrequency);
        pySig.pad(2048)

        Scale = 1024;

        classicBlock = py_pursuit_MDCTBlock.py_pursuit_MDCTBlock(Scale,pySig, 0,
                                                                 debugLevel=3,useC=False)
        spreadBlock = py_pursuit_MDCTBlock.py_pursuit_SpreadBlock(Scale,pySig,0,
                                                                 debugLevel=3,useC=False,
                                                                 penalty=0, maskSize = 5)

        # compute the projections, should be equivalent
        classicBlock.update(pySig, 0, -1)
        spreadBlock.update(pySig, 0, -1)

        maxClassicAtom1 = classicBlock.getMaxAtom();
        print maxClassicAtom1.length , maxClassicAtom1.frame , maxClassicAtom1.freq_bin , maxClassicAtom1.mdct_value
        maxSpreadcAtom1 = spreadBlock.getMaxAtom();
        print maxSpreadcAtom1.length , maxSpreadcAtom1.frame , maxSpreadcAtom1.freq_bin, maxSpreadcAtom1.mdct_value
        # assert equality using the inner comparison method of MDCT atoms
        self.assertEqual(maxClassicAtom1 ,maxSpreadcAtom1 )

        pySig.subtract(maxSpreadcAtom1);

        # recompute the projections
        classicBlock.update(pySig, 0, -1)
        spreadBlock.update(pySig, 0, -1)

#        plt.show()
        maxClassicAtom2 = classicBlock.getMaxAtom();
        print maxClassicAtom2.length , maxClassicAtom2.frame , maxClassicAtom2.freq_bin , maxClassicAtom2.mdct_value
        maxSpreadcAtom2 = spreadBlock.getMaxAtom();
        print maxSpreadcAtom2.length , maxSpreadcAtom2.frame , maxSpreadcAtom2.freq_bin, maxSpreadcAtom2.mdct_value
        self.assertNotEqual(maxClassicAtom2 ,maxSpreadcAtom2 )

    def dicos(self):
        # create a SpreadDico
        pySig = py_pursuit_Signal.InitFromFile('../../../data/glocs.wav',forceMono=True);
        pySig.crop(0, 5*pySig.samplingFrequency);
        pySig.pad(2048)

        dico = [128,1024,8192];

        classicDIco = py_pursuit_MDCTDico.py_pursuit_MDCTDico(dico, useC=False);
        spreadDico = py_pursuit_MDCTDico.py_pursuit_SpreadDico(dico , useC=False, allBases=True,penalty=0,maskSize = 3)

        classicDIco.initialize(pySig)
        spreadDico.initialize(pySig)

        classicDIco.update(pySig, 2)
        spreadDico.update(pySig, 2)

        classicAtom1 = classicDIco.getBestAtom(0)
        spreadAtom1 = spreadDico.getBestAtom(0)

        self.assertEqual(classicAtom1 ,spreadAtom1 )

        pySig.subtract(classicAtom1);
        classicDIco.update(pySig, 2)
        spreadDico.update(pySig, 2)

        classicAtom2 = classicDIco.getBestAtom(0)
        spreadAtom2 = spreadDico.getBestAtom(0)

        self.assertNotEqual(classicAtom2 ,spreadAtom2 )


    def realTest(self):
        name = 'orchestra';
        pySig = py_pursuit_Signal.InitFromFile('../../../data/'+name+'.wav',forceMono=True,doNormalize=True);
        pySig.crop(0, 5*pySig.samplingFrequency);
        pySig.pad(16384)
        sigEnergy = np.sum(pySig.dataVec**2);
        dico = [128,1024,8192];
        nbAtoms = 200;

        classicDIco = py_pursuit_MDCTDico.py_pursuit_MDCTDico(dico, useC=False);
        spreadDico = py_pursuit_MDCTDico.py_pursuit_SpreadDico(dico , useC=False, allBases=False,Spreadbases= [1024,8192],
                                                               penalty=0.1, maskSize = 3)

        approxClassic, decayClassic = MP.MP(pySig, classicDIco, 20, nbAtoms);
        approxSpread, decaySpread = MP.MP(pySig, spreadDico, 20, nbAtoms);

        plt.figure(figsize=(16,8))
        plt.subplot(121)
        approxClassic.plotTF(ylim=[0,4000])
        plt.title('Classic decomposition : 200 atoms 3xMDCT')
        plt.subplot(122)
        approxSpread.plotTF(ylim=[0,4000])
        plt.title('Decomposition with TF masking: 200 atoms 3xMDCT')
        plt.savefig(name+'_TestTFMasking.eps')

        plt.figure()
        plt.plot([10*np.log10(i/sigEnergy) for i in decayClassic])
        plt.plot([10*np.log10(i/sigEnergy) for i in decaySpread],'r')
        plt.legend(('Classic decomposition','Spreading Atoms'))
        plt.ylabel('Residual energy decay(dB)');
        plt.xlabel('Iteration');
        plt.savefig(name+'_decayTFMasking.eps')

        plt.show()
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
