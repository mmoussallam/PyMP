'''
Created on Sep 13, 2011

@author: moussall
'''
import unittest
from mdct.joint import block as joint_block
from mdct.joint import dico as joint_dico

from mdct import block as mdct_block
from mdct import dico as mdct_dico

import signals , mp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import parallelProjections
import cProfile

audioFilePath = '../../data/'

class Test(unittest.TestCase):


    def testJointMP(self):

        # First test: create a SpreadMDCTBlock and a classic MDCTBlock and check that some
        # atom selections are penalized

        self.blocks();


        self.dicos()
#

        self.realTest()


#        self.realTest()
#        self.perfTestsNL()
#        self.nonLinearTest()


#        self.symetryTest()
#        self.toleranceTests();

#        self.perfTests()

        print "Done"
#        self.realTest();

    def blocks(self):
        pySig = signals.InitFromFile(audioFilePath+'glocs.wav',forceMono=True);
#        pySig.crop(0, 5*pySig.samplingFrequency);
        pySig.crop(0, 16384);
        pySig.pad(2048)

        Scale = 1024;
        print "Starting"


        tol = [1];


        parallelProjections.initialize_plans(np.array([Scale]),np.array(tol))

        classicBlock = mdct_block.Block(Scale,pySig, 0,
                                                 debugLevel=3,useC=True)

        setBlock = joint_block.SetBlock(Scale,[pySig],
                                                    debugLevel=3,useC=True)

        # compute the projections, should be equivalent
        classicBlock.update(pySig, 0, -1)
        setBlock.update([pySig], [0], [-1])

        maxClassicAtom1 = classicBlock.getMaxAtom();
        print maxClassicAtom1.length , maxClassicAtom1.frame
        print maxClassicAtom1.frequencyBin , maxClassicAtom1.mdct_value

#        plt.figure()
#        plt.plot(classicBlock.projectionMatrix)
#        plt.plot(setBlock.projectionMatrix[:,0],'r:')
#        plt.show()

        maxSpreadcAtom1 = setBlock.getAdaptedBestAtoms(noAdapt=True)[0]
        print maxSpreadcAtom1.length , maxSpreadcAtom1.frame
        print maxSpreadcAtom1.frequencyBin, maxSpreadcAtom1.mdct_value

        # assert equality using the inner comparison method of MDCT atoms
        self.assertEqual(maxClassicAtom1 ,maxSpreadcAtom1 )

        pySig.subtract(maxSpreadcAtom1);

        # recompute the projections
        classicBlock.update(pySig, 0, -1)
        setBlock.update([pySig], [0], [-1])

#        plt.show()
        maxClassicAtom2 = classicBlock.getMaxAtom();
        print maxClassicAtom2.length , maxClassicAtom2.frame , maxClassicAtom2.frequencyBin , maxClassicAtom2.mdct_value
        maxSpreadcAtom2 = setBlock.getAdaptedBestAtoms(noAdapt=True)[0]
        print maxSpreadcAtom2.length , maxSpreadcAtom2.frame , maxSpreadcAtom2.frequencyBin, maxSpreadcAtom2.mdct_value

        self.assertEqual(maxClassicAtom2 ,maxSpreadcAtom2 )

        parallelProjections.clean_plans()


    def dicos(self):
        # create a SpreadDico
        pySig = signals.InitFromFile('../../../data/glocs.wav',forceMono=True);
        pySig2 = pySig.copy()


        decalage = 50;

        pySig.crop(0, 5*pySig.fs);
        pySig2.crop(decalage, decalage + 5*pySig.fs);

        pySig.pad(2048)
        pySig2.pad(2048)

        dico = [128,1024,8192];
        tol = [2]*len(dico);
        parallelProjections.initialize_plans(np.array(dico),np.array(tol))

        classicDIco = mdct_dico.Dico(dico, useC=True);
        DicoSet = joint_dico.SetDico(dico, useC=True)

        classicDIco.initialize(pySig)
        DicoSet.initialize((pySig,pySig2))

        classicDIco.update(pySig, 2)
        DicoSet.update((pySig,pySig2), 2)

        classicAtom1 = classicDIco.getBestAtom(0)
        JointAtoms = DicoSet.getBestAtom(0)



        print JointAtoms

#        self.assertEqual(classicAtom1 ,JointAtom1 )

#        pySig.subtract(classicAtom1);
#        classicDIco.update(pySig, 2)
#        spreadDico.update(pySig, 2)
#
#        classicAtom2 = classicDIco.getBestAtom(0)
#        spreadAtom2 = spreadDico.getBestAtom(0)
#
#        self.assertNotEqual(classicAtom2 ,spreadAtom2 )
        parallelProjections.clean_plans()

    def realTest(self):
        pySig = signals.InitFromFile('../../../data/glocs.wav',forceMono=True,doNormalize=True);
        pySig2 = pySig.copy()
#        pySig3 = pySig.copy()

        decalage = 500;

        pySig.crop(0, 4.5*pySig.fs);
        pySig2.crop(decalage, decalage + 4.5*pySig.fs);
#        pySig3.crop(decalage, decalage + 4.5*pySig.samplingFrequency);

        pySig.pad(16384)
        pySig2.pad(16384)
#        pySig3.pad(16384)

        dico = [128,1024,8192];
        tol = [2]*len(dico);
        parallelProjections.initialize_plans(np.array(dico),np.array(tol))

        classicDIco = mdct_dico.LODico(dico);

        jointDico = joint_dico.SetDico(dico ,
                                            selectNature='sum')


        nbAtoms = 50;

        print "%%%%%%%%%%%%%%% Testing mp with one signal %%%%%%%%%%%%%%%"
        print ""
        approxClassic, decayClassic = mp.mp(pySig, classicDIco, 20, nbAtoms, padSignal=False, debug=0);
        print ""
#        print "%%%%%%%%%%%%%%% Testing Joint mp with one signal %%%%%%%%%%%%%%%"
#        print ""
#        approxCommon, approxSpecList, decayList,residualSignalList = mp.mp_joint((pySig,), jointDico, 20, nbAtoms, debug=0);
#        print ""

#        plt.plot()
        print "%%%%%%%%%%%%%%% Testing Joint mp with two signal %%%%%%%%%%%%%%%"
        approxCommon, approxSpecList, decayList, residualSignalList = mp.mp_joint((pySig,pySig2),
                                                                                jointDico, 20,
                                                                                nbAtoms, debug=2);

        print [abs(atom.getAmplitude()) for atom in approxSpecList[0].atoms]
#        print approxSpecList
#        print decayList

        approxCommon.recomposed_signal.write('CommonPattern.wav')
        approxSpecList[0].recomposed_signal.write('pattern1.wav')
        approxSpecList[1].recomposed_signal.write('pattern2.wav')

#        plt.figure(figsize=(16,8))
#        plt.subplot(121)
#        approxClassic.plot_tf(ylim=[0,4000])
#        plt.title('Classic decomposition : 100 atoms 3xMDCT')
#        plt.subplot(122)
#        approxCommon.plot_tf(ylim=[0,4000])
#        plt.title('Common decomposition')
#        plt.savefig('TestTFMasking.png')

        plt.figure()
        plt.plot(decayClassic);
        plt.plot(decayList[0],'r')
        plt.plot(decayList[1],'g')
#        plt.plot(decayList[2],'k')

        plt.figure(figsize=(16,8))
        plt.subplot(121)
        approxSpecList[0].plot_tf(ylim=[0,4000])
        plt.title('Pattern 1')
        plt.subplot(122)
        approxSpecList[1].plot_tf(ylim=[0,4000])
        plt.title('Pattern 2 ')
#        plt.savefig('TestTFMasking.png')



        parallelProjections.clean_plans()

    def symetryTest(self):
        """ Youpi"""
        pySig = signals.InitFromFile('../../../data/glocs.wav',forceMono=True,doNormalize=True);
        pySig2 = signals.InitFromFile('../../../data/voicemale.wav',forceMono=True,doNormalize=True);
        pySig3 = signals.InitFromFile('../../../data/voicefemale.wav',forceMono=True,doNormalize=True);
        pySig4 = signals.InitFromFile('../../../data/orchestra.wav',forceMono=True,doNormalize=True);

        decalage = 0;

        Start = 1.0;
        Stop = 1.5

        pySig.crop(Start*pySig.fs, Stop*pySig.fs);
        pySig2.crop(Start*pySig.fs, Stop*pySig.fs);
        pySig3.crop(Start*pySig.fs, Stop*pySig.fs);
        pySig4.crop(Start*pySig.fs, Stop*pySig.fs);

        pySig2.data += pySig.data
        pySig3.data += pySig.data
        pySig4.data += pySig.data

        pySig2.pad(8192)
        pySig3.pad(8192)
        pySig4.pad(8192)
#        pySig3.pad(16384)

        dico = [128,1024,8192];
        nbAtoms = 1;

        jointDico = joint_dico.SetDico(dico ,
                                                                  selectNature='sum',
                                                                  tol=[2,2,2]);
        jointDicoNL = joint_dico.SetDico(dico , nonLinear = True,
                                                                  selectNature='penalized',
                                                                  tol=[2,2,2] , params=0);

        jointDico.initialize((pySig2,pySig3,pySig4))

        parallelProjections.initialize_plans(np.array(jointDico.sizes), np.array(jointDico.tolerances))
        jointDico.update((pySig2,pySig3,pySig4), 0, debug=2)

        plt.figure()
#        plt.subplot(211)
#        plt.plot([block.projectionMatrix for block in jointDico.blocks])
        plt.plot(jointDico.blocks[2].bestScoreTree)
        jointDico.update((pySig3,pySig2,pySig4), 0, debug=2)
#        plt.subplot(212)
#        plt.plot([block.projectionMatrix for block in jointDico.blocks])
        plt.plot(jointDico.blocks[2].bestScoreTree,'r:')


        plt.show()

        parallelProjections.clean_plans(np.array(jointDico.sizes))
#        # let us compare the two decomposition
#        approxCommon, approxSpecList, decayList, residualSignalList = mp.mp_joint((pySig2,pySig3,pySig4),
#                                                                                jointDico, 20,
#                                                                           nbAtoms, debug=2,
#                                                                           padSignal=False);
#
#        approxCommon2, approxSpecList2, decayList2, residualSignalList2 = mp.mp_joint((pySig3,pySig2,pySig4),
#                                                                                jointDico, 20,
#                                                                           nbAtoms, debug=0,
#                                                                           padSignal=False);
    def nonLinearTest(self):
        """ Youpi"""
        pySig = signals.InitFromFile('../../../data/glocs.wav',forceMono=True,doNormalize=True);
        pySig2 = signals.InitFromFile('../../../data/voicemale.wav',forceMono=True,doNormalize=True);
        pySig3 = signals.InitFromFile('../../../data/voicefemale.wav',forceMono=True,doNormalize=True);
        pySig4 = signals.InitFromFile('../../../data/orchestra.wav',forceMono=True,doNormalize=True);

        decalage = 0;

        Start = 1.0;
        Stop = 1.5

        pySig.crop(Start*pySig.fs, Stop*pySig.fs);
        pySig2.crop(Start*pySig.fs, Stop*pySig.fs);
        pySig3.crop(Start*pySig.fs, Stop*pySig.fs);
        pySig4.crop(Start*pySig.fs, Stop*pySig.fs);

        pySig2.data += pySig.data
        pySig3.data += pySig.data
        pySig4.data += pySig.data

        pySig2.pad(8192)
        pySig3.pad(8192)
        pySig4.pad(8192)
#        pySig3.pad(16384)

        dico = [128,1024,8192];
        nbAtoms = 200;

        jointDico = joint_dico.SetDico(dico ,
                                                                  selectNature='sum',
                                                                  tol=[2,2,2]);
        jointDicoNL = joint_dico.SetDico(dico , nonLinear = True,
                                                                  selectNature='weighted',
                                                                  tol=[2,2,2] , params=1);

        # let us compare the two decomposition
        approxCommon, approxSpecList, decayList, residualSignalList = mp.mp_joint((pySig2,pySig3,pySig4),
                                                                                jointDico, 20,
                                                                           nbAtoms, debug=0,
                                                                           padSignal=False);

        approxCommon2, approxSpecList2, decayList2, residualSignalList2 = mp.mp_joint((pySig3,pySig2,pySig4),
                                                                                jointDico, 20,
                                                                           nbAtoms, debug=0,
                                                                           padSignal=False);


        approxCommonNL, approxSpecListNL, decayListNL, residualSignalListNL = mp.mp_joint((pySig2,pySig3,pySig4),
                                                                                jointDicoNL, 20,
                                                                                nbAtoms, debug=0,
                                                                           padSignal=False);

        # shifting the order of the signals just to see if it's the same
        approxCommonNL2, approxSpecListNL2, decayListNL2, residualSignalListNL2 = mp.mp_joint((pySig3,pySig2,pySig4),
                                                                                jointDicoNL, 20,
                                                                                nbAtoms, debug=0,
                                                                           padSignal=False);

#        cProfile.runctx('mp.mp_joint((pySig2,pySig3), jointDico, 10, nbAtoms ,debug=0,doClean=False,padSignal=False)' , globals() , locals())
#        cProfile.runctx('mp.mp_joint((pySig2,pySig3), jointDicoNL, 10, nbAtoms ,debug=0,doClean=False,padSignal=False)' , globals() , locals())
#        print [block.sigNumber for block in jointDico.blocks]
#        print [block.sigNumber for block in jointDicoNL.blocks]
        print decayList[0][-1],decayList2[0][-1], decayListNL[0][-1], decayListNL2[1][-1]
        print decayList[1][-1],decayList2[1][-1], decayListNL[1][-1], decayListNL2[0][-1]
        print decayList[2][-1],decayList2[2][-1], decayListNL[2][-1], decayListNL2[2][-1]
        A = np.concatenate(jointDicoNL.blocks[2].intermediateProjection, axis = 1);

#        plt.figure()
#        plt.imshow(np.sqrt(np.abs(A[12000:14000,:].T)), interpolation='nearest',aspect='auto',cmap=cm.get_cmap('bone'))
#        plt.colorbar()
#        plt.figure()
#        plt.plot(jointDico.blocks[2].bestScoreTree,'b')
#        plt.plot(jointDicoNL.blocks[2].bestScoreTree,'r')
#        plt.figure()
#        plt.plot(jointDico.blocks[2].enframedDataMatrixList[0],'b')
#        plt.plot(jointDicoNL.blocks[2].enframedDataMatrixList[0],'r:')
#        plt.figure()
#        plt.plot(jointDico.blocks[2].projectionMatrix,'b')
#        plt.plot(jointDicoNL.blocks[2].projectionMatrix,'r')
#
        print [atom.getAmplitude() for atom in approxCommonNL.atoms]
#        plt.show()
        plt.figure()
        plt.subplot(211)
        approxCommon.plot_tf()
        plt.subplot(212)
        approxCommonNL.plot_tf()
        plt.show()




    def perfTests(self):
        print " Starting Performances Tests"
        pySig = signals.InitFromFile('../../../data/glocs.wav',forceMono=True,doNormalize=True);
        pySig2 = pySig.copy()
#        pySig3 = pySig.copy()

        decalage = 500;

        pySig.crop(0, 4.5*pySig.fs);
        pySig2.crop(decalage, decalage + 4.5*pySig.fs);
#        pySig3.crop(decalage, decalage + 4.5*pySig.samplingFrequency);

        pySig.pad(16384)
        pySig2.pad(16384)
#        pySig3.pad(16384)

        dico = [128,1024,8192];

        classicDIco = mdct_dico.LODico(dico);

        jointDico = joint_dico.SetDico(dico , selectNature='sum')
        jointDicoTol = joint_dico.SetDico(dico ,
                                               selectNature='sum',
                                                                  tol=[2,2,2]);

        cProfile.runctx('mp.mp_joint((pySig,pySig2,pySig2), jointDico, 10, 2000 ,debug=0,doClean=False)' , globals() , locals())
        cProfile.runctx('mp.mp_joint((pySig,pySig2,pySig2), jointDicoTol, 10, 2000 ,debug=0)' , globals() , locals())

    def perfTestsNL(self):
        print " Starting Performances Tests with NL techniques"
        pySig = signals.InitFromFile('../../../data/glocs.wav',forceMono=True,doNormalize=True);
        pySig2 = pySig.copy()
#        pySig3 = pySig.copy()

        decalage = 500;

        pySig.crop(0, 4.5*pySig.fs);
        pySig2.crop(decalage, decalage + 4.5*pySig.fs);
#        pySig3.crop(decalage, decalage + 4.5*pySig.samplingFrequency);

        pySig.pad(16384)
        pySig2.pad(16384)
#        pySig3.pad(16384)

        dico = [128,1024,8192];

        classicDIco = mdct_dico.LODico(dico);

#        cProfile.runctx('mp.mp(pySig, classicDIco, 30, 1500 ,debug=0)' , globals() , locals())

        natures = ('penalized',)
        for nature in natures:
            print "Starting ", nature
            jointDicoTol = joint_dico.SetDico(dico ,
                                                                  selectNature=nature,
                                                                  tol=[2,2,2],params=1);
            cProfile.runctx('mp.mp_joint((pySig,pySig2,pySig2), jointDicoTol, 10, 1000,interval=500 ,debug=0)' , globals() , locals())

            cProfile.runctx('mp.mp_joint((pySig,pySig2,pySig2), jointDicoTol, 10, 1000,escape=True,interval=500 ,debug=0)' , globals() , locals())




    def toleranceTests(self):
        print "%%%%%%%%%%%%%%% Testing Joint mp with increased Tolerance %%%%%%%%%%%%%%%"
        pySig = signals.InitFromFile('../../../data/glocs.wav',forceMono=True,doNormalize=True);
        pySig2 = pySig.copy()
#        pySig3 = pySig.copy()

        decalage = 750;

        pySig.crop(0, 4*16384);
        pySig2.crop(decalage, decalage + 4*16384);
#        pySig3.crop(decalage, decalage + 4.5*pySig.samplingFrequency);

        pySig.pad(16384)
        pySig2.pad(16384)
#        pySig3.pad(16384)

        dico = [128,1024];
        tol = [16,2];

        jointDico = joint_dico.SetDico(dico ,
                                                               selectNature='sum');
        jointDicoTol = joint_dico.SetDico(dico ,
                                                               selectNature='sum',
                                                               tol = tol);

        print jointDico.tolerances
        nbatoms = 50;
        meanApprox, currentApproxList, resEnergyList , residualSignalList = mp.mp_joint((pySig,pySig2,pySig2),
                                                                                       jointDico, 10, nbatoms ,debug=0)

        print jointDicoTol.tolerances
        meanApproxTol, currentApproxListTol, resEnergyListTol , residualSignalListTol = mp.mp_joint((pySig,pySig2,pySig2),
                                                                                       jointDicoTol, 10, nbatoms ,debug=0)

        for i in range(len(residualSignalList)):
            print np.sum(residualSignalList[i].data **2),  np.sum(residualSignalListTol[i].data **2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
#    suite  = unittest.TestSuite()

#    suite.addTest(py_MPTest3())
    unittest.main()
#    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()
#filepath = "/tsi/linuxhome/moussall/data/workspace/Python/Py_pursuit/data/glocs.wav";
#pySig = signals.InitFromFile(filepath,forceMono=True,doNormalize=True);
#
#pySig2 = signals.InitFromFile(filepath,forceMono=True,doNormalize=True);
##print pySig.length , pySig2.length

#pySig3 = pySig.copy()

#decalage = 0;
#
#pySig.crop(0, 4.5*pySig.samplingFrequency);
#pySig2.crop(decalage, decalage + 4.5*pySig.samplingFrequency);
#
#print pySig.length , pySig2.length
#
##pySig3.crop(decalage, decalage + 4.5*pySig.samplingFrequency);
#
#pySig.pad(16384)
#
#pySig2.pad(16384)
#
#print pySig.length , pySig2.length
##pySig3.pad(16384)
#
#dico = [128,1024,8192];
#
#classicDIco = mdct_dico.mdct_dico(dico);
#
#jointDico = mdct_dico.SetDico(dico)
#
#
#nbAtoms = 100;
##
#approxClassic, decayClassic = mp.mp(pySig, classicDIco, 20, nbAtoms,padSignal=False);
#
#
#print pySig.length , pySig2.length
#approxCommon, approxSpecList, decayList = mp.mp_joint((pySig,pySig2,pySig), jointDico, 20, nbAtoms, debug=2);
##
#print [abs(atom.getAmplitude()) for atom in approxSpecList[0].atoms]













