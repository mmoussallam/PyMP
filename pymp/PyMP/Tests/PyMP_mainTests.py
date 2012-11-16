"""
Python test file for PyMP engine

testing normal behavior of most MP functions and classes

M.Moussallam
"""

import unittest
from Classes import *
from Tools import mdct
import matplotlib.pyplot as plt
from numpy import  random , zeros , array
import time

import cProfile , os , math
global _Logger
from Classes.mdct import *
#from Classes.gabor import *
from Classes.mdct import pymp_MDCTAtom as MDCTAtom
from Classes import pymp_Approx as Approx
from Classes import pymp_Signal as Signal
import MP
audioFilePath = '../../data/'


#import scipy.io
class pymp_AtomTest(unittest.TestCase):    
    def setUp(self):
        pass

    def runTest(self):
        # empty creation
        pyAtom = pymp_Atom.pymp_Atom()
        self.assertEqual(pyAtom.length , 0)
        self.assertEqual(pyAtom.amplitude , 0)
        self.assertEqual(pyAtom.nature , 'Abstract')
        
        del pyAtom
        
        # full creation
        pyAtom2 = pymp_MDCTAtom.pymp_MDCTAtom(1024 , 1 , 12432 , 128 , 44100 , 0.57)
        self.assertEqual(pyAtom2.length , 1024)
        self.assertEqual(pyAtom2.amplitude , 1)
        self.assertEqual(pyAtom2.mdct_value , 0.57)
        self.assertEqual(pyAtom2.samplingFrequency , 44100)
        self.assertEqual(pyAtom2.reducedFrequency , float(128 + 0.5)/float(1024))
        self.assertEqual(pyAtom2.timePosition , 12432)
        self.assertEqual(pyAtom2.nature , 'MDCT')        

#        synthesizedAtom2 = pyAtom2.synthesize()                    
        
        synthAtom3 = pyAtom2.synthesizeIFFT()
        
#        energy1 = sum(synthesizedAtom2.waveform**2)
        energy2 = sum(synthAtom3.real**2)
        
        print  energy2
        
#        plt.plot(synthesizedAtom2.real)
        plt.plot(synthAtom3.real)
        del pyAtom2
        
        print " testing LOMP atoms synthesis "
        mdctValue = 0.57;
        timeShift = 144;
        projectionScore = -0.59;
        pyAtomLOMP = pymp_MDCTAtom.pymp_MDCTAtom(1024 , 1 , 12432 , 128 , 44100 , 0.57)
        pyAtomLOMP.timeShift = timeShift
        pyAtomLOMP.projectionScore = projectionScore
        
        # test 1 synthesis
        pyAtomLOMP.synthesizeIFFT()
        wf1 = pyAtomLOMP.waveform.copy()
                
        wf2 = -(math.sqrt(abs(projectionScore)/sum(wf1**2)))*wf1

        mdctVec = zeros(3*1024);
        mdctVec[1024 +  128] = projectionScore;
        wf3 =  mdct.imdct(mdctVec , 1024)[0.75*1024 : 1.75*1024] 
        
        plt.figure()
        plt.plot(wf1)
        plt.plot(wf2)
        plt.plot(wf3)
        plt.legend(('1', '2', '3'))
#        plt.show()
        
    def tearDown(self):
        pass
    
class pymp_DicoTest(unittest.TestCase):
    def runTest(self):   
        # test dictionary class
        pyDico = pymp_Dico.pymp_Dico()
        self.assertEqual(pyDico.nature , 'Abstract')
        
        del pyDico
        
        # test dictionary class
        pyDico = pymp_MDCTDico.pymp_MDCTDico([2**l for l in range(7, 15, 1)] ,debugLevel=3)
        self.assertEqual(pyDico.sizes , [128 , 256 , 512 , 1024 , 2048 , 4096 , 8192 , 16384])
        self.assertEqual(pyDico.nature , 'MDCT')
        
        del pyDico
        
        
class pymp_Signaltest(unittest.TestCase):
    def runTest(self):
        
        
        pySig = Signal.pymp_Signal(debugLevel=3);
        self.assertEqual(pySig.length , 0)
        self.assertEqual(len(pySig.dataVec) , 0)
        
        del pySig
        
        pySig = Signal.InitFromFile(audioFilePath +"ClocheB.wav")
        self.assertNotEqual(pySig.length , 0)
        self.assertNotEqual(pySig.dataVec , [])
        self.assertEqual(pySig.channelNumber , 2)
        self.assertEqual(pySig.location , audioFilePath +"ClocheB.wav")
        self.assertEqual(pySig.samplingFrequency , 8000)
        
        # Last test, the wigner ville plot
        
        pySig.crop(8000, 8256)
        
        pySig.WignerVPlot()
        
        
        del pySig
        
        pySig = Signal.InitFromFile(audioFilePath +"ClocheB.wav" , True , True)
        data1 = pySig.dataVec.copy()
        # try adding and subtracting an atom
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(1024 , 0.57 , 4500 , 128 , 8000 , 0.57)
        pySig.add(pyAtom)
        
        data2 = pySig.dataVec.copy()
        
        self.assertNotEqual(sum((data1 - data2)**2) , 0)
        pySig.subtract(pyAtom)
        
#        plt.plot(data1)
#        plt.plot(data2)
#        plt.plot(pySig.dataVec)
#        plt.legend(("original", "added" , "subtracted"))
#        
        self.assertAlmostEqual(sum((pySig.dataVec - data1)**2) , 0)
        
        # test on a long Signal 
        L = 4*16384
        longSignal = Signal.pymp_LongSignal(audioFilePath +"Bach_prelude_40s.wav", L)
        # suppose we want to retrieve the middle of the signal , namely from frame 5  to 12
        startSeg = 2;
        segNumber = 3;
        shortSignal = longSignal.getSubSignal(startSeg, segNumber, False)
        
        # witness signal
        witSignal = Signal.InitFromFile(audioFilePath +"Bach_prelude_40s.wav", True, False)
        
#        shortSignal.plot()
#        witSignal.plot()
        plt.figure()
        plt.plot(shortSignal.dataVec)
        plt.plot(witSignal.dataVec[startSeg*L:startSeg*L + segNumber*L] , ':')
        
#        plt.show()
        
        # test writing signal
        outputPath = 'subsignal.wav'
        if os.path.exists(outputPath):
            os.remove(outputPath)
        
        shortSignal.write(outputPath)
        
        # test long signal with overlap 50 %
        longSignal = Signal.pymp_LongSignal(audioFilePath +"Bach_prelude_40s.wav", L, True, 0.5)
        # suppose we want to retrieve the middle of the signal , namely from frame 5  to 12
        startSeg = 2;
        segNumber = 3;
        shortSignal = longSignal.getSubSignal(startSeg, segNumber, False)
        
        # witness signal
        witSignal = Signal.InitFromFile(audioFilePath +"Bach_prelude_40s.wav", True, False)
        
#        shortSignal.plot()
#        witSignal.plot()
        plt.figure()
        plt.plot(shortSignal.dataVec)
        plt.plot(witSignal.dataVec[startSeg*L/2:startSeg*L/2 + segNumber*L] , ':')
        
        
        # Testing the downsampliong utility
        plt.figure()
        plt.subplot(121)
        plt.specgram(shortSignal.dataVec, 256, shortSignal.samplingFrequency, noverlap=128)
        self.assertEqual(shortSignal.samplingFrequency, 44100)
        shortSignal.write('normal.wav');
        shortSignal.downsample(8000);
        self.assertEqual(shortSignal.samplingFrequency, 8000)
        plt.subplot(122)
        plt.specgram(shortSignal.dataVec, 256, shortSignal.samplingFrequency, noverlap=128)
        
        shortSignal.write('downsampled.wav');
        

        
        
#        plt.show()
        
class pymp_BlockTest(unittest.TestCase):
    def runTest(self):
        
        
        pySigOriginal = pymp_Signal.InitFromFile(audioFilePath +"ClocheB.wav", True , True);
        pySigOriginal.crop(0 , 5*8192)
        pySigOriginal.pad(2048)

        block = pymp_MDCTBlock.pymp_MDCTBlock(1024 , pySigOriginal , debugLevel=3 , useC=True);
        # testing the automated enframing of the data
        self.assertEqual(block.frameLength , 512);
        print block.frameNumber
#        import parallelProjections
#        parallelProjections.initialize_plans(array([1024,]))
        
        try:
            import fftw3
        except ImportError:
            print " FFTW3 python wrapper not installed, abandonning test"
            return;
        block.update(pySigOriginal)
        
        plt.plot(block.projectionMatrix.flatten(1))
        plt.plot(mdct.mdct(pySigOriginal.dataVec , block.scale))
#        plt.show()
        
        self.assertAlmostEqual(sum((block.projectionMatrix - mdct.mdct(pySigOriginal.dataVec , block.scale))**2) , 0)
        
#        parallelProjections.clean_plans(array([1024,]))
        del pySigOriginal



class py_MPTest(unittest.TestCase):
    def runTest(self):
#        pyDico = Dico.pymp_Dico([2**l for l in range(7,15,1)] , Atom.transformType.MDCT)        
        
        pyDico = pymp_MDCTDico.pymp_MDCTDico([256 , 2048 , 8192] ) 
        pySigOriginal = pymp_Signal.InitFromFile(audioFilePath +"ClocheB.wav", True , True);            
        pySigOriginal.crop(0 , 5*16384);
        
        
        pySigOriginal.dataVec += 0.01*random.random(5*16384)
        
#        pySigOriginal.plot()
        
        # first try with a single-atom signal
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(2048 , 1, 11775 , 128 , 8000 , 0.57)
        pyApprox_oneAtom = pymp_Approx.pymp_Approx(pyDico, [], pySigOriginal)
        pyApprox_oneAtom.addAtom(pyAtom)
        
#        plt.plot(pyApprox_oneAtom.synthesize(0).dataVec[12287:12287+2048])
#        plt.plot(pyAtom.synthesize(0))
#        plt.legend(("IMDCT","Atom"))
#        plt.show()
        
        pySignal_oneAtom = pymp_Signal.pymp_Signal(pyApprox_oneAtom.synthesize(0).dataVec, pySigOriginal.samplingFrequency, False)
        
        # add some noise
#        pySignal_oneAtom.dataVec += 0.0001*random.random(5*16384)
        
#        approximant = MP.MP_proto1(pySignal_oneAtom, pyDico, 10, 10)
##        approximant.plotTF()
#        plt.plot(pySignal_oneAtom.dataVec)
#        plt.plot(approximant.synthesize(0).dataVec)
#        plt.legend(("original","approximant"))        
#        plt.show()
#        del approximant
        
        # test two atoms
#        pyAtom2 = Atom.pymp_Atom(16384 , 1, 6*8192-1-4096 , 128 , 8000 , Atom.transformType.MDCT , -0.42)
#        pySignal_oneAtom.add(pyAtom2)
##        pySignal_oneAtom.plot()
#        approximant = MP.MP(pySignal_oneAtom, pyDico, 20, 10 , True, False)[0]
#        
#        plt.plot(approximant.synthesize(0).dataVec)
#        plt.title("Reconstituted signal")
#        plt.show()
#        plt.plot(pySignal_oneAtom.dataVec)
#        plt.plot(approximant.synthesize(0).dataVec)
#        plt.legend(("original","approximant"))        
#        plt.show()
        
#        cProfile.run('MP.MP_proto1')
        
        # second test 
        pySigOriginal = pymp_Signal.InitFromFile(audioFilePath +"ClocheB.wav", True , True);        
#        approximant = MP.MP_proto1(pySigOriginal, pyDico, 10, 10)  
#        self.assertTrue( isinstance(approximant , Approx.pymp_Approx));

        
        # last test - decomposition profonde
        pyDico2 = pymp_MDCTDico.pymp_MDCTDico([128 , 256 , 512 , 1024 , 2048 , 4096, 8192 , 16384]  ,parallel=False) 
        pyDico1 = pymp_MDCTDico.pymp_MDCTDico([16384]  ) 
        # profiling test
        print "Plain"
        cProfile.runctx('MP.MP(pySigOriginal, pyDico1, 20, 1000 ,debug=0 , itClean=True)' , globals() , locals())
        
        cProfile.runctx('MP.MP(pySigOriginal, pyDico2, 20, 1000 ,debug=0 , itClean=True)' , globals() , locals())

#        print "Parallel"
#        cProfile.runctx('MP.MP(pySigOriginal, pyDico_parallel, 40, 100 ,debug=0)' , globals() , locals())
#        
#        print pySigOriginal.length
#        t0 = time.clock()    
#        approximant = MP.MP(pySigOriginal, pyDico2, 20, 4000 ,False)[0]
#        t1 = time.clock()
#        print "SRR of ", approximant.computeSRR() , " dB achieved in ", t1 - t0 , " sec and ", approximant.atomNumber,"iteration with C"
#        
#        del approximant , pyDico2
#        
#        t2 = time.clock()
#        pyDico2 = Dico.pymp_LODico([128 , 256 , 512 , 1024 , 2048 , 4096, 8192 , 16384])     
#        approximant = MP.MP(pySigOriginal, pyDico2, 20, 4000 ,debug=0,padSignal=True)[0]
#        t3 = time.clock()
#        print "SRR of ", approximant.computeSRR() , " dB achieved in ", t3 - t2 , " sec with C and ", approximant.atomNumber,"iteration and LOMP"
#        
#        del approximant , pyDico2
##        

#        pyDico1 = Dico.pymp_LODico([2**j for j in range(7,15) ] , Atom.transformType.MDCT )     
#        pyDico2 = Dico.pymp_LODico([2**j for j in range(7,15) ] , Atom.transformType.MDCT )     
#        
#        t = time.clock()
#        approximant_High , decays_high = MP.MP(pySigOriginal, pyDico2, 40, 1000 ,debug=0 , forceHighFreqs=True , HFitNum = 800)
#        print "elapsed : " , time.clock() - t;
#        t = time.clock()
#        approximant , decay = MP.MP(pySigOriginal, pyDico1, 40, 1000 ,debug=0,padSignal=False)
#        print "elapsed : " , time.clock() - t;
#        approximant_High.recomposedSignal.write('recomposedHF.wav');
#        approximant.recomposedSignal.write('recomposed.wav');        
#
#        print "SRR of ", approximant.computeSRR() , " dB achieved in ", t3 - t2 , " sec without C"
#        plt.figure()
#        plt.subplot(211)
#        approximant.plotTF()
#        plt.subplot(212)
#        approximant_High.plotTF()
#        
#        plt.figure()
#        plt.plot(decay)
#        plt.plot(decays_high , 'r')
#        plt.legend(('Without HF forcing' , 'With HF forcing'));
    
#        plt.show()
        
#        del pySigOriginal


      


class pymp_ApproxTest(unittest.TestCase):
    
    def runTest(self):   
#        self.writeXmlTest()
        
        # test dictionary class

        pyApprox = pymp_Approx.pymp_Approx(debugLevel=3)        
        self.assertEqual(pyApprox.originalSignal , None)
        self.assertEqual(pyApprox.atomNumber , 0)
        self.assertEqual(pyApprox.SRR , 0)
        self.assertEqual (pyApprox.atoms , [])
        
        del pyApprox
        
        pyDico = pymp_MDCTDico.pymp_MDCTDico([2**l for l in range(7, 15, 1)] )        
        pySigOriginal = pymp_Signal.InitFromFile(audioFilePath +"ClocheB.wav" , True);
        pySigOriginal.crop(0 , 5*max(pyDico.sizes));
        
        pyApprox = pymp_Approx.pymp_Approx(pyDico , [] , pySigOriginal)  
        
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(1024 , 1 , 12288 - 256 , 128 , 44100  , 0.57)
        pyApprox.addAtom(pyAtom);
#        
#        approxSignal = pyApprox.synthesize(0)        
#        approxSignal.plot()
#        
#        del approxSignal
        
        pyApprox.addAtom(pymp_MDCTAtom.pymp_MDCTAtom(8192 , 1 , 4096 - 2048 , 32 , 44100 , -0.24));
        approxSignal1 = pyApprox.synthesize(0)        
        
        plt.figure()
        plt.subplot(121)
        pyApprox.plotTF()
        plt.subplot(122)
        pyApprox.plotTF(multicolor=True,keepValues=True)
#        approxSignal1.plot()
        
#        del approxSignal
        approxSignal2 = pyApprox.synthesize(1)        
#        approxSignal2.plot()
        plt.figure()
        plt.plot(approxSignal1.dataVec)        
        plt.plot(approxSignal2.dataVec)
        plt.plot(approxSignal1.dataVec - approxSignal2.dataVec)
        plt.legend(("MDCT", "AtomSynth", "Diff"))
#        plt.show() #TODO here correct mistakes
        
        # assert two methods are equivalent
        self.assertAlmostEqual(sum((approxSignal1.dataVec - approxSignal2.dataVec)**2), 0)
        
        # testing filtering
        self.assertEqual(pyAtom , pyApprox.filterAtoms(1024, None, None).atoms[0])        
        self.assertEqual(pyAtom , pyApprox.filterAtoms(1024, [12000 , 15000], None).atoms[0])
        
        print pyApprox.computeSRR()
        # TODO testing du SRR
        
        #testing the writeToXml and readFromXml methods
        
        
        del pySigOriginal
      
    
    def writeXmlTest(self):
        pySigOriginal = pymp_Signal.InitFromFile(audioFilePath +"ClocheB.wav" , True , True, debugLevel=0);            
        pySigOriginal.crop(0 , 1*16384);
        pyDico = pymp_MDCTDico.pymp_MDCTDico([256 , 2048 , 8192] , debugLevel=2 ) 
        # first compute an approximant using MP
        approximant = MP.MP(pySigOriginal, pyDico, 10, 10, debug=2)[0]
        
        outputXmlPath = "approx_test.xml";
        doc = approximant.writeToXml(outputXmlPath)

        # Test reading from the xml flow
        newApprox = pymp_Approx.readFromXml('', doc)        
        self.assertEqual(newApprox.dico.sizes , approximant.dico.sizes)
        self.assertEqual(newApprox.atomNumber , approximant.atomNumber)
        self.assertEqual(newApprox.length , approximant.length)
        
        # now the hardest test
        newApprox.synthesize()
        newApprox.recomposedSignal.plot()
        self.assertAlmostEquals(sum(approximant.recomposedSignal.dataVec - newApprox.recomposedSignal.dataVec) , 0)
        
        # test reading from the xml file
        del newApprox
        newApprox = pymp_Approx.readFromXml(outputXmlPath)        
        self.assertEqual(newApprox.dico.sizes , approximant.dico.sizes)
        self.assertEqual(newApprox.atomNumber , approximant.atomNumber)
        self.assertEqual(newApprox.length , approximant.length)
        
        # now the hardest test
        newApprox.synthesize()
        newApprox.recomposedSignal.plot()
        self.assertAlmostEquals(sum(approximant.recomposedSignal.dataVec - newApprox.recomposedSignal.dataVec) , 0)
        
        mdctOrig = approximant.toArray()[0]
        mdctRead = newApprox.toArray()[0]
        
        del doc , newApprox
        # test writing with LOMP atoms
        pyCCDico = pymp_MDCTDico.pymp_LODico([256 , 2048 , 8192] ) 
        approx_LOMP = MP.MP(pySigOriginal, pyCCDico, 10, 10, False)[0]
        
        outputXmlPath = "approxLOMP_test.xml";
        doc = approx_LOMP.writeToXml(outputXmlPath)
        
        # Test reading from the xml flow
        newApprox = pymp_Approx.readFromXml('', doc)        
        self.assertEqual(newApprox.dico.sizes , approx_LOMP.dico.sizes)
        self.assertEqual(newApprox.atomNumber , approx_LOMP.atomNumber)
        self.assertEqual(newApprox.length , approx_LOMP.length)
        self.assertAlmostEqual(sum(newApprox.toArray()[0] - approx_LOMP.toArray()[0]), 0)
        
        plt.figure()
        plt.plot(newApprox.toArray()[0])
        plt.plot(approx_LOMP.toArray()[0] , 'r:')
#        plt.show()
        
        # test reading from the xml file
        del newApprox
        newApprox = pymp_Approx.readFromXml(outputXmlPath)        
        self.assertEqual(newApprox.dico.sizes , approx_LOMP.dico.sizes)
        self.assertEqual(newApprox.atomNumber , approx_LOMP.atomNumber)
        self.assertEqual(newApprox.length , approx_LOMP.length)
        self.assertAlmostEqual(sum(newApprox.toArray()[0] - approx_LOMP.toArray()[0]), 0)
      
class py_MPTest2(unittest.TestCase):
    def runTest(self):
                              
        pyCCDico = pymp_MDCTDico.pymp_LODico([256 , 2048 , 8192]  ) 
        pyDico = pymp_MDCTDico.pymp_MDCTDico([256 , 2048 , 8192])         
        
        pySigOriginal = pymp_Signal.InitFromFile(audioFilePath +"ClocheB.wav" , True , True);            
        pySigOriginal.crop(0 , 1*16384);
        
        #intentionally non aligned atom -> objective is to fit it completely through time shift calculation
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(2048 , 1, 8192 - 512 , 128 , 8000 , 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = pymp_Signal.pymp_Signal(0.0001*random.random(pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)
        
        fig_1Atom = plt.figure()
        print "Testing one Aligned Atom with LOMP - should be perfect"
        approximant = MP.MP(pySignal_oneAtom, pyCCDico, 10, 10, debug=1)[0]
#        approximant.plotTF()
#        plt.subplot(211)
        plt.plot(pySignal_oneAtom.dataVec)
        plt.plot(approximant.recomposedSignal.dataVec)
        plt.plot(pySignal_oneAtom.dataVec - approximant.recomposedSignal.dataVec)
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Aligned Atom with LOMP")
#        plt.show()
        plt.xlabel("samples")
#        plt.savefig(figPath +"LOMP_aligned" +ext)
        
#        approximant.writeToXml(ApproxPath+"LOMP_aligned")
        
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        del pyAtom , approximant , pySignal_oneAtom
        
        print "Testing one Non-Aligned Atom with MP - should not be perfect"
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(2048 , 1, 7500 , 128 , 8000 , 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = pymp_Signal.pymp_Signal(0.0001*random.random(pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)
        
        approximant = MP.MP(pySignal_oneAtom, pyDico, 10, 10, debug=1)[0]
#        approximant.plotTF()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySignal_oneAtom.dataVec)
        plt.plot(approximant.recomposedSignal.dataVec)
        plt.plot(pySignal_oneAtom.dataVec - approximant.recomposedSignal.dataVec)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Non aligned atom with MP : SRR of " + str(int(approximant.computeSRR())) + " dB in " + str(approximant.atomNumber) + " iteration")        
#        plt.show()
        print " Approx Reached : " , int(approximant.computeSRR()) , " dB in " , approximant.atomNumber , " iteration"

        del approximant
        
        print "Testing one Non-Aligned Atom with LOMP - should be almost perfect"
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(2048 , 1, 7500 , 128 , 8000  , 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = pymp_Signal.pymp_Signal(0.0001*random.random(pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)
        
        approximant = MP.MP(pySignal_oneAtom, pyCCDico, 10, 10, False, False)[0]
#        approximant.plotTF()
        plt.subplot(212)
        plt.plot(pySignal_oneAtom.dataVec)
        plt.plot(approximant.recomposedSignal.dataVec)
        plt.plot(pySignal_oneAtom.dataVec - approximant.recomposedSignal.dataVec)
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Non aligned atom with LOMP : SRR of " + str(int(approximant.computeSRR())) + " dB in " + str(approximant.atomNumber) + " iteration")  
        plt.xlabel("samples")
#        plt.savefig(figPath+"nonAlignedAtom_MP_vs_LOMP" + ext)
#        plt.show()
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        
        
        del approximant
        
        print "Testing Real signal with MP"
        approximant = MP.MP(pySigOriginal, pyDico, 10, 10, False)[0]
#        approximant.plotTF()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.dataVec[4096:-4096])
        plt.plot(approximant.recomposedSignal.dataVec[4096:-4096])
        plt.plot(pySigOriginal.dataVec[4096:-4096] - approximant.recomposedSignal.dataVec[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signal with MP : SRR of " + str(int(approximant.computeSRR()))+ " dB in " + str(approximant.atomNumber) + " iteration")        
#        plt.show()
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        del approximant
        
        print "Testing Real signal with LOMP"
#        pyCCDico = 
        approximant = MP.MP(pySigOriginal, pyCCDico, 10, 10, False, False)[0]
#        approximant.plotTF()
        plt.subplot(212)
        plt.plot(pySigOriginal.dataVec[4096:-4096])
        plt.plot(approximant.recomposedSignal.dataVec[4096:-4096])
        plt.plot(pySigOriginal.dataVec[4096:-4096] - approximant.recomposedSignal.dataVec[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Bell Signal with LOMP : SRR of " + str(int(approximant.computeSRR())) + " dB in " + str(approximant.atomNumber) + " iteration")  
        plt.xlabel("samples")
#        plt.savefig(figPath+"RealSignal_MP_vs_LOMP_nbIt10"+ext)
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        
        del approximant
        print "comparing results and processing times for long decompositions"
        t0 = time.clock()
        approx1 =  MP.MP(pySigOriginal, pyDico, 20, 500, False)[0]
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.dataVec[12288:-12288])
        plt.plot(approx1.recomposedSignal.dataVec[12288:-12288])
        plt.plot(pySigOriginal.dataVec[12288:-12288] - approx1.recomposedSignal.dataVec[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell Signal with MP : SRR of " + str(int(approx1.computeSRR())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1-t0) + "s")  
#        plt.show()
        print " Approx Reached : " , approx1.computeSRR() , " dB in " , approx1.atomNumber , " iteration and: " , t1-t0 , " seconds"
        
        t2 = time.clock()
        approx2 =  MP.MP(pySigOriginal, pyCCDico, 20, 500, False, False)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pySigOriginal.dataVec[12288:-12288])
        plt.plot(approx2.recomposedSignal.dataVec[12288:-12288])
        plt.plot(pySigOriginal.dataVec[12288:-12288] - approx2.recomposedSignal.dataVec[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Bell Signal with LOMP : SRR of " + str(int(approx2.computeSRR())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3-t2) + "s")  
#        plt.savefig(figPath+"RealSignal_MP_vs_LOMP_SRR20"+ext)
        print " Approx Reached : " , approx2.computeSRR() , " dB in " , approx2.atomNumber , " iteration and: " , t3-t2 , " seconds"
        
        del approx1 , approx2
        del pySigOriginal
        print "comparing results and processing times for long decompositions of white gaussian noise"
        pyNoiseSignal = pymp_Signal.pymp_Signal(0.5*random.random(5*16384), 44100, False)
        pyNoiseSignal.pad(16384)
        t0 = time.clock()
        approx1 =  MP.MP(pyNoiseSignal, pyDico, 10, 500, False, True)[0]
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pyNoiseSignal.dataVec[16384:-16384])
        plt.plot(approx1.recomposedSignal.dataVec[16384:-16384])
        plt.plot(pyNoiseSignal.dataVec[16384:-16384] - approx1.recomposedSignal.dataVec[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("White Noise signal with MP : SRR of " + str(int(approx1.computeSRR())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1-t0) + "s")  
#        plt.show()
        print " Approx Reached : " , approx1.computeSRR() , " dB in " , approx1.atomNumber , " iteration and: " , t1-t0 , " seconds"
        
        t2 = time.clock()
        approx2 =  MP.MP(pyNoiseSignal, pyCCDico, 10, 500, False, True)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pyNoiseSignal.dataVec[16384:-16384])
        plt.plot(approx2.recomposedSignal.dataVec[16384:-16384])
        plt.plot(pyNoiseSignal.dataVec[16384:-16384] - approx2.recomposedSignal.dataVec[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Noise Signal with LOMP : SRR of " + str(int(approx2.computeSRR())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3-t2) + "s")  
        plt.xlabel("samples")
#        plt.savefig(figPath+"NoiseSignal_MP_vs_LOMP_SRR20"+ext)
        print " Approx Reached : " , approx2.computeSRR() , " dB in " , approx2.atomNumber , " iteration and: " , t3-t2 , " seconds"

class py_MPTest2Bis(unittest.TestCase):
    def runTest(self):
        ApproxPath = "../Approxs/"
#        ext = ".png"
                              
        pyRandomDico = pymp_RandomDicos.pymp_RandomDico([256 , 2048 , 8192] , 'scale') 
        pyDico = pymp_MDCTDico.pymp_MDCTDico([256 , 2048 , 8192])         
        
        pySigOriginal = Signal.InitFromFile(audioFilePath +"ClocheB.wav" , True , True);            
        pySigOriginal.crop(0 , 1*16384);
        
        #intentionally non aligned atom -> objective is to fit it completely through time shift calculation
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(2048 , 1, 8192 - 512 , 128 , 8000, 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = Signal.pymp_Signal(0.0001*random.random(pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)
        
        fig_1Atom = plt.figure()
        print "Testing one Aligned Atom with Random"
        approximant = MP.MP(pySignal_oneAtom, pyRandomDico, 10, 10, False, False)[0]
#        approximant.plotTF()
#        plt.subplot(211)
        plt.plot(pySignal_oneAtom.dataVec)
        plt.plot(approximant.recomposedSignal.dataVec)
        plt.plot(pySignal_oneAtom.dataVec - approximant.recomposedSignal.dataVec)
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Aligned Atom with LOMP")
#        plt.show()
        plt.xlabel("samples")
#        plt.savefig(figPath +"LOMP_aligned" +ext)
        
        approximant.writeToXml(ApproxPath+"LOMP_aligned")
        
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        del pyAtom , approximant , pySignal_oneAtom
        
        print "Testing one Non-Aligned Atom with MP - should not be perfect"
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(2048 , 1, 7500 , 128 , 8000 , 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = Signal.pymp_Signal(0.0001*random.random(pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)
        
        approximant , decay = MP.MP(pySignal_oneAtom, pyDico, 10, 10, False, False)
#        approximant.plotTF()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySignal_oneAtom.dataVec)
        plt.plot(approximant.recomposedSignal.dataVec)
        plt.plot(pySignal_oneAtom.dataVec - approximant.recomposedSignal.dataVec)
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Non aligned atom with MP : SRR of " + str(int(approximant.computeSRR())) + " dB in " + str(approximant.atomNumber) + " iteration")        
#        plt.show()
        print " Approx Reached : " , int(approximant.computeSRR()) , " dB in " , approximant.atomNumber , " iteration"

        del approximant
        
        print "Testing one Non-Aligned Atom with Random"
        pyAtom = pymp_MDCTAtom.pymp_MDCTAtom(2048 , 1, 7500 , 128 , 8000 , 0.57)
        pyAtom.synthesize()
        pySignal_oneAtom = Signal.pymp_Signal(0.0001*random.random(pySigOriginal.length), pySigOriginal.samplingFrequency, False)
        pySignal_oneAtom.add(pyAtom)
        
        approximant, decay = MP.MP(pySignal_oneAtom, pyRandomDico, 10, 10, False, False)
#        approximant.plotTF()
        plt.subplot(212)
        plt.plot(pySignal_oneAtom.dataVec)
        plt.plot(approximant.recomposedSignal.dataVec)
        plt.plot(pySignal_oneAtom.dataVec - approximant.recomposedSignal.dataVec)
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Non aligned atom with LOMP : SRR of " + str(int(approximant.computeSRR())) + " dB in " + str(approximant.atomNumber) + " iteration")  
        plt.xlabel("samples")
#        plt.savefig(figPath+"nonAlignedAtom_MP_vs_LOMP" + ext)
#        plt.show()
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        
        
        del approximant
        
        print "Testing Real signal with MP"
        approximant, decay  = MP.MP(pySigOriginal, pyDico, 10, 10, False)
#        approximant.plotTF()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.dataVec[4096:-4096])
        plt.plot(approximant.recomposedSignal.dataVec[4096:-4096])
        plt.plot(pySigOriginal.dataVec[4096:-4096] - approximant.recomposedSignal.dataVec[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell signal with MP : SRR of " + str(int(approximant.computeSRR()))+ " dB in " + str(approximant.atomNumber) + " iteration")        
#        plt.show()
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        del approximant
        
        print "Testing Real signal with RandomMP"
#        pyCCDico = 
        approximant  = MP.MP(pySigOriginal, pyRandomDico, 10, 10, False, False)[0]
#        approximant.plotTF()
        plt.subplot(212)
        plt.plot(pySigOriginal.dataVec[4096:-4096])
        plt.plot(approximant.recomposedSignal.dataVec[4096:-4096])
        plt.plot(pySigOriginal.dataVec[4096:-4096] - approximant.recomposedSignal.dataVec[4096:-4096])
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Bell Signal with LOMP : SRR of " + str(int(approximant.computeSRR())) + " dB in " + str(approximant.atomNumber) + " iteration")  
        plt.xlabel("samples")
#        plt.savefig(figPath+"RealSignal_MP_vs_LOMP_nbIt10"+ext)
        print " Approx Reached : " , approximant.computeSRR() , " dB in " , approximant.atomNumber , " iteration"
        
        
        del approximant
        print "comparing results and processing times for long decompositions"
        t0 = time.clock()
        approx1, decay  =  MP.MP(pySigOriginal, pyDico, 20, 500, False)
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pySigOriginal.dataVec[12288:-12288])
        plt.plot(approx1.recomposedSignal.dataVec[12288:-12288])
        plt.plot(pySigOriginal.dataVec[12288:-12288] - approx1.recomposedSignal.dataVec[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("Bell Signal with MP : SRR of " + str(int(approx1.computeSRR())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1-t0) + "s")  
#        plt.show()
        print " Approx Reached : " , approx1.computeSRR() , " dB in " , approx1.atomNumber , " iteration and: " , t1-t0 , " seconds"
        
        t2 = time.clock()
        approx2 =  MP.MP(pySigOriginal, pyRandomDico, 20, 500, False, False)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pySigOriginal.dataVec[12288:-12288])
        plt.plot(approx2.recomposedSignal.dataVec[12288:-12288])
        plt.plot(pySigOriginal.dataVec[12288:-12288] - approx2.recomposedSignal.dataVec[12288:-12288])
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Bell Signal with LOMP : SRR of " + str(int(approx2.computeSRR())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3-t2) + "s")  
#        plt.savefig(figPath+"RealSignal_MP_vs_LOMP_SRR20"+ext)
        print " Approx Reached : " , approx2.computeSRR() , " dB in " , approx2.atomNumber , " iteration and: " , t3-t2 , " seconds"
        
        del approx1 , approx2
        del pySigOriginal
        print "comparing results and processing times for long decompositions of white gaussian noise"
        pyNoiseSignal = Signal.pymp_Signal(0.5*random.random(5*16384), 44100, False)
        pyNoiseSignal.pad(16384)
        t0 = time.clock()
        approx1, decay  =  MP.MP(pyNoiseSignal, pyDico, 10, 500, False, True)
        t1 = time.clock()
        plt.figure()
        plt.subplot(211)
        plt.plot(pyNoiseSignal.dataVec[16384:-16384])
        plt.plot(approx1.recomposedSignal.dataVec[16384:-16384])
        plt.plot(pyNoiseSignal.dataVec[16384:-16384] - approx1.recomposedSignal.dataVec[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))
        plt.title("White Noise signal with MP : SRR of " + str(int(approx1.computeSRR())) + " dB in " + str(approx1.atomNumber) + " iteration and " + str(t1-t0) + "s")  
#        plt.show()
        print " Approx Reached : " , approx1.computeSRR() , " dB in " , approx1.atomNumber , " iteration and: " , t1-t0 , " seconds"
        
        t2 = time.clock()
        approx2  =  MP.MP(pyNoiseSignal, pyRandomDico, 10, 500, False, True)[0]
        t3 = time.clock()
        plt.subplot(212)
        plt.plot(pyNoiseSignal.dataVec[16384:-16384])
        plt.plot(approx2.recomposedSignal.dataVec[16384:-16384])
        plt.plot(pyNoiseSignal.dataVec[16384:-16384] - approx2.recomposedSignal.dataVec[16384:-16384])
        plt.legend(("original", "approximant", "resisual"))        
        plt.title("Noise Signal with LOMP : SRR of " + str(int(approx2.computeSRR())) + " dB in " + str(approx2.atomNumber) + " iteration and " + str(t3-t2) + "s")  
        plt.xlabel("samples")
#        plt.savefig(figPath+"NoiseSignal_MP_vs_LOMP_SRR20"+ext)
        print " Approx Reached : " , approx2.computeSRR() , " dB in " , approx2.atomNumber , " iteration and: " , t3-t2 , " seconds"
                

class py_MPTest3(unittest.TestCase):
    """ this time we decompose a longer signal with the Mp_LongSignal : result in an enframed signal """
    def runTest(self):
        filePath = audioFilePath + "Bach_prelude_4s.wav"
        
        # Let us load a long signal: not loaded in memory for now, only segment by segment in the MP process
        mdctDico = [256, 1024, 8192];
        frameSize = 5*8192
        
        # Signal buidling
        originalSignal =  Signal.pymp_LongSignal(filePath, frameSize, True)
        
        # dictionaries
        pyCCDico = pymp_MDCTDico.pymp_LODico(mdctDico) 
        pyDico = pymp_MDCTDico.pymp_MDCTDico(mdctDico ) 
        
        # Let's feed the proto 3 with these:
        # we should get a collection of approximants (one for each frame) in return
        xmlOutPutDir = '../../Approxs/Bach_prelude/LOMP/'
        approximants , decays = MP.MP_LongSignal(originalSignal, pyCCDico, 10, 100, False, True, xmlOutPutDir)
        
#        del approximants
#        xmlOutPutDir = '../Approxs/Bach_prelude/MP/'
#        approximants , decays = MP.MP_LongSignal(originalSignal, pyDico, 5, 100, False, True, xmlOutPutDir)
#        
        # concatenate all segments to retrieve the global approximation
#        recomposedData = zeros(8192)
#        for segIdx in range(len(approximants)) :       
#            recomposedData = concatenate()
            
        self.assertEqual(len(approximants), originalSignal.segmentNumber)
        
        fusionnedApprox = Approx.FusionApproxs(approximants)
        self.assertEqual(fusionnedApprox.samplingFrequency, originalSignal.samplingFrequency)
#        self.assertEqual(fusionnedApprox.length, originalSignal.length )
        plt.figure
        fusionnedApprox.plotTF()
#        plt.show()
        
        
if __name__ == '__main__':
    import matplotlib
    print matplotlib.__version__
    
    _Logger = pymp_Log.pymp_Log('pymp_test', level = 3 , imode=False)
    _Logger.info('Starting Tests');
    suite  = unittest.TestSuite()
    
    suite.addTest(py_MPTest3())
    suite.addTest(py_MPTest())
    suite.addTest(py_MPTest2())
    suite.addTest(pymp_ApproxTest())
    suite.addTest(pymp_AtomTest())
    suite.addTest(pymp_DicoTest())
    suite.addTest(pymp_BlockTest())
    suite.addTest(pymp_Signaltest())
##    
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    plt.show()
    _Logger.info('Tests stopped');
    