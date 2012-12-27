"""

"""

from Classes import *

myPympSignal =  pymp_Signal.InitFromFile('data/glocs.wav',debugLevel=3)
print myPympSignal

print myPympSignal.dataVec

#myPympSignal.plot()

#myPympSignal.write('newDestFile.wav')

# editing

print 'Before cropping Length of ' , myPympSignal.length
myPympSignal.crop(0 , 2048);
print 'After cropping Length of ', myPympSignal.length


from Classes import *
from numpy import ones
newSig = pymp_Signal.pymp_Signal(ones((8,)), 1);
newSig.dataVec
print "Padding"
newSig.pad(4)
newSig.dataVec
print "De-Padding"
newSig.depad(4)
newSig.dataVec

from Classes.mdct import pymp_MDCTDico , pymp_MDCTAtom
pyDico = pymp_MDCTDico.pymp_MDCTDico([128,1024,8192]);
myPympSignal =  pymp_Signal.InitFromFile('data/glocs.wav',forceMono=True)
pyApprox = pymp_Approx.pymp_Approx(pyDico, [], myPympSignal);


pyApprox.addAtom(pymp_MDCTAtom.pymp_MDCTAtom(256, 1, 256, 10, 8000, 1))

pyApprox.computeSRR()
