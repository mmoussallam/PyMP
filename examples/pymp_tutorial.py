"""

"""

import signals , approx

myPympSignal =  signals.InitFromFile('data/glocs.wav',debugLevel=3)
print myPympSignal

print myPympSignal.dataVec

#myPympSignal.plot()

#myPympSignal.write('newDestFile.wav')

# editing

print 'Before cropping Length of ' , myPympSignal.length
myPympSignal.crop(0 , 2048);
print 'After cropping Length of ', myPympSignal.length


from numpy import ones
newSig = signals.Signal(ones((8,)), 1);
newSig.dataVec
print "Padding"
newSig.pad(4)
newSig.dataVec
print "De-Padding"
newSig.depad(4)
newSig.dataVec

from mdct import dico , atom
pyDico = dico.Dico([128,1024,8192]);
myPympSignal =  signals.InitFromFile('data/glocs.wav',forceMono=True)
pyApprox = approx.Approx(pyDico, [], myPympSignal);


pyApprox.addAtom(atom.Atom(256, 1, 256, 10, 8000, 1))

pyApprox.computeSRR()
