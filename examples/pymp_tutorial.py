"""

"""
import numpy as np
from PyMP.mdct import dico, atom
from PyMP import signals, approx

myPympSignal = signals.Signal('../data/glocs.wav', debug_level=3)
print myPympSignal
print myPympSignal.data

# myPympSignal.plot()
# myPympSignal.write('newDestFile.wav')
# editing

print 'Before cropping Length of ', myPympSignal.length
myPympSignal.crop(0, 2048)
print 'After cropping Length of ', myPympSignal.length

subSig = myPympSignal[0:2048]
print subSig

newSig = signals.Signal(np.ones((8,)), 1)
newSig.data
print "Padding"
newSig.pad(4)
newSig.data
print "De-Padding"
newSig.depad(4)
newSig.data


pyDico = dico.Dico([128, 1024, 8192])
myPympSignal = signals.Signal('../data/glocs.wav', mono=True)
pyApprox = approx.Approx(pyDico, [], myPympSignal)


pyApprox.add(atom.Atom(256, 1, 256, 10, 8000, 1))

pyApprox.compute_srr()
print pyApprox

