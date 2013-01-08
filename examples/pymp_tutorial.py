"""

"""
import numpy as np
from PyMP.mdct import Dico, atom
from PyMP import Signal, approx

sig = Signal('../data/glocs.wav', debug_level=3)
print sig
print sig.data

# sig.plot()
# sig.write('newDestFile.wav')
# editing

print 'Before cropping Length of ', sig.length
sig.crop(0, 2048)
print 'After cropping Length of ', sig.length

sub_sig = sig[0:2048]
print sub_sig

new_sig = Signal(np.ones((8,)), 1)
new_sig.data
print "Padding"
new_sig.pad(4)
new_sig.data
print "De-Padding"
new_sig.depad(4)
new_sig.data


dico = Dico([128, 1024, 8192])
sig = Signal('../data/glocs.wav', mono=True)
app = approx.Approx(dico, [], sig)


app.add(atom.Atom(256, 1, 256, 10, 8000, 1))

app.compute_srr()
print app

