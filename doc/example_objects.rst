Manipulating Signals
--------------------
First thing you may want to do is load, analyse, plot and write signals. These operations are handled using the
:class:`.Signal` class. 

Reading and Writing
*******************
You can use the constructor of the :class:`.Signal` class to load a wav file from the disk. 
This is internally done using the **wave** python module shipped with the standard library. Assuming you've correctly set the PYTHONPATH variable so that it contains
the pymp/PyMP/ directory, you can simply type in a python shell:

>>> from PyMP import signals
>>> myPympSignal = signals.Signal('data/glocs.wav')

As for many pymp Objects, you can specify a debug level that manages info and warning printing degrees. Try for example::

>>> myPympSignal =  signals.Signal('data/glocs.wav',debug_level=3)
>>> print myPympSignal
Signal object located in data/glocs.wav
        length: 276640
        energy of 271319943.000
        sampling frequency 32000
        number of channels 1

The :class:`.Signal` object `myPympSignal` wraps the content of *glocs.wav* as a numpy array and descriptors such as
sampling frequency, sample format, etc.. you can access the samples directly as a numpy array by doing::

>>> myPympSignal.data

Meaning you can also create a :class:`.Signal` object directly from a numpy array::

>>>  myPympSignal =  Signal(np.zeros((1024,), Fs=8000)

where `Fs` is the sampling frequency (default is zero)
At any moment you can visualize the data using the :func:`plot` function::

>>> myPympSignal.plot()

Writing a signal is also quite straightforward::

>>> myPympSignal.write('newDestFile.wav')

Signal Edition
**************
Often you need to edit signals, e.g. crop them or pad them with zeroes on the borders, this can be done easily::

>>> print 'Before cropping Length of ' , myPympSignal.length
>>> myPympSignal.crop(0 , 2048)
>>> print 'After cropping Length of ', myPympSignal.length

will plot the new signal length. 

Another way is to use:

>>> subSig = myPympSignal[0 , 2048]
>>> print subSig
Signal object located in 
        length: 2048
        energy of 0.000
        sampling frequency 32000
        number of channels 1

Revesely you can pad signals with zeroes, this is done on both sides with pad and depad methods.
For example, we can create a signal with only ones and pad it with zeroes on the edges::

>>> newSig = signals.Signal(ones((8,)), 1)
>>> newSig.data
>>> print "Padding"
>>> newSig.pad(4)
>>> newSig.data

Removing the zeroes is also straightforward::

>>> print "De-Padding"
>>> newSig.depad(4)
>>> newSig.data


Manipulating Approximation objects
----------------------------------

.. note::
   
   :class:`.Approx` objects are the equivalent of *Book* objects in MPTK. 
   They handle the approximation of a signal on a given dictionary. 

Creation
********

A trivial creation takes no further arguments.::
 
>>> from PyMP.approx import Approx 
>>> pyApprox = Approx() 
 
Basically, an approximant is just a collection of atoms, this means we can enrich this object py adding some atoms to it.
For example we can add 3 MDCT atoms of different scales, time and frequency localization to obtain an approximant
as in the following example:

.. plot:: pyplots/approx_ex1.py

This example use the :class:`.Atom` objects. The long atom (2048 samples or 256 ms at a sampling rate of 8000 Hz) is built using the command:: 

>>> from PyMP.mdct.atom import Atom
>>> atomLong = Atom(2048, 1, 0, 40, 8000, 1)

where we have specified its size, amplitude (Deprecated, always put 1 in there) , time localization (0) , frequency bin (40 which corresponds to 156 Hz) and mdct_coefficient value (1)
then the atom's waveform is synthesized using internal routine and used to create a :class:`.pymp_Approx` object::

>>> atomLong.synthesize()
>>> approx  = Approx(None, [], None, atomLong.length, atomLong.fs)

Other atoms can be added ::

>>> approx.add(Atom(256, 1, 256, 10, 8000, 1))

Approximation in a MP context
*****************************

Although you can manipulate :class:`.Approx` objects on their own, it is much more interesting to link them to existing signals and to a dictionary.
For example, let us define a dictionary as a union of 3 MDCT basis::

>>> from PyMP.mdct import dico
>>> pyDico = dico.Dico([128,1024,8192])

We can now create an approximation of a specified signal on this dictionary this way::

>>> myPympSignal = Signal('data/glocs.wav',forceMono=True)
>>> pyApprox = Approx.(pyDico, [], myPympSignal)

for now this approximation is empty (the *pyApprox.atoms* list is empty). But we can still add an atom to it::

>>> pyApprox.add(Atom(256, 1, 256, 10, 8000, 1))

Now we have a reference signal and an approximant of it, we can evaluate the quality of the approximation using the Signal to Residual Ratio (SRR):

>>> print pyApprox.compute_srr()
-116.6369995336029

Since we picked a random atom with no link to the signal, the SRR (in dB) is very poor. It will be much better when MP select atoms based on their correlation to the signal

