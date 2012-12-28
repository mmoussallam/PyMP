
PyMP Tutorial 1. Signal decomposition 
=====================================

This Tutorial will describe how pymp works, and mostly how to use it to perform audio signal decompositions on multiscale MDCT dictionaries
for any question feel free to contact me (firstname.lastname@gmail.com)
Documentation for the modules that we're using here is provided in the next sections. 

Manipulating Signals
--------------------
First thing you may want to do is load, analyse, plot and write signals. These operations are handled using the
:class:`.Signal` class. 

Reading and Writing
*******************
You can use the constructor of the :class:`.Signal` class to load a wav file from the disk. 
This is internally done using the **wave** python module shipped with the standard library. Assuming you've correctly set the PYTHONPATH variable so that it contains
the pymp/PyMP/ directory, you can simply type in a python shell:

>>> from PyMP.signals import Signal
>>> myPympSignal = Signal('data/glocs.wav')

As for many pymp Objects, you can specify a debug level that manages info and warning printing degrees. Try for example::

>>> myPympSignal =  Signal('data/glocs.wav',debugLevel=3)

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
Revesely you can pad signals with zeroes, this is done on both sides with pad and depad methods.
For example, we can create a signal with only ones and pad it with zeroes on the edges::

>>> newSig = pymp_Signal.pymp_Signal(ones((8,)), 1)
>>> newSig.data
>>> print "Padding"
>>> newSig.pad(4)
>>> newSig.data

Removing the zeroes is also straightforward::

>>> print "De-Padding"
>>> newSig.depad(4)
>>> newSig.data

Standard MP over a union of MDCT Basis
--------------------------------------

All important objects have been introduced, let us present some examples of what PyMP does.

Standard Algorithm
******************
In this example, the standard algorithm is used to decomposed the glockenspiel signal over a union of 3 MDCT basis::

>>> from PyMP.signals import Signal
>>> from PyMP.approx import Approx
>>> from PyMP.mdct import dico
>>> sizes = [128,1024,8192]
>>> Natom = 1000
>>> pySig = Signal('data/glocs.wav',forceMono=True,doNormalize=True)
>>> pySig.crop(0, 3.5*pySig.samplingFrequency)
>>> pySig.pad(8192)
>>> pySig.data += 0.0001*np.random.randn(pySig.length)
>>> pyDico= dico.Dico(sizes)
>>> approx, decay = mp.mp(pySig, pyDico, 50, Natom)

.. note::
	
	IMPORTANT: the fact that we know it's the standard algorithm that is used is because we choosed a :class:`.Dico` object as dictionary.

First plot (a) is the original glockenspiel waveform. (b) presents the 3 MDCT (absolute values) considered. 
(c) is the reconstructed signal, accessible via::

>>> approx.recomposedSignal

and (d) is the time-frequency plot of the 1000 atoms that have been used to approximate the original glockenspiel signal

.. plot:: pyplots/MP_example.py

You can evaluate the quality of the approximation:

>>> approx.computeSRR()
23.657038395028287

and save the result in various formats (see the :class:`.Approx` documentation)::

>>> approx.recomposedSignal.write('outPutPath.wav')

Locally Optimized MP
--------------------

To run a locally adaptive (or optimized) MP, all we have to do is to pick a :class:`.LODico` object as a dictionary. The internal
routines of its blocks will perform the local optimization so that at our level this is quite transparent:


>>> approxMP, decayMP = mp.mp(pySig, dico.DICO(scales) ,srr,nbAtom,padSignal=True)
>>> approxLoMP, decayLoMP = mp.mp(pySig, dico.LODICO(scales), srr,nbAtom,padSignal=False)


.. plot:: pyplots/LoMP_example.py

In addition to plotting, we can compare the quality of the approximations, given a fixed number of atoms (here 500):

>>> approxMP.computeSRR() , approxMP.atomNumber
(19.50500304181195, 500)
>>> approxLoMP.computeSRR() , approxLoMP.atomNumber
(23.205178754903638, 500)

The locally adaptive Matching pursuit has yielded a better decomposition (in the sense of mean squared error).
Alternatively one can verify that for a given level of SRR, LoMP will use a smaller number of atoms.

MP with Random Sequential Subdictionaries
-----------------------------------------

RSSMP is explained in the journal paper_ .  

.. _paper: http://dx.doi.org/10.1016/j.sigpro.2012.03.019


Implementation of RSSMP is quite transparent, it's done through the use of a :class:`.RandomDico` object as dictionary::

>>> from PyMP.mdct.random import dico as random_dico
>>> pyRandomDico = random_dico.RandomDico(sizes, 'random')

We can now compare the three strategies in terms of normalized reconstruction error 

.. math::

	10 \log_{10} (\| \tilde{x}_m - x \|^2) -  10 \log_{10} (\| x \|^2)

This gives the following results:

.. plot:: pyplots/RSSMP_example.py

And that's it.  

Approximation objects
---------------------

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

