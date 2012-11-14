
PyMP Tutorial
=============

This Tutorial will describe how pymp works
for any question feel free to contact me (firstname.lastname@gmail.com)

Manipulating Signals
--------------------
First thing you may want to do is load, analyse, plot and write signals. These operations are handled using the
:class:`.pymp_Signal` class. 

Reading and Writing
*******************
:class:`.pymp_Signal` has a static method called :func:`.InitFromFile` that can be used to load a wav from from disk. 
This is internally done using the **wave** python module. Assuming your current directory is pymp/src you can 
load a signal doing::

>>> from Classes import *
>>> myPympSignal =  pymp_Signal.InitFromFile('data/glocs.wav')

The :class:`.pymp_Signal` object `myPympSignal` mainly wraps the content of *glocs.wav* as a numpy array and descriptors such as
sampling frequency, sample format, etc.. you can access the samples directly as a numpy array by doing::

>>> myPympSignal.dataVec

Alternatively, you can visualize the data using the :func:`plot` function::

>>> myPympSignal.plot()



