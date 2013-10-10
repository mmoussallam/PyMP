
What is PyMP?
=============

PyMP is a collection of classes and scripts mostly written in Python
during my PhD to perform fast decomposition of audio signals on
multiscale time-frequency dictionaries.

For now, it can only handle MDCT-based dictionaries

Architecture
------------

PyMP is very much structured like the MPTK toolbox from INRIA.
Some of my research required that I would rewrite the core
decomposition algorithm in order to design variants of it. In
particular, the rewriting allows the use of Randomized Algorithms,
such as the Matching Pursuit with Random Sequences of subdictionaries
(RSSMP [1]). If you don't intend to use this kind of variants, I
recommend you use one of the existing implementation (indeed MPTK) as
it can be fairly faster and more thoroughly tested that this piece of
software.

In the opposite case let's have a look at how this is built. The archtecture is built upon these objects:

.. uml:: /home/manu/workspace/PyMP/PyMP/base.py


A few notions
*************

This section only recall MP in order to describe the meaning of PyMP
methods and classes. This is NOT an introduction to greedy algorithms.

MP will decompose a signal :math:`x` into a linear combination of atoms :math:`\phi` taken from a dictionary :math:`\Phi` by iteratively selecting them according to their correlation to the
signal. Thus, the algorithm builds an *approximant* :math:`\tilde{x}_{m}` of :math:`x` in :math:`m` iterations such that:

.. math:: \tilde{x}_{m}=\sum_{n=0}^{m-1}\alpha_{n}\phi_{\gamma^{n}}\approx x

Implementation
**************

PyMP provide an Object-Oriented implementation of the MP algorithm in the following manner:

	- Signals (such as :math:`x`) are handled through :class:`.Signal` objects

	- Dictionaries (such as :math:`\Phi`) are handled through :class:`.BaseDico` objects

	- Approximants (such as :math:`\tilde{x}_{m}`) are handled through :class:`.Approx` objects

The algorithm itself is performed using function from the :mod:`.mp` module. A comprehensive view of how it works is provided in the next topic.

How to have it work?
--------------------

Installation
************
Linux/Mac OS
............
PyMP is available from Python Package repository with the command: 
   *pip install -U pypursuit*

Alternatively a 64 bit archive is available here:
   http://manuel.moussallam.net/PyMP/windist/PyPursuit-1.1.0.linux-x86_64.tar.gz

Windows
.......
For windows platform, a 32-bit installer can be found here: 
   http://manuel.moussallam.net/PyMP/windist/PyPursuit-1.1.0.win32-py2.7.exe 

Building from source
********************
Download the latest sources from github: https://github.com/mmoussallam/PyMP
On unix-based systems it should be really easy provided you have all the required libraries, simply run ::

   $make all

If it's not sufficient, please follow instructions below

.. warning::

	You will need to have *fftw*  and *openMP* libraries installed and header accessible. On Debian-based OS you'll need the following packages:

		- libfftw3-dev

		- python-dev

	Installation has been succesfully tested on linux, Mac OS and windows 32 bit.
	Since Microsoft compiler (at least the free version of it) has bad handling of 64 bits libraries
	I recommend the use of cygwin on Windows Platform, although it appears matplotlib installation on cygwin
	may not be as straightforward as it should. 

Additionnaly you'll want to have the following Python packages installed:

		- Numpy

		- Matplotlib

		- Scipy


PyMP is mainly a collection of pure python modules, which installation is quite traditionnal.
However, in order to accelerate the inner product computations, it uses a low-level pure C library that is
used through a python C extension module.

Which means there are two installation steps:

	- Build and install the Python C extension module called *parallelProjections*

	- Build and install the pure Python modules among which:

			- :mod:`.base`: a module describing abstract atom, dictionary and block objects

			- :mod:`.signals`: a module containing the `.Signal` class and routines to manipulate them

			- :mod:`.approx`: a module containing the `.Approx` class and routines to manipulate them

			- :mod:`.tools`: a collection of tools

			- :mod:`.tests`: a package of tests

.. note::

	Hopefully you won't need to perform these operations, it will be done for you by executing by the *setup.py* script
	in the root directory. Simply run::

    $python setup.py install

	And (provided all headers and libraries are present and accessible) it should be fine.
	This should compile the C extension and install all sources and packages in your dist-package
	local directory. You may need to have writing rights to perform this operation (e.g. using *sudo*).



Bibliography
------------

    [1]. M. Moussallam , L. Daudet , et G. Richard , "Matching Pursuits with Random Sequential Subdictionaries"
    Signal Processing, vol. 92, pp. 2532-2544, 2012. pdf_ .

.. _pdf: http://dx.doi.org/10.1016/j.sigpro.2012.03.019

