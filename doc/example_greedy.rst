PyMP Tutorial 3. Greedy decomposition variants
==============================================

There are many variants of the MP algorithm. Only a few of them are directly implemented in PyMP.
In the first tutorial, we saw some variants of the selection step (e.g. RSSMP). Here we're interested 
in the update step. In order to use this variants, we use the :func:`mp.greedy` function of the :mod:`mp` 
module.

Here's a example of use with a simple multiscale MDCT dictionary and the standard MP algorithm:

>>> import numpy as np
>>> from PyMP.mdct import Dico
>>> from PyMP import mp, Signal
>>> sig = Signal('data/ClocheB.wav', mono=True)  # Load Signal
>>> sig.crop(0, 4.0 * sig.fs)     # Keep only 4 seconds
>>> # atom of scales 8, 64 and 512 ms
>>> scales = [(s * sig.fs / 1000) for s in (8, 64, 512)]
>>> sig.pad(scales[-1])
>>> mp_dico = Dico(scales)
>>> # Launching decomposition, stops either at 20 dB of SRR or 2000 iterations
>>> mp_approx, mp_decay = mp.greedy(sig, mp_dico, 20, 2000, pad=False, update='mp')

This is equivalent to using the :func:`.mp.mp` method. However, the additionnal *update* keyword
argument allows to change the nature of the update step. 

Orthogonal Matching Pursuit
***************************
The principle of OMP is to ensure that at each iteration, chosen atom weights are the best possible
ones (in a least squares sense).  An orthogobnal projection of the residual onto the subspace spanned
by all previously selected atoms is thus performed. Original paper is [1].
The keyword to use OMP is *omp*.

.. warning::

   OMP is extremely resource consuming. You should not try to use it on 
   signals with lengths greater than a few thousand samples. Try the local version instead or GP.

Gradient Pursuit
****************
Gradient Pursuits have been introduced by Blumensath and Davis in [2]. The idea is to replace the
costly orthogonal projection by a much more affordable directionnal update step.

.. warning::

   Orthogonal and directionnal pursuits implementation are not compatible (yet) with :class:`.FullDico` is implemented.


Local versions of OMP/GP
************************
Mailhé *et al* [3] have presented variants of OMP and GP that are much more efficient since they only perform
the update in a neighborhood of the last selected atom. 
For the kind of dictionaries that are implemented in PyMP (i.e. time-localized MDCT and Wavelets), this is a great news.
The keyword arguments to use the local OMP is *locomp* and local GP is *locgp*


Cyclic Matching Pursuit
***********************
Coming Soon!

example
*******

The following example build a k-sparse signal and compares decompositions using
MP, locOMP and locGP:

.. literalinclude:: snippets/greedy_snippets.py


Then you can plot the decays

.. plot:: pyplots/OMP_example.py


Bibliography
************
   1 : Pati, Y. C. C., Rezaiifar, R., & Krishnaprasad, P. S. S. Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition (pdfomp_). 
       Proceedings of the 27 th Annual Asilomar Conference on Signals, Systems, and Computers (pp. 40–44). 
   2 : Blumensath, T., & Davies, M. . (2008). Gradient Pursuits. 
       IEEE Transactions on Signal Processing, 56(6), 2370–2382 (pdfgp_).
   3 : Mailhé, B., Gribonval, R., Vandergheynst, P., & Bimbot, F. (2011). Fast orthogonal sparse approximation algorithms over local dictionaries. 
       Signal Processing vol 91(12) (2822-2835) (pdfloc_).

.. _pdfomp: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.55.1254 
.. _pdfgp: http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4480155&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4480155
.. _pdfloc: http://www.sciencedirect.com/science/article/pii/S0165168411000053