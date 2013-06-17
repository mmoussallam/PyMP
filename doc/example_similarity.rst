PyMP Tutorial 4. Similarity Measures
====================================
There are many ways to assess similarities between two signals :math:`x` and :math:`y`. 
Here we're particularly interested in measuring similarity using their sparse representations
:math:`\tilde{x}` and :math:`\tilde{y}`. More details can be found in Chapter 7 of [2] (in french)

Signal Decomposition
********************
We will use a musical example, the first few seconds of Bach's first prelude in C played by G. Gould
That is available in the data directory of PyMP. It consists in a piano melody that is played two times.
Let us load the audio, cut the signal in two and decompose it:

.. note::

   This example assumes you have set up a PYMP_PATH environment variable targetting your
   PyMP directory.
   
>>> import os
>>> import os.path as op
>>> import numpy as np
>>> from PyMP.mdct import Dico
>>> from PyMP import mp, Signal
>>> # load
>>> signal = Signal(op.join(os.environ['PYMP_PATH'],'data/Bach_prelude_4s.wav'), mono=True)
>>> # cut in two
>>> sig_occ1 = signal[:signal.length/2]
>>> sig_occ2 = signal[signal.length/2:]
>>> dico = Dico([128,1024,8192])
>>> target_srr = 5
>>> max_atom_num = 200
>>> # MP decompositions
>>> app_1, _ = mp.mp(sig_occ1, dico, target_srr, max_atom_num)
>>> app_2, _ = mp.mp(sig_occ2, dico, target_srr, max_atom_num)

.. plot:: pyplots/bach_2_plot.py


Measuring Similarity
********************
Similarity can be measured in the representation domain virtually by any distance metric between two vectors
The simplest examples one can think of are the euclidean distance, and the hamming distance. Those 
are defined in the tools package

>>> from PyMP.tools.Misc import euclid_dist, hamming_dist

Getting a sparse vector from a :class:`.Approx` object is quite simple:

>>> sp_vec_1 = app_1.to_array()[0]
>>> sp_vec_2 = app_2.to_array()[0]
>>> print "%1.5f, %1.5f"%(euclid_dist(sp_vec_1,sp_vec_2), hamming_dist(sp_vec_1,sp_vec_2)) 
-5.27440, 0.75510


Information Distance
********************
Following the idea of the paper [1], we can build proxies of information distances between the two
sparse representation by quantifying the amount of *complexity* that is required to transform one into another

The underlying idea is that if :math:`\tilde{x}` and :math:`\tilde{y}` are similar, then joint coding should be effective.
In particular, one would guess that atoms in the support of :math:`\tilde{x}` are efficient for an approximation :math:`\tilde{x}` of :math:`y`.
A simple version of this paradigm is implemented in the mp_coder module:

>>> from PyMP.mp_coder import joint_coding_distortion

This metric measures the signal to residual ratio that is achieved by using the atoms of the reference to approximate the target
the best time shift :math:`\tau^{n}` for each of the m atoms of :math:`\tilde{x}`.

.. math:: D_{R}(x,y) = 10\log_{10}(\frac{\|\hat{y}-y\|_{2}^{2}}{\|\hat{y}\|_{2}^{2}}) 


such that:
 
.. math:: R(\hat{y}|\tilde{x})\leq R


where :math:`R` is a fixed number of bits that corresponds to the amount of information that is 
needed to *transform* the reference into the target using

.. math:: R(\hat{y}_{m}|\tilde{x}_{m})=\sum_{n=0}^{m-1}\log_{2}\tau^{n}

One can then further normalize by the reference approximation srr to have similarity metrics smaller than one

>>> max_rate = 1000 # maximum bitrate allowed (in bits)
>>> search_width = 1024 # maximum time shift allowed in samples
>>> info_dist = joint_coding_distortion(sig_occ2, app_1, max_rate, search_width)
>>> info_dist_rev = joint_coding_distortion(sig_occ1, app_2, max_rate, search_width)
>>> print "%1.5f  - %1.5f"%(info_dist/target_srr, info_dist_rev/target_srr)
0.99585  - 0.99650

.. note::

   This metric is not symmetric


Building a self-similarity matrix
*********************************
Let us use this method to compute a similarity matrix for a longer version of the prelude
that lasts 40 seconds. First let's load it into a :class:`.LongSignal` object:

>>> from PyMP.signals import LongSignal
>>> seg_size = 5*8192 # roughly 1 seconds at 44100 Hz
>>> long_signal = LongSignal(op.join(os.environ['PYMP_PATH'],'data/Bach_prelude_40s.wav'), seg_size, mono=True, Noverlap=0.5)
>>> long_signal.n_seg
89

For this demonstration, the first 15 seconds are sufficient. We can thus limit the number of segments

>>> long_signal.n_seg = 32

We want each segment to be decomposed up to a certain srr. To do that we use the :func:`.mp_long` utility:

>>> # decomposing the long signal
>>> apps, _ = mp.mp_long(long_signal, dico, target_srr, max_atom_num)

and we end up with a list of :class:`.Approx` objects.

>>> apps[0], apps[1]
(Approx Object: 28 atoms, SRR of 5.00 dB, Approx Object: 26 atoms, SRR of 5.06 dB)

Now to build a similarity matrix, we compute pairwise information distances. To accelerate
we can prune the computations. For instance, if the srr after 15 atoms is not positive, then 
we can consider the factorization to have failed and stop the process.
Also we can limit the pairwise comparisons to causal ones, that is we only try to factorize a segment using previously observed ones
You should be able to get the following similarity matrix:

.. plot:: pyplots/build_sim_matrix.py

.. note::

   One may wonder why scores higher than one can be observed on the diagonal.
   Since we used a simple :class:`.Dico` object, we ran MP on a coarse Time-Frequency grid.
   Here the joint coder optimizes the time localization of the atoms, thus reaching better approximation levels.

You can further play with the parameters, in particular the overlap ratio, the segment sizes etc..
And that's about it.

Bibliography
************
   1 : Moussallam, M., Daudet, L., & Richard, G. Audio Signal Representations for Factorization in the sparse Domain. 
       ICASSP 2011 (pp. 513–516). (pdficassp11_)
   2 : Moussallam, M. Représentation redondantes et Hiérarchiques pour l’archivage et la compression de scènes sonores.
       PhD Thesis Telecom ParisTech 2012 (pdfthesis_)
       
.. _pdficassp11: http://manuel.moussallam.net/docs/moussallam_icassp11.pdf 
.. _pdfthesis: http://manuel.moussallam.net/docs/manuscrit_final.pdf

 

