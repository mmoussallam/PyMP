

A Python Matching Pursuit Implementation developed during my PhD 


What is PyMP?
=============

PyMP is a collection of classes and scripts mostly written in Python
during my PhD to perform fast decomposition of audio signals on
multiscale time-frequency dictionaries. 


Architecture
------------

PyMP is very much structured like the MPTK toolbox from INRIA [#]_ . Some of my research required that I would rewrite the core
decomposition algorithm in order to design variants of it. In
particular, the rewriting allows the use of Randomized Algorithms,
such as the Matching Pursuit with Random Sequences of subdictionaries
(RSSMP [1]). If you don't intend to use this kind of variants, I
recommend you use one of the existing implementation (indeed MPTK) as
it can be fairly faster and more thoroughly tested that this piece of
software. 

In the opposite case let's have a look at how this is built: 



.. [#] 
    XXX: Unknown inset Plain Layout
    


A few notions
-------------

This section only recall MP in order to describe the meaning of PyMP
methods and classes. This is NOT an introduction to greedy algorithm. 

MP will decompose a signal :math:`x` into a linear combination of atoms :math:`\phi` taken from a dictionary :math:`\Phi` by iteratively selecting them according to their correlation to the
signal. Thus, the algorithm builds an *approximant* :math:`\tilde{x}_{m}` of :math:`x` in :math:`m` iterations such that: 

.. math::

    \[ \tilde{x}_{m}=\sum_{n=0}^{m-1}\alpha_{n}\phi_{\gamma^{n}}\approx x \]




How to have it work?
====================

Once you've downloaded and unzipped the source file, here's how to
proceed to install python modules on your machine and start using it? 


Examples
========


Bibliography
============

    [1]. M. Moussallam , L. Daudet , et G. Richard , "Matching Pursuits with Random Sequential Subdictionaries"
    Signal Processing, vol. 92, pp. 2532-2544, 2012.

