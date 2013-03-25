Classes Package Documentation:
------------------------------

The root PyMP package contains all basic pymp objects such as :class:`.Signal` ,  :class:`.Approx` and abstract classes :class:`.Dico`, 
:class:`.Atom` and :class:`.Block`
    
		
Signal and Approximation Handles
********************************
	
Module signals
--------------

The main class is :class:`.Signal`, it can be instantiated from a numpy array
using the main constructor (multichannel is allowed).

It can also be created from a file on the disk using the path as argument
in the constructor

Longer Signals are handled with :class:`LongSignal` objects.
   

   .. automodule:: PyMP.signals
      :members: Signal, LongSignal

Module approx
-------------

The main class is :class:`.Approx`. It describe an approximant of a signal. 
This object handles MP constructed approximations. It isis quite similar in nature to the MPTK [1] 
Book object but allows further manipulations in python such as plotting the time frequency distribution of atoms 

An approximant can be manipulated in various ways. Obviously atoms can be edited by several methods among which
:func:`PyMP.Approx.add` ,  :func:`.remove`  and :func:`.filterAtom`

Measure of distorsion can be estimated by the method :func:`.Approx.compute_srr`

Approx objets can be exported in various formats using
:func:`to_dico` ,
:func:`to_sparse_array` ,
:func:`to_array` ,
:func:`write_to_xml` ,
:func:`dumpToDisk` ,

and reversely can be recovered from these formats

A useful plotting routine, :func:`plot_tf` is provided to visualize atom distribution in the time frequency plane
Also an experimental 3D plot taking the atom iteration number as a depth parameter
:func:`plot_3d`

	.. automodule:: PyMP.approx
		:members: Approx

Abstract MP classes
*******************
	.. automodule:: PyMP.base
		:members: BaseAtom, BaseDico, BaseBlock

