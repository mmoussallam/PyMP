from distutils.core import setup, Extension

import os


''' You need to locate the following header files:
    - Python.h            % Mandatory for using Python objects
    - ndarrayobject.h     % Mandatory for using numpy arrays
    - omp.h               % Mandatory for parallelization
    - fftw3.h             % Mandatory for fast fft calculation
    - parProj.h           % header of libfastPursuit for parallelized projections
    
    everything should be done for you, roblems will most probably arise if the omp library cannot 
    be located: edit the *library_dirs* variable accordingly
    '''
    
import numpy.core
NUMPYDIR = os.path.dirname(numpy.core.__file__)
libraries=['fftw3']
include_dirs = [os.path.join(NUMPYDIR, r'include/numpy')]
library_dirs = ['/usr/lib/openmpi/lib/openmpi/']

ext_module = Extension('parallelProjections',
                libraries = libraries,
                include_dirs = include_dirs,
                library_dirs=library_dirs,
                sources = ['parProj.c','parallelProjections.c'],                                  
                extra_compile_args =['-fopenmp','-fPIC','-DDEBUG=0'],
                extra_link_args = ['-lgomp'])


setup (name = 'parallelProjections',
       version = '0.9',
       description = 'This is a package used for accelerating PyMP using parallelized C computations of projections',
       author = 'Manuel Moussallam',
       author_email = 'manuel.moussallam@gmail.com',
       url = 'http://docs.python.org/extending/building',       
       long_description = '''
        This package is used in PyMP to enhance computing performances of MDCT and MCLTs. Please refer to the documentation file provided
        or contact manuel.moussallam@gmail.com for issues
        ''',
       ext_modules = [ext_module] )


