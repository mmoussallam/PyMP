'''
Created on 14 nov. 2012

@author: mmoussallam
'''

from distutils.core import setup, Extension
# First step: it needs to install the C extension module
import os, platform,sys
import commands
import string

import numpy.core
NUMPYDIR = os.path.dirname(numpy.core.__file__)
libraries=['fftw3']
include_dirs = [os.path.join(NUMPYDIR, r'include/numpy')]
library_dirs = ['/usr/lib/openmpi/lib/openmpi/']

ext_module = Extension('parallelProjections',
                libraries = libraries,
                include_dirs = include_dirs,
                library_dirs=library_dirs,
                sources = ['PyMP/src/parProj.c','PyMP/src/parallelProjections.c'],                                  
                extra_compile_args =['-fopenmp','-fPIC','-DDEBUG=0'],
                extra_link_args = ['-lgomp'])


setup(name='PyMP',
      version='1.0',
      description='Python Matching Pursuit Modules',
      author='Manuel Moussallam',
      author_email='manuel.moussallam@gmail.com',
      url='https://github.com/mmoussallam/PyMP',
      #package_dir = {'': 'src'},
      packages=[ 'PyMP.tools','PyMP.mdct','PyMP.mdct.random'],
      py_modules = ['PyMP.mp','PyMP.base','PyMP.approx','PyMP.signals','PyMP.mp_cmd','PyMP.log','PyMP.win_server','PyMP.mp_coder'],
      ext_modules = [ext_module] 
     )

