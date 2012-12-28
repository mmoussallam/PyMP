from os.path import join
import numpy
import os
import numpy.core

NUMPYDIR = os.path.dirname(numpy.core.__file__)
libraries = ['fftw3']
include_dirs = [os.path.join(NUMPYDIR, r'include/numpy')]
library_dirs = ['/usr/lib/openmpi/lib/openmpi/', "/Users/alex/work/src/fftw-3.2.2/.libs/"]


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('PyMP', parent_package, top_path)

    config.add_extension('parallelProjections',
                         libraries=libraries,
                         include_dirs=include_dirs,
                         library_dirs=library_dirs,
                         sources=[join('src', 'parProj.c'),
                                  join('src', 'parallelProjections.c')],
                         extra_compile_args=['-fopenmp', '-fPIC', '-DDEBUG=0'],
                         extra_link_args=['-lgomp'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
