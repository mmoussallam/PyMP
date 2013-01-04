from os.path import join
import numpy
import os
import numpy.core

NUMPYDIR = os.path.dirname(numpy.core.__file__)
libraries = ['fftw3']
include_dirs = [os.path.join(NUMPYDIR, r'include/numpy'),'/usr/local/include']
library_dirs = ['/usr/lib/openmpi/lib/openmpi/', "/usr/local/lib"]


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import platform
    print platform.architecture()[0]
    if (platform.architecture()[0] == '32bit'):
        plt_str = '-DX=1'
    else:
        plt_str = '-DX=2'

    config = Configuration('PyMP', parent_package, top_path)

    config.add_extension('parallelProjections',
                         libraries=libraries,
                         include_dirs=include_dirs,
                         library_dirs=library_dirs,
                         sources=[join('src', 'parProj.c'),
                                  join('src', 'parallelProjections.c')],
                         extra_compile_args=['-fopenmp', '-fPIC', '-DDEBUG=0',plt_str],
                         extra_link_args=['-lgomp',])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
