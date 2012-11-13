from distutils.core import setup, Extension

import os,sys
import commands

info = os.uname();

# CHANGE according to your configuration
arch = info[0] + '-' + info[4];
pythonVersion = sys.version[0:3];

#pythonVersion = '2.7'
#fullfastPursuitPath = commands.getoutput('locate parProj.h')
fastPursuitPath = os.path.abspath('../../lib/fastPursuit/')

if os.name == 'posix':  

    ''' You need to locate the following header files:
        - Python.h            % Mandatory for using Python objects
        - ndarrayobject.h     % Mandatory for using numpy arrays
        - omp.h               % Mandatory for parallelization
        - fftw3.h             % Mandatory for fast fft calculation
        - parProj.h           % header of libfastPursuit for parallelized projections
        
        and fill the includeDirList variable accordingly if any changes in your configuration
        '''
    includeDirList = ['/usr/local/include',
                      '/usr/share/pyshared/numpy/core/include/numpy/',
                      '/usr/lib/pyshared/python'+pythonVersion+'/numpy/core/include/numpy/',
                      '/usr/lib/openmpi/lib/openmpi/',
                       fastPursuitPath+'/src/']


    ext_module = Extension('parallelProjections',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '2')],
                    include_dirs = includeDirList,
                    library_dirs = ['/usr/local/lib'  ],                    
                    sources = ['parallelProjections.c' ],
                    extra_compile_args =['-fopenmp','-fPIC'] ,
                    extra_link_args = ['-lgomp','-lfftw3',fastPursuitPath+'/libfastPursuit.a' ])

# TODO Windows compatibility    
#else:
#    module1 = Extension('computeMCLT',
#                    sources = ['computeMCLT.c' ],
#                    include_dirs = ['C:/Python26/Lib/site-packages/numpy/core/include/numpy/',
#                                    'C:/Users/moussall/Documents/WorkSpace/others/win32/output/inc/',
#                                    'C:/Users/moussall/Documents/WorkSpace/others/win32/Release-DLL/'],
#                    library_dirs = ['C:/Users/moussall/Documents/WorkSpace/others/win32/Release-DLL/'],
#                    platform = ['x64'],
#                    extra_link_args = ['C:/Users/moussall/Documents/WorkSpace/py_pursuit/Libs/libfftw3.lib' , 'C:/Python26/libs/Python26.lib']) # link to fftw3.lib here



setup (name = 'parallelProjections',
       version = '0.9',
       description = 'This is a package used for accelerating PyMP using parallelized C computations of projections',
       author = 'Manuel Moussallam',
       author_email = 'manuel.moussallam@gmail.com',
       url = 'http://docs.python.org/extending/building',
       #package_dir = {'computeMCLT':'/claude/home/moussall/data/workspace/Python/Py_pursuit/libs'},
       long_description = '''
This package is used in PyMP to enhance computing performances of MDCT and MCLTs. Please refer to the documentation file provided
or contact manuel.moussallam@gmail.com for issues
''',
       ext_modules = [ext_module] )
