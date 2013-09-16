'''
Created on 14 nov. 2012

@author: mmoussallam
'''

import os
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('PyMP')

    return config

if __name__ == '__main__':
    with open('README.md') as file:
        long_description = file.read()
    
    setup(configuration=configuration,
          name='PyPursuit',
          version='1.1.0',
          description='Pythonic Matching Pursuit Modules',
          author='Manuel Moussallam',
          author_email='manuel.moussallam@gmail.com',
          long_description = long_description,
          url='https://github.com/mmoussallam/PyMP',
#          install_requires=['numpy',],
          #package_dir = {'': 'src'},
          package_data={'': ['license.txt','Makefile','README.md']},
          packages=['PyMP.tools', 'PyMP.mdct', 'PyMP.mdct.rand','PyMP.mdct.joint'],
          py_modules=['PyMP.mp', 'PyMP.base','PyMP.baserand', 'PyMP.approx', 'PyMP.signals',
              'PyMP.mp_cmd', 'PyMP.log', 'PyMP.win_server', 'PyMP.mp_coder'],
         )
