'''
Created on 14 nov. 2012

@author: mmoussallam
'''


from distutils.core import setup

# First step: it needs to install the C extension module
import os,sys
import commands
import string

info = os.uname();

arch = string.lower(info[0]) + '-' + info[4];
pythonVersion = sys.version[0:3];

#print os.getcwd()
#
targetPath = os.path.abspath('PyMP/parallelProjections/setup.py');
# Unix Systems
if os.name == 'posix':    
    print commands.getoutput('python '+targetPath+' build --verbose')    
    print commands.getoutput('python '+targetPath+' install')  
    print commands.getoutput('mv build/lib.'+arch+'-'+pythonVersion+'/parallelProjections.so ../parallelProjections.so')

elif os.name == 'nt':
    print os.popen('python setup.py install').read()
    
print "---- Test import du module!"
import parallelProjections
print "---- OK"

setup(name='PyMP',
      version='1.0',
      description='Python Matching Pursuit Modules',
      author='Manuel Moussallam',
      author_email='manuel.moussallam@gmail.com',
      url='https://github.com/mmoussallam/PyMP',
      #package_dir = {'': 'src'},
      packages=[ 'PyMP.Tools','PyMP.Classes','PyMP.Tests'],
      py_modules = ['PyMP.MP',]
     )