'''
Created on 18 aout 2012

@author: manumouss
'''
import unittest
import sys, os, commands
import cProfile
from MatchingPursuit import py_pursuit_Approx
class Test(unittest.TestCase):



    def testMPCmd(self):
        """ Tests the call to MPCmd on a short musical excerpt """

        mpPath = os.path.realpath('../MPCmd.py');

        print "Appending ", mpPath[:-8] , " to current pythonpath"
        sys.path.append(mpPath)

        print sys.path
        print os.path.realpath('./')
        print commands.getoutput('python ../MPcmd.py -f ../../../data/orchestra.wav --debug=1 -l 5 -m 100')


#        approx  = py_pursuit_Approx.loadFromDisk('dumpTestApprox');

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testMPCmd']

    unittest.main()
