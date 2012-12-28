#*****************************************************************************/
#                                                                            */
#                               MPcmd.py                                     */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
#                                                                            */
#                                                                            */
#  This program is free software; you can redistribute it and/or             */
#  modify it under the terms of the GNU General Public License               */
#  as published by the Free Software Foundation; either version 2            */
#  of the License, or (at your option) any later version.                    */
#                                                                            */
#  This program is distributed in the hope that it will be useful,           */
#  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
#  GNU General Public License for more details.                              */
#                                                                            */
#  You should have received a copy of the GNU General Public License         */
#  along with this program; if not, write to the Free Software               */
#  Foundation, Inc., 59 Temple Place - Suite 330,                            */
#  Boston, MA  02111-1307, USA.                                              */
#                                                                            */
# *****************************************************************************
# */

'''
Module MPcmd
============

PyPursuit command line arguments looks as follow:
             -h or --help : print this help

             -f [filePath] : specifies the path to the audiofile, only wav files supported for now

             -w : flag for writing the output (approx and recomposedSignal) to output files

             -o [outPutFilePath] : path for the output

             -m [maxAtomNumber] :  specifies the maximum number of selected atoms

             -s [SRR] : specifies the target Signal-To-Residual Ratio

             -d [dictionarySizes] : specifies the sizes: e.g -d 128,1024,8192

             -t or --type ['mp' | 'LOMP'] : specify the chosen decomposition algorithm

             --debug=[0|1|2|3] : debug level. Default is 0

             -a or --pad pad signal with zeroes , default is 1

             -p or --plot plots the approximation using matplotib

             -l [segmentDuration] : length of the segments

             Example::

                 >>> python MPcmd.py -f sndFile.wav -m 100 -s 10 -d 128,1024,8192 --debug=1 -a -p

'''
import matplotlib.pyplot as plt
import sys
import getopt

import signals, approx
from mdct import dico
from mdct.random import dico as random_dico
import mp



def usage():
    print """Thanks for using Py_Pursuit. All questions please : manuel.moussallam@telecom-paristech.fr
PyPursuit command line arguments looks as follow:
             -h or --help : print this help

             -f [filePath] : specifies the path to the audiofile, only wav files supported for now

             -w : flag for writing the output (approx and recomposedSignal) to output files

             -o [outPutFilePath] : path for the output

             -m [maxAtomNumber] :  specifies the maximum number of selected atoms

             -s [SRR] : specifies the target Signal-To-Residual Ratio

             -d [dictionarySizes] : specifies the sizes: e.g -d 128,1024,8192

             -t or --type ['mp' | 'LOMP'] : specify the chosen decomposition algorithm

             --debug=[0|1|2|3] : debug level. Default is 0

             -a or --pad pad signal with zeroes , default is 1

             -p or --plot plots the approximation using matplotib

             -l [segmentDuration] : length of the segments

             example : python MPcmd.py -f sndFile.wav -m 100 -s 10 -d 128,1024,8192 --debug=1 -a -p

             """


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hk:apt:bf:wo:m:s:d:l:", [
            "help", "debug=", "pad", "plot", "type=", "pipe"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    # initialize default values
    global _debug
    _debug = 0
    filePath = ''
    writeOutput = False
    plotOutput = False
    outputPath = 'output.xml'
    maxAtomNumber = 100
    targetSRR = 10
    padSignal = True
    decType = 'mp'
    mdctDico = [2 ** j for j in range(7, 15)]
    matpipe = False
    segmentLength = 10
    # argument parsing

    try:
        for opt, arg in opts:
    #        print opt, arg
            if opt in ("-h", "--help"):
                usage()
                sys.exit()

            elif opt in ("-b", "--pipe"):
                matpipe = True

            # set debug level
            elif opt == "--debug":
                _debug = int(arg)

            elif opt == '-f':
                filePath = arg

            elif opt == '-w':
                writeOutput = True

            elif opt in ("-p", "--plot"):
                plotOutput = True

            elif opt == '-o':
                outputPath = arg

            elif opt == '-m':
                maxAtomNumber = int(arg)

            elif opt == '-s':
                targetSRR = int(arg)

            elif opt == '-l':
                segmentLength = float(arg)

            # tricky dictionary sizes reading
            elif opt == '-d':
                mdctDico = []
#                print arg
                for size in arg.split(','):
#                    print size
                    mdctDico.append(int(size))

            elif opt in ('-a', '--pad'):
                padSignal = True

            elif opt in ('-t , --type'):
                decType = arg
    except:
        print "sorry wrong syntax "
        usage()
        sys.exit()

    # end of argument parsing : print some info if debug level is above 0
    if _debug > 0:
        print "Starting ", decType, " on ", filePath, " targetSRR: ", targetSRR, " maxAtomNumber: ", maxAtomNumber, " on ", len(mdctDico), "xMDCT dico"
        print "debug=", _debug, "- Write ", prtBool(
            writeOutput), " - Plot : ", prtBool(plotOutput)

    ####### Load audio signal - TODO more options ##########"""
# originalSignal = py_pursuit_Signal.InitFromFile(filePath, forceMono=True,
# doNormalize=True)
    longSignal = pymp_Signal.pymp_LongSignal(
        filePath, frameDuration=segmentLength)

    # now crop the signal to an adequate length
# originalSignal.crop(0 , math.floor(originalSignal.length/max(mdctDico)) *
# max(mdctDico))
#
#    if _debug > 0:
#        print "Cropping signal to ", originalSignal.length

    ####### Build dictionary - TODO more options and more dictionaries ########
    if decType == 'mp':
        dictionary = Dico.Dico(mdctDico, useC=True)
    elif decType == 'LOMP':
        dictionary = Dico.LODico(mdctDico, useC=True)
    elif decType == 'RSSMP':
        dictionary = pymp_RandomDicos.RandomDico(mdctDico, useC=True)
    else:
        print "unrecognized dictionary type for now, availables are mp and LOMP"
        sys.exit()

    # main Decomposition algorithm
    if _debug > 0:
        print targetSRR, maxAtomNumber

    # additional parameter: segment length analysis and overlapping
    #segmentOverlap = 0.25 # in per one
# segPad = math.floor(segmentLength * originalSignal.samplingFrequency
# /max(mdctDico)) * max(mdctDico); # increment in samples
#
#    # calcul of the number of segment needed
#    segmentNumber = int(math.floor(originalSignal.length / segPad))
    segmentNumber = longSignal.segmentNumber

    if _debug > 0:
        print segmentNumber, " segments"

    # the maxAtomNumber is fixed in all segments

    # initialize approx
# approx = py_pursuit_Approx.py_pursuit_Approx(dictionary, [], None,
# originalSignal.length + dictionary.getN(), originalSignal.samplingFrequency)

    approxs = mp.mp_long(longSignal, dictionary, targetSRR, maxAtomNumber,
                               debug=_debug - 1, padSignal=padSignal, outputDir='', doWrite=False)[0]

#    for segidx in range(segmentNumber):
#        if _debug >0:
#            print "Starting work on segment " , segidx;
#
#        subSignal = longSignal.getSubSignal(segidx, 1,)
#
##        if segidx == 8:
##            _debug =2;
##        else:
##            _debug=1;
#        subApprox = mp.mp(subSignal,dictionary,targetSRR, maxAtomNumber,
# debug=_debug-1,padSignal=padSignal, silentFail=True)[0]
#        # add all atoms to overall approx
#        for atom in subApprox.atoms:
#            atom.timePosition += segPad*segidx
#
#            # BUGFIX on the approx length due to padding
##            approx.addAtom(atom)
# approx.atoms.append(atom) # no need to add it to the recomposed signal if no
# synthesis

    # Fusion the sub approximants
    approx = pymp_Approx.fusion_approxs(approxs, unPad=padSignal)

    if writeOutput:
        if _debug > 0:
            print "Writing to output file :", outputPath
        approx.originalSignal = longSignal
        approx.dumpToDisk(outputPath)
#        approx.writeToXml(outputPath)

    if plotOutput:
        plt.figure()
        approx.plot_tf()
        plt.show()

    # end of program
    if _debug > 0:
        print "Exiting"

    # addon for matlab interface: outputs all the coeff
    if matpipe:
        for atom in approx.atoms:
            print atom.length, atom.frequencyBin, atom.timePosition
    #sys.exit()


def prtBool(bool):
    if bool:
        return 'yes'
    else:
        return 'no'

if __name__ == "__main__":
    main(sys.argv[1:])
