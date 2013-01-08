#*****************************************************************************/
#                                                                            */
#                               Command line for Pymp.py                     */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Mon Aug 16 2010  */
# -------------------------------------------------------------------------- */
# */

'''
Module mp_cmd
=============

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

from PyMP import signals, approx
from PyMP.mdct import Dico, LODico
from PyMP.mdct.rand import SequenceDico
from PyMP import  mp



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

             example : python mp_cmd.py -f sndFile.wav -m 100 -s 10 -d 128,1024,8192 --debug=1 -a -p

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
    file_path = ''
    write = False
    plot = False
    out_path = 'output.xml'
    max_it_num = 100
    target_srr = 10
    pad = True
    dec_algo = 'mp'
    scales = [2 ** j for j in range(7, 15)]
    matpipe = False
    seg_length = 10
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
                file_path = arg

            elif opt == '-w':
                write = True

            elif opt in ("-p", "--plot"):
                plot = True

            elif opt == '-o':
                out_path = arg

            elif opt == '-m':
                max_it_num = int(arg)

            elif opt == '-s':
                target_srr = int(arg)

            elif opt == '-l':
                seg_length = float(arg)

            # tricky dictionary sizes reading
            elif opt == '-d':
                scales = []
#                print arg
                for size in arg.split(','):
#                    print size
                    scales.append(int(size))

            elif opt in ('-a', '--pad'):
                pad = True

            elif opt in ('-t , --type'):
                dec_algo = arg
    except:
        print "sorry wrong syntax "
        usage()
        sys.exit()

    # end of argument parsing : print some info if debug level is above 0
    if _debug > 0:
        print "Starting ", dec_algo, " on ", file_path, " target_srr: ", target_srr, " max_it_num: ", max_it_num, " on ", len(scales), "xMDCT dico"
        print "debug=", _debug, "- Write ", prtBool(
            write), " - Plot : ", prtBool(plot)

    ####### Load audio signal - TODO more options ##########"""
# originalSignal = py_pursuit_Signal.InitFromFile(file_path, forceMono=True,
# doNormalize=True)
    longSignal = signals.LongSignal(
        file_path, frameDuration=seg_length)

    # now crop the signal to an adequate length
# originalSignal.crop(0 , math.floor(originalSignal.length/max(scales)) *
# max(scales))
#
#    if _debug > 0:
#        print "Cropping signal to ", originalSignal.length

    ####### Build dictionary - TODO more options and more dictionaries ########
    if dec_algo == 'mp':
        dictionary = Dico(scales)
    elif dec_algo == 'LOMP':
        dictionary = LODico(scales)
    elif dec_algo == 'RSSMP':
        dictionary = SequenceDico(scales)
    else:
        print "unrecognized dictionary type for now, availables are mp and LOMP"
        sys.exit()

    # main Decomposition algorithm
    if _debug > 0:
        print target_srr, max_it_num

    # additional parameter: segment length analysis and overlapping
    #segmentOverlap = 0.25 # in per one
# segPad = math.floor(seg_length * originalSignal.samplingFrequency
# /max(scales)) * max(scales); # increment in samples
#
#    # calcul of the number of segment needed
#    segmentNumber = int(math.floor(originalSignal.length / segPad))
    segmentNumber = longSignal.segmentNumber

    if _debug > 0:
        print segmentNumber, " segments"

    # the max_it_num is fixed in all segments

    # initialize approx
# approx = py_pursuit_Approx.py_pursuit_Approx(dictionary, [], None,
# originalSignal.length + dictionary.getN(), originalSignal.samplingFrequency)

    approxs = mp.mp_long(longSignal, dictionary, target_srr, max_it_num,
                               debug=_debug - 1, pad=pad, output_dir='', write=False)[0]

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
#        subApprox = mp.mp(subSignal,dictionary,target_srr, max_it_num,
# debug=_debug-1,pad=pad, silentFail=True)[0]
#        # add all atoms to overall approx
#        for atom in subApprox.atoms:
#            atom.timePosition += segPad*segidx
#
#            # BUGFIX on the approx length due to padding
##            approx.addAtom(atom)
# approx.atoms.append(atom) # no need to add it to the recomposed signal if no
# synthesis

    # Fusion the sub approximants
    approx = approx.fusion_approxs(approxs, unPad=pad)

    if write:
        if _debug > 0:
            print "Writing to output file :", out_path
        approx.original_signal = longSignal
        approx.dumpToDisk(out_path)
#        approx.writeToXml(out_path)

    if plot:
        plt.figure()
        approx.plot_tf()
        plt.show()

    # end of program
    if _debug > 0:
        print "Exiting"

    # addon for matlab interface: outputs all the coeff
    if matpipe:
        for atom in approx.atoms:
            print atom.length, atom.freq_bin, atom.time_position
    #sys.exit()


def prtBool(bool):
    if bool:
        return 'yes'
    else:
        return 'no'

if __name__ == "__main__":
    main(sys.argv[1:])
