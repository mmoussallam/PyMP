from numpy.fft import ifft , fft
from numpy import concatenate , nonzero
import matplotlib.pyplot as plt
import math


def GetMaxXCorr(X , Y , maxlag=0 , debug=0):
    
    if len(X) != len(Y):
        raise ValueError("FFts should be the same size")
        
    # Compute classic cross-correlation
    classic_xcorr = (ifft(X*(Y.conj()))).real
    if maxlag ==0:
        maxlag = len(X)/2
    
    classic_xcorr = concatenate((classic_xcorr[-maxlag:],classic_xcorr[0:maxlag]));

    if debug > 1:
        plt.figure()
        plt.plot(classic_xcorr)
        plt.show()

    ind = abs(classic_xcorr).argmax();
#    ind = classic_xcorr.argmax()
    val = classic_xcorr[ind]
#    return (len(X)/2 - ind) , val;
    return (maxlag - ind) , val;


def GetXCorr(x,y):
    if len(x) != len(y):
        raise ValueError("signals should be the same size")
    
    X = fft(x)
    Y = fft(y)
    
    # Compute classic cross-correlation
    classic_xcorr = (ifft(X*(Y.conj()))).real
    maxlag = len(X)/2
    classic_xcorr = concatenate((classic_xcorr[-maxlag:],classic_xcorr[0:maxlag]));
    ind = abs(classic_xcorr).argmax();
#    ind = classic_xcorr.argmax()
    val = classic_xcorr[ind]
    return classic_xcorr , (len(X)/2 - ind) , val;


def XcorrNormed(x,y):
    """ method useful to compare binary vectors """
    if len(x) != len(y):
        raise ValueError("signals should be the same size")
    
    X = fft(x)
    Y = fft(y)
    
    # Compute classic cross-correlation
    classic_xcorr = (ifft(X*(Y.conj()))).real
    maxlag = len(X)/2
    classic_xcorr = concatenate((classic_xcorr[-maxlag:],classic_xcorr[0:maxlag]));
    ind = abs(classic_xcorr).argmax();
#    ind = classic_xcorr.argmax()
    val = classic_xcorr[ind]
    
    # normalize
    normx = math.sqrt(sum((x)**2))
    normy = math.sqrt(sum((y)**2))
#    print 'Norm of ' , normx*normy , ' for a value found of ' , val
#    val = float(val)/(float(len(nonzero(x)[0]) + len(nonzero(y)[0]))/2)
    if (normx * normy) != 0:
        val = float(val)/(normx * normy)
    
    return classic_xcorr , (len(X)/2 - ind) , val;