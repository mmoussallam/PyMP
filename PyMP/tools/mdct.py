""" this module handles Modified Discrete Cosine Transform and Inverse transform

M. Moussallam as part of the PyMP package
"""
# from numpy import *


from numpy.fft import fft, ifft
from numpy import mod, zeros, array, sin, concatenate, real
from math import pi, sqrt
import cmath


def mdct(x, L):

    # Signal length
    N = len(x)

    # Number of frequency channels
    K = L / 2

    # Test length
    if mod(N, K) != 0:
        print "Input length must be a multiple of half window size."
        raise ValueError(N, " is not a multiple of ", K)

    # Pad edges with zeros
    x = concatenate((concatenate((zeros(L / 4), x)), zeros(L / 4).T))

    # Number of frames
    P = N / K
    if P < 2:
        print "Signal too short"
        raise ValueError

    # Framing
    temp = x.copy()
    output = zeros(N)
    x = zeros((L, P), complex)
    y = zeros((L, P), complex)

    # Windowing
    wLong = array([sin(float(l + 0.5) * (pi / L)) for l in range(L)])

    wEdgeL = wLong.copy()
    wEdgeL[0:L / 4 - 1] = 0
    wEdgeL[L / 4:L / 2 - 1] = 1
    wEdgeR = wLong.copy()
    wEdgeR[L / 2:L / 2 + L / 4 - 1] = 1
    wEdgeR[L / 2 + L / 4:L - 1] = 0

    # twidlle coefficients
    pre_twidVec = array([cmath.exp(n * (-1j) * pi / L) for n in range(L)])
    post_twidVec = array([cmath.exp(
        (float(n) + 0.5) * -1j * pi * (L / 2 + 1) / L) for n in range(L / 2)])

    # we're note in Matlab anymore, loops are more straight than indexing
    for i in range(P):
        x[:, i] = temp[i * K: i * K + L]
        if(i == 0):
            x[:, i] = x[:, i] * wEdgeL
        elif(i == P - 1):
            x[:, i] = x[:, i] * wEdgeR
        else:
            x[:, i] = x[:, i] * wLong

        # do the pre-twiddle
        x[:, i] = x[:, i] * pre_twidVec

        # compute fft
        y[:, i] = fft(x[:, i], L)

        # post-twiddle
        y[0:L / 2, i] = y[0:L / 2, i] * post_twidVec

#        output[i*K : (i+1)*K] = sqrt(2/float(K))*real(y[0:L/2 , i]);
        output[i * K: (i + 1) * K] = sqrt(2 / float(K)) * real(y[0:L / 2, i])

    return output

# inverse transform still using fft calculation


def imdct(y, L):
    # Signal length
    N = len(y)

    # Number of frequency channels
    K = L / 2

    # Test length
    if mod(N, K) != 0:
        print ["Input length ", N,
               " must be a multiple of half window size : ", K]
        raise ValueError(
            "The length was not a multipe of half the window size")

    # Pad edges with zeros
    # x = concatenate( (concatenate( (zeros(L/4) , x) )  , zeros(L/4).T) );

    # Number of frames
    P = N / K
    if P < 2:
        print "Signal too short"
        raise ValueError

    # Framing
    temp = y.copy()
    output = zeros(P * K + K)

    x = zeros((L, P), complex)
    y = zeros((L, P), complex)

    # Windowing
    wLong = array([sin(float(l + 0.5) * (pi / L)) for l in range(L)])
    wEdgeL = wLong.copy()
    wEdgeL[0:L / 4 - 1] = 0
    wEdgeL[L / 4:L / 2 - 1] = 1
    wEdgeR = wLong.copy()
    wEdgeR[L / 2:L / 2 + L / 4 - 1] = 1
    wEdgeR[L / 2 + L / 4:L - 1] = 0

    # twidlle coefficients
    pre_twidVec = array([cmath.exp(1j * 2 * float(
        n) * pi * (float(float(L / 4 + 0.5)) / float(L))) for n in range(L)])
    post_twidVec = array(
        [cmath.exp(0.5 * 1j * 2 * pi * (float(n) + (L / 4 + 0.5)) / L) for n in range(L)])

    # we're note in Matlab anymore, loops are more straight than indexing
    for i in range(P):
        y[0:K, i] = temp[i * K: (i + 1) * K]
        # do the pre-twiddle
        y[:, i] = y[:, i] * pre_twidVec

        # compute ifft
        x[:, i] = ifft(y[:, i])

        # post-twiddle
        x[:, i] = x[:, i] * post_twidVec

        # real part and scaling
#        x[:, i] = sqrt(2/float(K))*L*real(x[:, i]);
        x[:, i] = 2 * sqrt(1 / float(L)) * L * real(x[:, i])
        if(i == 0):
            x[:, i] = x[:, i] * wEdgeL
        elif(i == P - 1):
            x[:, i] = x[:, i] * wEdgeR
        else:
            x[:, i] = x[:, i] * wLong

        # overlapp - add
        output[i * K: i * K + L] = output[i * K: i * K + L] + x[:, i]

    # scrap zeroes on the borders
    return output[K / 2:-K / 2]
