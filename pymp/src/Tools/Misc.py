from math import floor

def dicho(x , sequence):    
    """ creates a dichotomic sequence """        
    if len(sequence) <1:
        return x
    
    if len(sequence) > 2:
        x.append(sequence[len(sequence)/2])
        dicho(x , sequence[1:len(sequence)/2])
        dicho(x , sequence[len(sequence)/2+1:-1])
            
    x.append(sequence[-1])
    x.append(sequence[0])
    return x

def jump(sequence):
    L = len(sequence)
    x = []
    for l in range(L):
        x.append(sequence[((l * ((L/2) +1)) % L)])
    return x

def sine(sequence):
    from numpy import sin,pi    
    L = len(sequence)
    x = [abs(int((L/2)*sin(x)) )  for x in sequence]
    return x

def binary(sequence):
    L = len(sequence)
    x = (sequence[0] , sequence[L/2]) * int(L/2)
    return x

def binom(sequence):
    "uses a bernoulli variable to generate a white noise by inverse fft"
    L = len(sequence)
    from numpy.random import binomial
    
    p= 0.5
    binom = binomial(1, p , size=L)
    
    from numpy.fft import ifft
    from numpy import real
    from math import sqrt , pi
    from cmath import exp
    randPhases = [(L/8)*exp(-1j*pi*i) for i in binom];
    x = sqrt(L)*real(ifft(randPhases,L));
    
    for i in range(len(x)):
        x[i] = floor(x[i]);

    return x


###############" testing ##########"
#print jump(range(16))
#print sine(range(64))
#print binary(range(16))
