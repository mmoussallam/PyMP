#
# M. Moussallam                                             Mon Nov 12 2012  */
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
#

"""
 This class tests whether the extension C module parallelProjections is
 correctly installed and accessible.

 This module is intended to greatly accelerate calculations projections step of mp
 using MDCT and Gabor Dictionaries by using optimized C routines and fftw library

 Before proceeding, make sure FFTW library is available on your system and that
 proper authorization are granted



"""
import os, sys, platform
import commands
import string

print "---- Test import du module!"

from PyMP import parallelProjections

print "---- OK"


from PyMP import mp, signals
from PyMP.mdct import dico
from PyMP.mdct import atom

import cProfile


import numpy as np
from scipy.stats import gmean
from cmath import exp , pi
import matplotlib.pyplot as plt




print "-----Test mp sur multi-echelle MDCT"
mdctDico = [32,64,128 , 256 , 512 , 1024 , 2048 , 4096, 8192 , 16384] ;
tol = [2 for i in mdctDico];

print "test the initialization function"
if parallelProjections.initialize_plans(np.array(mdctDico),np.array(tol)) != 1:

    print "Initiliazing Stage Failed"
if parallelProjections.clean_plans() != 1:
    print "Initiliazing Stage Failed"


pySigOriginal = signals.Signal("../../data/ClocheB.wav" , doNormalize=True , forceMono=True);
pyDico2 = dico.Dico(mdctDico)

pyDico_Lomp = dico.LODico(mdctDico)
residualSignal = pySigOriginal.copy();

print " profiling test with C integration"
cProfile.runctx('mp.mp(pySigOriginal, pyDico2, 20, 200 ,0)' , globals() , locals())

cProfile.runctx('mp.mp(pySigOriginal, pyDico_Lomp, 20, 200 ,0)' , globals() , locals())




################" C binding tests ########

N = 64
L = 16;
if parallelProjections.initialize_plans(np.array([L]), np.array([2])) != 1:
    print "Initiliazing Stage Failed"

P = N/(L/2);
input_data = 0.42*np.random.random((N,1));
projectionMatrix_real = np.zeros((N,1));
projectionMatrix_comp = np.zeros((N,1),complex);
scoreTree = np.zeros((P,1))
pre_twidVec = np.array([exp(n*(-1j)*pi/L) for n in range(L)]).reshape(L,1);
post_twidVec = np.array([exp((float(n) + 0.5) * -1j*pi*(L/2 +1)/L) for n in range(L/2)]).reshape(L/2,1) ;

print scoreTree.shape , pre_twidVec.shape , post_twidVec.shape

i= 1;
j=10;
takeReal = 1;

#print "Testing Bad call:"
#computeMCLT.project(input_data )

print " ---Testing good call"
parallelProjections.project(input_data, scoreTree ,
                    projectionMatrix_real ,
                    pre_twidVec , post_twidVec , i , j , L ,0)

#if parallelFFT.clean_plans() != 1:
#    print "Cleaning Stage Failed"
###
##print  projectionMatrix_real
print  scoreTree
print "--- OK"
#
#
#if computeMCLT.initialize_plans(np.array([L])) != 1:
#    print "Initiliazing Stage Failed"

print "---Testing good call: complex"
res = parallelProjections.project_mclt(input_data, scoreTree ,
                         projectionMatrix_comp ,
                         pre_twidVec , post_twidVec , i , j , L)
print scoreTree
if res is not None:
    print "--- Ok"
else:
    print "ERROR"
#
print "---Testing good call: complex set"

res =parallelProjections.project_mclt_set(input_data, scoreTree ,
                             projectionMatrix_comp ,
                             pre_twidVec , post_twidVec , i , j , L , 1)
if res is not None:
    print "--- Ok"
else:
    print "ERRRORRRR"
    raise TypeError("ARf");


print "---Testing good call: complex Non Linear set with Median"
# Feed it with numpy matrices
sigNumber = 3;
NLinput_data =  np.concatenate((input_data,0.42*np.random.randn(N,sigNumber-1)),axis=1)

NLinput_data = NLinput_data.T;

print NLinput_data.shape
NLprojectionMatrix_comp = np.zeros(NLinput_data.shape)
projResult = np.zeros((N,1))
res =parallelProjections.project_mclt_NLset(NLinput_data, scoreTree ,
                             NLprojectionMatrix_comp ,
                             projResult,
                             pre_twidVec , post_twidVec , i , j , L , 0)

A = np.median((NLprojectionMatrix_comp)**2, axis=0)
#plt.figure()
#plt.plot(A)
#plt.plot(projResult,'r:')
#plt.draw()
#plt.draw()
assert np.sum((A.reshape(projResult.shape)-projResult)**2) ==0;
if res is not None:
    print "--- Ok",scoreTree
else:
    print "ERRRORRRR"
    raise TypeError("ARf");

print "---Testing good call: complex Non Linear set with Penalized"
projResult = np.zeros((N,1))
res =parallelProjections.project_mclt_NLset(NLinput_data, scoreTree ,
                             NLprojectionMatrix_comp ,
                             projResult,
                             pre_twidVec , post_twidVec , i , j , L , 1)

A = (NLprojectionMatrix_comp)**2;
B = np.sum(A,axis=0)
for l in range(sigNumber):
    for m in range(l+1,sigNumber):
#                    print i,j
        diff = (abs(NLprojectionMatrix_comp[l,:]) - abs(NLprojectionMatrix_comp[m,:]))**2
#                    print diff
        B[:] += diff

#plt.figure()
#plt.plot(B)
#plt.plot(projResult,'r:')
#plt.draw()
#plt.draw()
assert np.sum((B.reshape(projResult.shape)-projResult)**2) < 0.000000000001;
if res is not None:
    print "--- Ok",scoreTree
else:
    print "ERRRORRRR"
    raise TypeError("ARf");
print "---Testing good call: complex Non Linear set with Weighted"
projResult = np.zeros((N,1))
scoreTree = np.zeros((P,1))
res =parallelProjections.project_mclt_NLset(NLinput_data, scoreTree ,
                             NLprojectionMatrix_comp ,
                             projResult,
                             pre_twidVec , post_twidVec , i , j , L , 2)

A = abs(NLprojectionMatrix_comp)
flatness = (np.exp((1.0/sigNumber)*np.sum(np.log(A),axis=0))/np.mean(A,axis=0))

#print flatness


B = np.multiply(np.nan_to_num(flatness),np.sum(A**2 , axis=0))
#plt.figure()
#plt.plot(B)
#plt.plot(np.sum(A**2,axis=0),'g')
#plt.plot(projResult,'r:')
#plt.show()
assert np.sum((B.reshape(projResult.shape)-projResult)**2) < 0.000000000001;


if res is not None:
    print "--- Ok",scoreTree
else:
    print "ERRRORRRR"
    raise TypeError("ARf");

print "---Testing good call: subprojection"
# TODO pass this in the library
res = parallelProjections.subproject(input_data, scoreTree , projectionMatrix_real , pre_twidVec , post_twidVec , i , j , L ,0, 4)
if res is not None:
    print "--- Ok"
else:
    print "ERRRORRRR"
    raise TypeError("ARf");

#print "---Testing atom projection"
#scoreVec = np.array([0.0]);
#
#input2 = np.concatenate( (np.concatenate((np.zeros(L/2) , Atom.waveform) ) , zeros(self.scale/2) ) );
#
#print parallelFFT.project_atom(input_data , input_data , scoreVec )
#
#print "--- Ok"


print "Cleaning"
if parallelProjections.clean_plans() != 1:
    print "Cleaning Stage Failed"


print "---testing atom projection and creation"
scale = 128;
k = 14
if parallelProjections.initialize_plans(np.array([scale]), np.array([2]))!= 1:
    print "Initialization Stage Failed"

Atom_test = atom.Atom(scale, 1, 1200, k, 8000);
#Atom_test.mdct_value = 0.57;
Atom_test.synthesize(value=1);
#
ref = Atom_test.waveform.copy();
ts = 45;
#
input2 = np.concatenate( (np.concatenate((np.zeros(scale/2) , Atom_test.waveform ) )  ,np.zeros(scale/2) ) );
input1 = 0.01*np.random.randn(2*scale) + np.concatenate( (np.concatenate((np.zeros(scale/2-ts) , Atom_test.waveform ) )  ,np.zeros(scale/2+ts) ) );

input3 = np.array(input2);
input4 = np.array(input1);
score = np.array([0.0]);

import time
nbIt = 10000;
t = time.clock();
for j in range(nbIt):
    timeShift = parallelProjections.project_atom(input1,input2 ,score , scale)
print "C code Took",  time.clock()-t

t = time.clock();
for j in range(nbIt):
    Xcor = np.correlate(input4,  input3 , "full");
#    maxI = abs(Xcor).argmax();
#    max = abs(Xcor).max();
print "Numpy took ",  time.clock()-t


#print "See if numpy correlate is efficient"
#print score , abs(Xcor).max();
#print timeShift , abs(Xcor).argmax() - 255;

#scoreOld = np.array([0.0]);
#timeShift2 = computeMCLT.project_atom(input4,input3 ,scoreOld)

score3 = np.array([0.0]);
timeShift3 = parallelProjections.project_atom(input1,input2 ,score3 , scale)

#scoreOld2 = np.array([0.0]);
#timeShift4 = computeMCLT.project_atom(input4,input3 ,scoreOld2)

#if not(scoreOld == score):
#    print "ERROR: new score calculus isn't consistent with old one"
#    print scoreOld , score
#    print timeShift , timeShift2
#    raise TypeError("ARf");
#print score3 , scoreOld2
#
if ts==-timeShift:
    print "---- cross-correlation works!"
else:
    print "--- ERROR : cross correlation did not pass!"
    print timeShift
    raise TypeError("ARf");
print timeShift , score

print sum(Atom_test.waveform * input1[scale/2-ts:scale/2-ts + Atom_test.length])
#plt.figure()
#plt.plot(np.concatenate( (np.concatenate((np.zeros(scale/2) , ref ) )  ,np.zeros(scale/2) ) ))
#plt.plot(input1)
#plt.plot(input2);
#plt.legend(('origAtom','signal','newAtom'))
#plt.show()
#
#k = 0
#wf = parallelProjections.get_atom(scale ,  k);
#wf_gab = parallelProjections.get_real_gabor_atom(scale ,  k , 0.45);
#
#gabAtom = pymp_GaborAtom.py_pursuit_GaborAtom(scale, 1, 1, k, 1, 0.45)
#wf_gab_test = gabAtom.synthesize()

#plt.figure()
#plt.plot(wf_gab)
#plt.plot(wf_gab_test , 'r:')
#print sum((wf_gab_test - wf_gab)**2)
#
#print (sum(wf_gab**2)) , (sum(wf**2)) , (sum(wf_gab_test**2))
#
#if sum((wf_gab_test - wf_gab)**2) < 0.0000000001:
#    print "--- atom construction test OK"


print "Cleaning"
if parallelProjections.clean_plans() != 1:
    print "Cleaning Stage Failed"
    raise TypeError("ARf");
print "------------ ALL TESTS PASSED SUCCESSFULLY -----------------"
