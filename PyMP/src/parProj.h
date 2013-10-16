/*
 * parProj.h
 * #  This program is free software; you can redistribute it and/or             /
#  modify it under the terms of the GNU General Public License               /
#  as published by the Free Software Foundation; either version 2            /
#  of the License, or (at your option) any later version.                    /
#                                                                            /
#  This program is distributed in the hope that it will be useful,           /
#  but WITHOUT ANY WARRANTY; without even the implied warranty of            /
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             /
#  GNU General Public License for more details.                              /
#                                                                            /
#  You should have received a copy of the GNU General Public License         /
#  along with this program; if not, write to the Free Software               /
#  Foundation, Inc., 59 Temple Place - Suite 330,                            /
#  Boston, MA  02111-1307, USA.
 *  Created on: Oct 20, 2011
 *      Author: moussall
 */

#include <stdio.h>
#include <stdlib.h>
/*#include <complex.h>
 * REMOVED THIS FOR PORTABILITY ON WINDOWS SYSTEMS: NO C99 SUPPORT THERE*/
#include <fftw3.h>
#include <omp.h>
#include <unistd.h>
#include <math.h>
#define PI 3.14159265359

#ifndef PARPROJ_H_
#define PARPROJ_H_

/* Basic feature 64  bits platform */
#define VERSION 0.1
#ifndef X
#define X 2
#endif

/*#define DEBUG 1
 #define X 2
// modify if 32 bit detected
 #ifdef _M_X86
 #define X 1
 #endif */

#endif /* PARPROJ_H_ */


/* Hello world function to check version */
 void sayHelloBastard(void);

/* Initializing FFTW variables and ThreadPools */
 int initialize(int * sizes , int size_number , int *  tolerances, int max_thread_num);

/* Cleaning FFTW variables and ThreadPools */
 int clean(void);

 int getSizeNumber(void);

 /*
  * Parallelized Projection routine doing MDCT
  * Computation on a frame by frame basis *
  * Output is  real valued
  */
 int projectMDCT(double * cin_data ,
 		double * cin_vecProj , double * cout_scoreTree,
 		fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
 		int start , int end , int L, int Ts);

 /*
  * Parallelized Projection routine doing MCLT
  * Computation on a frame by frame basis *
  * Output is  complex valued -  CHANGES not anymore : accelerating trick!
  */
 int projectMCLT(double * cin_data ,
		 double * cin_vecProj ,
		 double * cout_scoreTree,
 		fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
 		int start , int end , int L);

/* Specialized Routines for Masked and Penalized selection criterions*/
 int projectMaskedGabor(double * cin_data,
                  double * cout_scoreTree,
                  fftw_complex * cin_vecProj,
                  double *penalty_mask,
                  int   start,
                  int   end,
                  int L);

 int projectPenalizedMDCT(double * cin_data,
                  double * cout_scoreTree,
                  double * cin_vecProj,
                  double * cin_vecPenalizedProj,
                  double *penalty_mask,
                  fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
                  int   start,
                  int   end,
                  int L, double lambda);

 /* Multidimensionnal projection */
 int projectSet(double * cin_data ,
		 double * cin_vecProj , double * cout_scoreTree,
 		fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
 		int start , int end , int L, int type);

 /* Plain calculus of an atom projection */
 int projectAtom(double *cin_sigData,
 		double *cin_atomData ,
 		double *cout_score,
 		int L, int scale);

 /*
  * Project an atom and find the optimal time shift and projection value
  * assuming the atom's waveform fft is already known
  */
 int reprojectAtom(double *cin_sigData,
 		double *cin_atomData,
 		fftw_complex *cin_atomfft ,
 		double *cout_score,
 		int L, int scale,int sigIdx, int maj);

 int subprojectMDCT(double * cin_data ,
 		double * cin_vecProj , double * cout_scoreTree,
 		fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
 		int start , int end , int L, int Ts,  int subFactor);


 int findMaxXcorr(fftw_plan fftw_ip, double *out_atomScore, int maxIndex,
 		int halfOffsetWidth, fftw_complex *fftw_output, int scale, int L);

 double modulus( fftw_complex c );

 void product( fftw_complex c1 , fftw_complex c2 , fftw_complex prod);
