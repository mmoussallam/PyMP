/*****************************************************************************
#                                                                            /
#                              parallelFFT.c                                 /
#                                                                            /
#                        Matching Pursuit Library                            /
#                                                                            /
# M. Moussallam                                             Tue Mar 22 2011  /
# -------------------------------------------------------------------------- /
#                                                                            /
#                                                                            /
#  This program is free software; you can redistribute it and/or             /
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
#  Boston, MA  02111-1307, USA.                                              /
#                                                                            /
#******************************************************************************/

/* This file contains method to be callable from Python through the parallelFFT
 * module, it is a binding to parallelized routines from the fastPursuit library */

#include "parallelProjections.h"
/* k-th smallest calculation routine taken from the internet */
//===================== Method 1: =============================================
//Algorithm from N. Wirth's book Algorithms + data structures = programs of 1976

typedef double elem_type ;

#ifndef ELEM_SWAP(a,b)
#define ELEM_SWAP(a,b) { register elem_type t=(a);(a)=(b);(b)=t; }

elem_type kth_smallest(elem_type a[], uint16_t n, uint16_t k)
{
    uint64_t i,j,l,m ;
    elem_type x ;
    l=0 ; m=n-1 ;
    while (l<m) {
    x=a[k] ;
    i=l ;
    j=m ;
    do {
    while (a[i]<x) i++ ;
    while (x<a[j]) j-- ;
    if (i<=j) {
    ELEM_SWAP(a[i],a[j]) ;
    i++ ; j-- ;
    }
    } while (i<=j) ;
    if (j<k) l=i ;
    if (k<i) m=j ;
    }
    return a[k] ;
}

    #define wirth_median(a,n) kth_smallest(a,n,(((n)&1)?((n)/2):(((n)/2)-1)))

//===================== Method 2: =============================================
//This is the faster median determination method.
//Algorithm from Numerical recipes in C of 1992

elem_type quick_select_median(elem_type arr[], uint16_t n)
{
    uint16_t low, high ;
    uint16_t median;
    uint16_t middle, ll, hh;
    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
    if (high <= low) /* One element only */
    return arr[median] ;
    if (high == low + 1) { /* Two elements only */
    if (arr[low] > arr[high])
    ELEM_SWAP(arr[low], arr[high]) ;
    return arr[median] ;
    }
    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])
    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])
    ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])
    ELEM_SWAP(arr[middle], arr[low]) ;
    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;
    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
    do ll++; while (arr[low] > arr[ll]) ;
    do hh--; while (arr[hh] > arr[low]) ;
    if (hh < ll)
    break;
    ELEM_SWAP(arr[ll], arr[hh]) ;
    }
    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;
    /* Re-set active partition */
    if (hh <= median)
    low = ll;
    if (hh >= median)
    high = hh - 1;
    }
    return arr[median] ;
}
#endif
/* This function should be called before others to ensure plans are statically created
 * WARNING : ensure all corresponding plans are properly cleaned after use
 */
static PyObject * initialize_plans(PyObject *self, PyObject *args)
{

	/* declarations */
	PyArrayObject *in_data ,* in_tolerances;
	
	int dim2, size_number;
	int * sizes , * tolerances;
	/*int  threadIdx;*/
	int res ;
	/* Declarations - end */
	
	/* parsing arguments */
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &in_data,
										&PyArray_Type, &in_tolerances )){
	}

	/* Check null pointers */
       if (NULL == in_data)  return NULL;
       if (NULL == in_tolerances)  return NULL;

       /* convert to integer vectors */
    size_number = in_data->dimensions[0];
    dim2 = in_data->dimensions[1];

    if (!(size_number == in_tolerances->dimensions[0])){
    	printf("ERROR in initialization: tolerance tab size %d is not the same as fft_sizes %d?\n",(int)in_tolerances->dimensions[0], (int)size_number);
    	return NULL;
    }

    /* retrieve input vector */
    sizes = pyvector_to_intCarrayptrs(in_data);
    tolerances = pyvector_to_intCarrayptrs(in_tolerances);
    
    res = initialize(sizes , size_number , tolerances);
    if (res<0) {
    	printf("ERROR in initialization, zero-size found is system 32 bits?\n");
    	return NULL;
    }
    
    return Py_BuildValue("i", 1);
    
}

/* Call top deallocate and properly destroy all fftw_plans needed
 */
static PyObject * clean_plans(PyObject *self, PyObject *args)
{
	/* Simple Call to library function */
	if (clean() < 0){
		return NULL;
	}
    return Py_BuildValue("i", 1);     	
       
}


/* Main function : project input on a MDCT dictionary using FFTs 
 * usage in Python :  scoreTree = project(data , projVec , preTwid, postTwid , startingFrame , endingFrame , scale)
 * data , projVec ,preTwid and postTwid must be Numpy arrays , the last three are integers
 * output scoreTree is a Numpy array already instantiated: only fftw variables will be instantiated and freed
 * 
 * So output should be given in the input parameters
 * */ 
static PyObject *
project(PyObject *self, PyObject *args)
{
	/* declarations */
       PyArrayObject *in_data , *in_vecProj , *in_vecPreTwid, *in_vecPostTwid, *out_scoreTree;  // The python objects to be extracted from the args
       double *cin_data , *cin_vecProj , *cout_scoreTree;   // The C vectors to be created to point to the 
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively
       
       fftw_complex * cin_vecPreTwid, *cin_vecPostTwid; // complex twiddling coefficients       
       int start,  end , L , Ts, res; // touched frame indexes, size and potential offsets
       int n_frames;
	   //printf("%d Threads have been created \n", nb_threads);
       /* Parse tuples separately since args will differ between C fcns */
       if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiii", &PyArray_Type, &in_data,
       												&PyArray_Type, &out_scoreTree, 
       											  &PyArray_Type, &in_vecProj,
       											  &PyArray_Type, &in_vecPreTwid,
       											  &PyArray_Type, &in_vecPostTwid,
       											  &start,  &end , &L ,&Ts))  return NULL;
       											  
		/* Check null pointers */
       if (NULL == in_data)  {
       	printf("in_data was null..\n");
       	return NULL;}
       if (NULL == in_vecProj)  {
       	printf("in_vecProj was null..\n");
       	return NULL;}
       if (NULL == in_vecPreTwid)  {
       	printf("in_vecPreTwid was null..\n");
       	return NULL;}
       if (NULL == in_vecPostTwid)  {
       	printf("in_vecPostTwid was null..\n");
       	return NULL;}
       if (NULL == out_scoreTree)  {
       	printf("out_scoreTree was null..\n");
       	return NULL;}
       
	   n_frames = out_scoreTree->dimensions[0];

		// Check on maximum frame index search: force last frame index to be in the bounds
	   if (end >= n_frames) {
	   		//printf("Warning : asked frame end is out of bounds\n");
	   		end = n_frames-1;
	   }
       
       /* Change contiguous arrays into C * arrays   */
       cin_data=pyvector_to_Carrayptrs(in_data);
       cin_vecProj=pyvector_to_Carrayptrs(in_vecProj);
       cout_scoreTree=pyvector_to_Carrayptrs(out_scoreTree);
       
       /* These vectors are complex */
       cin_vecPreTwid=pyvector_to_complexCarrayptrs(in_vecPreTwid);
       cin_vecPostTwid=pyvector_to_complexCarrayptrs(in_vecPostTwid);
       
       /* Then call library function to compute the projections */
       res = projectMDCT(cin_data ,cin_vecProj ,cout_scoreTree,
    		   cin_vecPreTwid , cin_vecPostTwid,
    		   start ,end ,L, Ts);
	   
       if (res < 0) return NULL;

	  return Py_BuildValue("i", 1);
}

/* Derived function : Everything is the same except that only even frames are computed: simulate a two-times subsampled MDCT dictionary projection
 * */ 
static PyObject *
subproject(PyObject *self, PyObject *args)
{
	 /*declarations*/
       PyArrayObject *in_data , *in_vecProj , *in_vecPreTwid, *in_vecPostTwid, *out_scoreTree;  // The python objects to be extracted from the args
       double *cin_data , *cin_vecProj , *cout_scoreTree;   // The C vectors to be created to point to the 
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively
       
       fftw_complex * cin_vecPreTwid, *cin_vecPostTwid; // complex twiddling coefficients       
       int start,  end , L ,n_data, n_frames; // touched frame indexes, size and potential offsets
       int Ts, subFactor , res;

	   	       	  


        /*Parse tuples separately since args will differ between C fcns*/
       if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiiii", &PyArray_Type, &in_data,
       												&PyArray_Type, &out_scoreTree, 
       											  &PyArray_Type, &in_vecProj,
       											  &PyArray_Type, &in_vecPreTwid,
       											  &PyArray_Type, &in_vecPostTwid,
       											  &start,  &end , &L ,&Ts , &subFactor))  return NULL;
       											  
		/* Check null pointers*/
       if (NULL == in_data)  {
       	printf("in_data was null..\n");
       	return NULL;}
       if (NULL == in_vecProj)  {
       	printf("in_vecProj was null..\n");
       	return NULL;}
       if (NULL == in_vecPreTwid)  {
       	printf("in_vecPreTwid was null..\n");
       	return NULL;}
       if (NULL == in_vecPostTwid)  {
       	printf("in_vecPostTwid was null..\n");
       	return NULL;}
       if (NULL == out_scoreTree)  {
       	printf("out_scoreTree was null..\n");
       	return NULL;}
       
       
       /* Change contiguous arrays into C * arrays*/
       cin_data=pyvector_to_Carrayptrs(in_data);
       cin_vecProj=pyvector_to_Carrayptrs(in_vecProj);
       cout_scoreTree=pyvector_to_Carrayptrs(out_scoreTree);
       
       /* These vectors are complex*/
       cin_vecPreTwid=pyvector_to_complexCarrayptrs(in_vecPreTwid);
       cin_vecPostTwid=pyvector_to_complexCarrayptrs(in_vecPostTwid);
       
       /* Get input data dimension.*/
       n_data=in_data->dimensions[0];

   	    /*retrieve maximum frame number*/
   	   n_frames = out_scoreTree->dimensions[0];

   		// Check on maximum frame index search: force last frame index to be in the bounds
   	   if (end >= n_frames) {
   	   		//printf("Warning : asked frame end is out of bounds\n");
   	   		end = n_frames;
   	   }

       res = subprojectMDCT(cin_data ,
					   cin_vecProj ,
					   cout_scoreTree ,
					   cin_vecPreTwid ,
					   cin_vecPostTwid,
					   start ,end , L, Ts, subFactor);
       
       if (res < 0) return NULL;
	   
	  return Py_BuildValue("i", 1);
}

/* New transform based on Complex Lapped Transform for LOMP algorithm
 Basically the same as above except some vectors are complex and the
 * inner products are computed just a bit differently */
static PyObject *
project_mclt(PyObject *self, PyObject *args)
{
	
	 /* Declarations */
       PyArrayObject *in_data , *in_vecProj , *in_vecPreTwid, *in_vecPostTwid, *out_scoreTree;  // The python objects to be extracted from the args
       double *cin_data  , *cout_scoreTree , *cin_vecProj;   // The C vectors to be created to point to the
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively
       
       fftw_complex * cin_vecPreTwid, *cin_vecPostTwid;/* , *cin_vecProj; // complex twiddling coefficients */
       int start,  end , L;
       int res;
       /* Declarations -  end*/
       int n_frames;
        /*Parse tuples separately since args will differ between C fcns*/
       if (!PyArg_ParseTuple(args, "O!O!O!O!O!iii", &PyArray_Type, &in_data,
       												&PyArray_Type, &out_scoreTree, 
       											  &PyArray_Type, &in_vecProj,
       											  &PyArray_Type, &in_vecPreTwid,
       											  &PyArray_Type, &in_vecPostTwid,
       											  &start,  &end , &L ))  return NULL;
       											  
		/* Checking null pointers*/
       if (NULL == in_data)  return NULL;
       if (NULL == in_vecProj)  return NULL;
       if (NULL == in_vecPreTwid)  return NULL;
       if (NULL == in_vecPostTwid)  return NULL;
       if (NULL == out_scoreTree)  return NULL;
       
	   n_frames = out_scoreTree->dimensions[0];

		// Check on maximum frame index search: force last frame index to be in the bounds
	   if (end >= n_frames) {
	   		//printf("Warning : asked frame end is out of bounds\n");
	   		end = n_frames-1;
	   }
       
       /* Change contiguous arrays into C * arrays*/
       cin_data=pyvector_to_Carrayptrs(in_data);       
       /*cin_vecProj=pyvector_to_complexCarrayptrs(in_vecProj);*/
       cin_vecProj=pyvector_to_Carrayptrs(in_vecProj);

       cin_vecPreTwid=pyvector_to_complexCarrayptrs(in_vecPreTwid);
       cin_vecPostTwid=pyvector_to_complexCarrayptrs(in_vecPostTwid);       
       cout_scoreTree=pyvector_to_Carrayptrs(out_scoreTree);
       
       res = projectMCLT(cin_data ,
				   cin_vecProj ,
				   cout_scoreTree,
           		   cin_vecPreTwid , cin_vecPostTwid,
           		   start ,end ,L);
       
       if (res < 0) return NULL;

	  return Py_BuildValue("i", 1);
}

/* Simple STFT transform for Gabor dictionary projections (with possible masking)
 Basically the same as above except some vectors are complex and the
 * inner products are computed just a bit differently */
static PyObject *
project_masked_gabor(PyObject *self, PyObject *args)
{

	 /* Declarations */
       PyArrayObject *in_data , *in_vecProj , *out_scoreTree, *in_mask;  // The python objects to be extracted from the args
       double *cin_data  , *cout_scoreTree , *cin_mask;   // The C vectors to be created to point to the
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively
       fftw_complex  *cin_vecProj;
       int start,  end , L;
       int res;
       /* Declarations -  end*/
       int n_frames;
        /*Parse tuples separately since args will differ between C fcns*/
       if (!PyArg_ParseTuple(args, "O!O!O!O!iii", &PyArray_Type, &in_data,
       												&PyArray_Type, &out_scoreTree,
       											  &PyArray_Type, &in_vecProj,
       											  &PyArray_Type, &in_mask,
       											  &start,  &end , &L)){
    	   printf("Failed to parse arguments");
    	   return NULL;
       }

		/* Checking null pointers*/
       if (NULL == in_data)  return NULL;
       if (NULL == in_vecProj)  return NULL;
       if (NULL == in_mask)  return NULL;
       if (NULL == out_scoreTree)  return NULL;



	   n_frames = out_scoreTree->dimensions[0];

		// Check on maximum frame index search: force last frame index to be in the bounds
	   if (end >= n_frames) {
	   		//printf("Warning : asked frame end is out of bounds\n");
	   		end = n_frames-1;
	   }

       /* Change contiguous arrays into C * arrays*/
       cin_data=pyvector_to_Carrayptrs(in_data);
       /*cin_vecProj=pyvector_to_complexCarrayptrs(in_vecProj);*/
       cin_vecProj=pyvector_to_complexCarrayptrs(in_vecProj);

       cin_mask=pyvector_to_Carrayptrs(in_mask);
       cout_scoreTree=pyvector_to_Carrayptrs(out_scoreTree);

       if(DEBUG) printf("calling projectGabor function");

       res = projectMaskedGabor(cin_data ,
				   cout_scoreTree,
				   cin_vecProj ,
           		   cin_mask,
           		   start ,end ,L);

       if (res < 0) return NULL;

	  return Py_BuildValue("i", 1);
}


/* Here we are interested in summing , multiplying or maximizing the minimums of
 * the projections on a set of signal: only difference with project_mclt is the
 * update of the projection matrix: no pure replacement. Also need an additional
 * entry to control the update mechanism */
static PyObject *
project_mclt_set(PyObject *self, PyObject *args)
{

	 /*Declarations*/
       PyArrayObject *in_data , *in_vecProj , *in_vecPreTwid, *in_vecPostTwid, *out_scoreTree;  // The python objects to be extracted from the args
       double *cin_data  , *cout_scoreTree , *cin_vecProj;   // The C vectors to be created to point to the
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively

       fftw_complex * cin_vecPreTwid, *cin_vecPostTwid ;//, *cin_vecProj; // complex twiddling coefficients
       int start,  end , L, res;
       int n_data, n_frames;
       int type; // 0 for summation, 1 for multiplying , 2 for taking the minimum

	   //fftw_complex prod;
        /*Declarations -  end*/

       /* Parse tuples separately since args will differ between C fcns*/
       if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiii", &PyArray_Type, &in_data,
       												&PyArray_Type, &out_scoreTree,
       											  &PyArray_Type, &in_vecProj,
       											  &PyArray_Type, &in_vecPreTwid,
       											  &PyArray_Type, &in_vecPostTwid,
       											  &start,  &end , &L , &type))  return NULL;

		/* Checking null pointers*/
       if (NULL == in_data)  return NULL;
       if (NULL == in_vecProj)  return NULL;
       if (NULL == in_vecPreTwid)  return NULL;
       if (NULL == in_vecPostTwid)  return NULL;
       if (NULL == out_scoreTree)  return NULL;


       /* Change contiguous arrays into C * arrays*/
       cin_data=pyvector_to_Carrayptrs(in_data);

       /*cin_vecProj=pyvector_to_complexCarrayptrs(in_vecProj);*/
       cin_vecProj=pyvector_to_Carrayptrs(in_vecProj);

       cin_vecPreTwid=pyvector_to_complexCarrayptrs(in_vecPreTwid);
       cin_vecPostTwid=pyvector_to_complexCarrayptrs(in_vecPostTwid);
       cout_scoreTree=pyvector_to_Carrayptrs(out_scoreTree);

       /* Get input data dimension.*/
       n_data=in_data->dimensions[0];

	   /* retrieve maximum frame number*/
	   n_frames = out_scoreTree->dimensions[0];

	   /*printf("Projection matrix of %d frames and %d samples\n" ,n_frames, n_data);*/

		// Check on maximum frame index search
	   if (end >= n_frames) {
	   		//printf("Warning : asked frame end is out of bounds\n");
	   		end = n_frames-1;
	   }

	   res = projectSet(cin_data ,
			   cin_vecProj ,cout_scoreTree,
       		   cin_vecPreTwid , cin_vecPostTwid,
       		   start ,end ,L, type);

	   if (res < 0) return NULL;

	  return Py_BuildValue("i", 1);
}

/* Here we are interested in NonLinear operations such as median filtering, penalization etc..
 * the projections on a set of signal: only difference with project_mclt is the
 * update of the projection matrix: no pure replacement. Also need an additional
 * entry to control the update mechanism */
static PyObject *
project_mclt_NLset(PyObject *self, PyObject *args)
{

	 /*Declarations*/
       PyArrayObject *in_data , *in_vecProj, *out_vecProj , *in_vecPreTwid, *in_vecPostTwid, *out_scoreTree;  // The python objects to be extracted from the args
       double *cin_data  , *cout_scoreTree , *cin_vecProj, *cout_vecProj;   // The C vectors to be created to point to the
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively

       fftw_complex * cin_vecPreTwid, *cin_vecPostTwid ;//, *cin_vecProj; // complex twiddling coefficients
       int start,  end , L, res,K;
       int n_data, n_frames;
       int type; // 0 for summation, 1 for multiplying , 2 for taking the minimum
       int n_sig , sigIdx, otherSigIdx ,i,j;
       double * localArray;
       double geoMean, arithMean,ratio,max;
	   //fftw_complex prod;
        /*Declarations -  end*/

       /* Parse tuples separately since args will differ between C fcns*/
       if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiii", &PyArray_Type, &in_data,
       												&PyArray_Type, &out_scoreTree,
       											  &PyArray_Type, &in_vecProj,
       											  &PyArray_Type, &out_vecProj,
       											  &PyArray_Type, &in_vecPreTwid,
       											  &PyArray_Type, &in_vecPostTwid,
       											  &start,  &end , &L , &type))  return NULL;

		/* Checking null pointers*/
       if (NULL == in_data)  return NULL;
       if (NULL == in_vecProj)  return NULL;
       if (NULL == out_vecProj)  return NULL;
       if (NULL == in_vecPreTwid)  return NULL;
       if (NULL == in_vecPostTwid)  return NULL;
       if (NULL == out_scoreTree)  return NULL;



       /* The changes are here, we need to perform the projection for each signal before calculating the scores        */
       /* Change contiguous arrays into C * arrays*/

       /*printf("Array Dimensions : %d - %d\n", n,m);*/
       cin_data=pyvector_to_Carrayptrs(in_data);

       /*cin_vecProj=pyvector_to_complexCarrayptrs(in_vecProj);*/
       cin_vecProj=pyvector_to_Carrayptrs(in_vecProj);

       /* The difference here is that we compute the projections on each signal before calculating
        * the desired function (e.g median or penalized or weighted..)
        */
       cout_vecProj = pyvector_to_Carrayptrs(out_vecProj);

       cin_vecPreTwid=pyvector_to_complexCarrayptrs(in_vecPreTwid);
       cin_vecPostTwid=pyvector_to_complexCarrayptrs(in_vecPostTwid);
       cout_scoreTree=pyvector_to_Carrayptrs(out_scoreTree);

       /* Get input data dimension.*/
       n_data=in_data->dimensions[1];
       n_sig = in_data->dimensions[0];

       localArray = (double*)malloc(sizeof(double) * n_sig);
	   /* retrieve maximum frame number*/
	   n_frames = out_scoreTree->dimensions[0];

	   /*printf("n_data:%d ; n_sig:%d ; n_frames:%d",n_data,n_sig,n_frames);*/
	   /*printf("Projection matrix of %d frames and %d samples\n" ,n_frames, n_data);*/

		// Check on maximum frame index search
	   if (end >= n_frames) {
	   		//printf("Warning : asked frame end is out of bounds\n");
	   		end = n_frames-1;
	   }
	   /* Now for each signal we compute the projection */
	   for(sigIdx=0; sigIdx<n_sig ; sigIdx++){
		   /*printf("Computing proj for sig %d \n",sigIdx);*/
		   res = projectMCLT(&cin_data[sigIdx*n_data] ,
				   &cin_vecProj[sigIdx*n_data] ,
				   cout_scoreTree,
				   cin_vecPreTwid ,
				   cin_vecPostTwid,
				   start ,end ,L);
		   if (res < 0) return NULL;
	   }
	   K = L/2;
	   /* We can now combine all the projections to process the selection critera */
	   switch(type){
	   case 0: // The out_vecProj will contain the MEDIAN value of squared projections on all the signals

		   // TODO : compute only values that have changed
		   for(i=start*K;i<end*K;i++){
			   // fill the local array with columns from the projection Matrix
			   for(sigIdx=0;sigIdx<n_sig;sigIdx++){
				   localArray[sigIdx] = cin_vecProj[sigIdx*n_data + i] * cin_vecProj[sigIdx*n_data + i];
			   }
			   // now fill the overall projection vector with the median value in each column
			   cout_vecProj[i] = quick_select_median(localArray ,n_sig);
		   }

		   break;
	   case 1: // Penalization by the sum of the squared differences (TODO Lambda parameter is fixed)

		   // TODO : compute only values that have changed
		   for(i=start*K;i<end*K;i++){
			   cout_vecProj[i] = 0;
			   // fill the local array with columns from the projection Matrix : taking the squared value
			   for(sigIdx=0;sigIdx<n_sig;sigIdx++){
				   localArray[sigIdx] = fabs(cin_vecProj[sigIdx*n_data + i]) ;
				   cout_vecProj[i] += localArray[sigIdx] * localArray[sigIdx];
			   }
			   // The penalty added is the sum of the squared differences between channels
			   for(sigIdx=0;sigIdx<n_sig;sigIdx++){

				  for(otherSigIdx=sigIdx+1; otherSigIdx<n_sig ; otherSigIdx++){
					  cout_vecProj[i] += (localArray[otherSigIdx] - localArray[sigIdx])*(localArray[otherSigIdx] - localArray[sigIdx]);
				  }
			   }
			   // now fill the overall projection vector with the median value in each column
			   /*cout_vecProj[i] = quick_select_median(localArray ,n_sig);*/
		   }


		   break;


	   case 2: // Weighting the sum os squared projs by the ratio og geometric mean over arithmetic mean of absolute projs

	   		   /*printf("Type of log: 2 ? %2.2f, 10? %2.2f , ln? %2.2f",log(2),log(10), log(exp(1)));*/
	   		   // TODO : compute only values that have changed
	   		   for(i=start*K;i<end*K;i++){
	   			   cout_vecProj[i] = 0;
	   			   geoMean = 0;
	   			   arithMean = 0;
	   			   // fill the local array with columns from the projection Matrix
	   			   for(sigIdx=0;sigIdx<n_sig;sigIdx++){
	   				   localArray[sigIdx] = fabs(cin_vecProj[sigIdx*n_data + i]);

	   				   geoMean += log(localArray[sigIdx]);
	   				   arithMean += localArray[sigIdx];

	   				   cout_vecProj[i] += localArray[sigIdx] * localArray[sigIdx];
	   			   }
	   			   if (arithMean != 0){
	   				   ratio = exp(((double)(1.0/n_sig))*geoMean)/(((double)(1.0/n_sig))*arithMean);
	   				   /*printf(" %2.2f ",ratio);*/
	   				   cout_vecProj[i] *= ratio;

	   			   }
	   			   else{ // case where all values are zeroes
	   				   /*printf(" NaN ");*/
	   				   cout_vecProj[i] = 0;
	   			   }

	   		   }
	   		/*printf("\n");*/

	   		   break;
	   default:
		   printf("parallelFFT.c: line 618 : Unrecognized Type");
		   return NULL;
	   }
	   /*printf("MAx search %d , %d, %d\n", start,end,K);*/
	   /* before quiting let us populate the scoreTree */
	   for(i=start; i < end ; i++){
		   max = 0;
				   for(j=0; j < K; j++){
					   /*printf("vecProj frame %d k %d: %2.2f\n",i,j,cout_vecProj[j + i*K]);*/
					   if (cout_vecProj[j + i*K] > max) {
						   max = cout_vecProj[j + i*K];
						   cout_scoreTree[i] = max;
						   /*printf("Score frame %d : %2.2f \n",i,cout_scoreTree[i]);*/
					   }
				   }
	   }
	   free(localArray);
	  return Py_BuildValue("i", 1);
}


/*
 PROJECT_ATOM
 * replaces Python code: optimize atom positionning using ffts and
 * retrieve the best time-shift and the resulting projection score
 *
 */
static PyObject *
project_atom(PyObject *self, PyObject *args){

	 /*Declarations*/
       PyArrayObject *in_sigData , *in_atomData , *out_score;  // The python objects to be extracted from the args
       double *cin_sigData, *cin_atomData , *cout_score ;   // The C vectors to be created to point to the
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively
       int L, scale , out_atomTS;


       if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &in_sigData,
       											&PyArray_Type, &in_atomData ,
       											&PyArray_Type, &out_score ,&scale))  return NULL;

       if (NULL == in_sigData)  return NULL;
       if (NULL == in_atomData)  return NULL;
       if (NULL == out_score)  return NULL;

	   /* Change contiguous arrays into C * arrays*/
       cin_sigData=pyvector_to_Carrayptrs(in_sigData);
       cin_atomData=pyvector_to_Carrayptrs(in_atomData);
       cout_score = pyvector_to_Carrayptrs(out_score);

	   // retrieve FFT length
	   L = in_sigData->dimensions[0];

	   out_atomTS  = projectAtom(cin_sigData , cin_atomData , cout_score, L , scale);

	   /*printf("found %d - %d\n", out_atomTS, L);*/

        // return value of evaluated time shift
        return Py_BuildValue("i", out_atomTS);

}

/*
 PROJECT_ATOM
 * replaces Python code: optimize atom positionning using ffts and
 * retrieve the best time-shift and the resulting projection score
 *
 */
static PyObject *
project_atom_set(PyObject *self, PyObject *args){

	 /*Declarations*/
       PyArrayObject *in_sigData , *in_atomData , *in_atomfft , *out_score;  // The python objects to be extracted from the args
       double *cin_sigData, *cin_atomData , *cout_score ;   // The C vectors to be created to point to the
                                       //   python vectors, cin and cout point to the row
                                       //   of vecin and vecout, respectively
       fftw_complex *cin_atomfft;
       int L, scale , out_atomTS , sigIdx;


       if (!PyArg_ParseTuple(args, "O!O!O!O!ii", &PyArray_Type, &in_sigData,
       											&PyArray_Type, &in_atomData ,
       											&PyArray_Type, &in_atomfft ,
       											&PyArray_Type, &out_score ,&scale,&sigIdx))  return NULL;

       if (NULL == in_sigData)  return NULL;
       if (NULL == in_atomData)  return NULL;
       if (NULL == in_atomfft)  return NULL;
       if (NULL == out_score)  return NULL;

	   /* Change contiguous arrays into C * arrays*/
       cin_sigData=pyvector_to_Carrayptrs(in_sigData);
       cin_atomData=pyvector_to_Carrayptrs(in_atomData);
       cin_atomfft=pyvector_to_complexCarrayptrs(in_atomfft);
       cout_score = pyvector_to_Carrayptrs(out_score);

	   // retrieve FFT length
	   L = in_sigData->dimensions[0];

	   out_atomTS  = reprojectAtom(cin_sigData ,
								   cin_atomData,
								   cin_atomfft ,
								   cout_score, L , scale, sigIdx);

	   /*printf("found %d - %d\n", out_atomTS, L);*/

        // return value of evaluated time shift
        return Py_BuildValue("i", out_atomTS);

}


/* Useful routine for constructing Atom's waveforms */
static PyObject * get_atom(PyObject *self, PyObject *args){

	/* Declarations */
	PyArrayObject * py_out;
    int length , freqBin , i ;
    double *waveform ;
    double L, K , fact , constFact , constOffset , f;
    npy_intp dimensions[1];

    /* End of declarations */

    // check input argument
    if (!PyArg_ParseTuple(args, "ii", &length, &freqBin))  return NULL;

    dimensions[0] = (npy_intp) length;
    /* Create a new double vector of same dimension */

    /*py_out=(PyArrayObject *) PyArray_FromDims(1,dimensions,NPY_DOUBLE);*/
    py_out=(PyArrayObject *) PyArray_SimpleNew(1,dimensions,NPY_DOUBLE);

    /* Change contiguous arrays into C *arrays   */
    waveform=pyvector_to_Carrayptrs(py_out);

    /* Do the calculation of the waveform */
    L = ((double) length);
    K = L/2;
    fact = sqrt(2 / K);
    constFact = (PI/K)*(((double)freqBin) +0.5);
    constOffset =  (L + 1.0)/2.0;
    f = (PI/L);
    for ( i=0; i<length; i++)  {
            waveform[i]= (double)fact*sin(f*(i+0.5))*cos(constFact*((i - K/2) + constOffset));
    }

    /* return the waveform as a Python Numpy Array object */
    return PyArray_Return(py_out);

}


/* Useful routine for constructing Atom's waveforms */
static PyObject * get_real_gabor_atom(PyObject *self, PyObject *args){

	/* Declarations */
	PyArrayObject * py_out;
    int length , i ;
    double *waveform , phase, freqBin ;
    double L, K , fact ,  f;
    npy_intp dimensions[1];

    /* End of declarations */

    // check input argument
    if (!PyArg_ParseTuple(args, "idd", &length, &freqBin,&phase))  return NULL;

    dimensions[0] = (npy_intp) length;
    /* Create a new double vector of same dimension */

    /*py_out=(PyArrayObject *) PyArray_FromDims(1,dimensions,NPY_DOUBLE);*/
    py_out=(PyArrayObject *) PyArray_SimpleNew(1,dimensions,NPY_DOUBLE);

    /* Change contiguous arrays into C *arrays   */
    waveform=pyvector_to_Carrayptrs(py_out);

    /* Do the calculation of the waveform */
    L = ((double) length);
    K = L/2;
    /*fact = sqrt(16.0/(3.0*L)) ;*/
	fact = 0.0;

    f = 2.0*(PI/L);
    for ( i=0; i<length; i++)  {
            waveform[i]= (0.5*(1 - cos(f*(i))))*cos(f*i*freqBin + phase);
            fact += (waveform[i])*(waveform[i]);
    }
    fact = sqrt(fact);

    if (fact <= 0){
    	printf("Warning zero energy waveform");
    	return PyArray_Return(py_out);
    }

    /* Normalize the waveform */
    for ( i=0; i<length; i++)  {
            waveform[i]= waveform[i]/fact;
    }

    /* return the waveform as a Python Numpy Array object */
    return PyArray_Return(py_out);

}

  /* ==== Create 1D Carray from PyArray ======================
     Assumes PyArray is contiguous in memory.             */
 double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    return (double *) arrayin->data;  /* pointer to arrayin data as double */
}


/*     Assumes PyArray is contiguous in memory.             */
 int *pyvector_to_intCarrayptrs(PyArrayObject *arrayin)  {
    return (int *) arrayin->data;  /* pointer to arrayin data as double */
}

  /* ==== Create 1D Complex Carray from PyArray ======================
     Assumes PyArray is contiguous in memory.             */
fftw_complex *pyvector_to_complexCarrayptrs(PyArrayObject *arrayin)  {
    return (fftw_complex *) arrayin->data; /* pointer to arrayin data as fftw_complex */
}





static PyObject * test(PyObject *self, PyObject *args)
  {
    const int size = 16384;
    double sinTable[size];
    int n;
    
    #pragma omp parallel for
    for(n=0; n<size; ++n)
      sinTable[n] = sin(2 * PI * n / size);
  
  	return Py_BuildValue("i", 1);
    // the table is now initialized
  }

// Listing the methods
static PyMethodDef AllMethods[] = {
        
    {"project",  project, METH_VARARGS,  "Loop of mdct computations."},
    {"subproject",  subproject, METH_VARARGS,  "Loop of mdct computations on a subsampled dictionary."},
    {"project_mclt",  project_mclt, METH_VARARGS,  "Loop of mdct computations."},
    {"project_masked_gabor",  project_masked_gabor, METH_VARARGS,  "Loop of gabor computations."},
    {"project_mclt_set",  project_mclt_set, METH_VARARGS,  "Loop of mdct computations."},
    {"project_mclt_NLset",  project_mclt_NLset, METH_VARARGS,  "Loop of mdct computations."},
    {"project_atom",  project_atom, METH_VARARGS,  "Project atom and retrieve characteristics"},
    {"project_atom_set",  project_atom_set, METH_VARARGS,  "Project atom and retrieve characteristics and reusing atom transform"},
    {"get_atom",  get_atom, METH_VARARGS,  "computes an atom waveform"},
    {"get_real_gabor_atom",  get_real_gabor_atom, METH_VARARGS,  "computes an atom waveform"},
    {"initialize_plans",  initialize_plans, METH_VARARGS,  "routine to initialize fftw plans"},
    {"clean_plans",  clean_plans, METH_VARARGS,  "The cleaning stage: must be called after computations"},  
    {"test",  test, METH_VARARGS,  "testing the parallelization."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};




/* This function is mandatory for Python Extension build */
PyMODINIT_FUNC
initparallelProjections(void)
{
    (void) Py_InitModule("parallelProjections", AllMethods);
 	import_array();  // Must be present for NumPy.  Called first after above line.
        
    
}
