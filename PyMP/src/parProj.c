/*
 * parProj.c
 *
 *  Created on: Oct 20, 2011
 *      Author: moussall
 */

#include "parProj.h"

/* global variable of fftw_plans */
fftw_plan * ThreadPool_plans;  /*array of stratic fftw_plan struc*/
fftw_complex ** ThreadPool_inputs;
fftw_complex ** ThreadPool_outputs;

/* Also Initialize plans that will serve for local optimization calculations: Not parallelized (unnecessary) */
fftw_plan * ProjectionPlans;  /*array of stratic fftw_plan struc */
fftw_plan * InverseProjectionPlans;  /*array of stratic fftw_plan struc*/
fftw_complex ** Projectionsinputs;
fftw_complex ** Projectionsoutputs;

/* A bunch of global variables */
int * fft_sizes;
int * fft_opt_width; /* Width of optimization window  */
int size_number;
int nb_threads;


void sayHelloBastard(void){

	printf("Hello Dude: Lib fast Pursuit Version  %1.1f by M. Moussallam \n Contact moussall@telecom-paristech.fr \n",VERSION);


}

/*
 * Initializing FFTW plans, inputs and output vectors
 *
 */
int initialize(int * sizes  , int sNumber , int * tolerances, int max_thread_num){

	int size ;
	int  threadIdx;
	size_number = sNumber;
	/* Determines the number of available Threads */

	#pragma omp parallel
		{
			nb_threads = omp_get_num_threads();

		}

	if(max_thread_num>0){
		if (max_thread_num<=nb_threads){
			nb_threads = max_thread_num;
			omp_set_num_threads(nb_threads);

		}else{
			printf("You asked for more threads (%d) than there are available (%d), ignoring your request\n",
					max_thread_num, nb_threads);
		}
	}
	/*printf("%d threads availables \n", nb_threads);*/
	/* retrive the sizes for which the plan must be created */
	fft_sizes = (int*) malloc(sizeof(int) * size_number);
	fft_opt_width = (int*) malloc(sizeof(int) * size_number);
	if(DEBUG) printf("Allocating for %d sizes, %d threads \n", size_number, nb_threads);

	for (size=0; size < size_number ; size++){
		if (sizes[X*size] == 0){
			printf("ERROR in initialization, zero-size found is system 32 bits?\n");
			free(fft_sizes);
			free(fft_opt_width);
			return -1;
		}

		fft_sizes[size] =  sizes[X*size];
		fft_opt_width[size] = tolerances[X*size] * fft_sizes[size];
	}

	/* allocate for all threads*/
	ThreadPool_plans =  (fftw_plan*) fftw_malloc(sizeof(fftw_plan) * size_number * nb_threads);

	ThreadPool_inputs = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * size_number * nb_threads);
	ThreadPool_outputs = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * size_number * nb_threads);

	/* Also allocate non-threaded plans for LOMP */
	ProjectionPlans =  (fftw_plan*) fftw_malloc(sizeof(fftw_plan) * size_number );
	InverseProjectionPlans =  (fftw_plan*) fftw_malloc(sizeof(fftw_plan) * size_number );

	Projectionsinputs = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * size_number );
	Projectionsoutputs = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * size_number );


	for (size=0; size < size_number ; size++){

		for(threadIdx=0; threadIdx<nb_threads; threadIdx++){
			/* allocate FFTW vectors*/
			ThreadPool_inputs[(threadIdx*size_number)+size] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_sizes[size]);
			ThreadPool_outputs[(threadIdx*size_number)+size] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_sizes[size]);

			/* initialize FFTW plan: forward transform*/
			ThreadPool_plans[(threadIdx*size_number)+size] = fftw_plan_dft_1d(fft_sizes[size],
					ThreadPool_inputs[(threadIdx*size_number)+size],
					ThreadPool_outputs[(threadIdx*size_number)+size],
					FFTW_FORWARD, FFTW_ESTIMATE);


			if(DEBUG){
				printf(" Allocated plan for size %d and Thread %d \n" ,  fft_sizes[size], threadIdx);
			}
		}

		/* Also allocate double sized vectors for local projections */
		Projectionsinputs[size] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_opt_width[size]);
		Projectionsoutputs[size] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_opt_width[size]);

		/* initialize FFTW plans: forward and backward transform */
		ProjectionPlans[size] = fftw_plan_dft_1d(fft_opt_width[size],
				Projectionsinputs[size],
				Projectionsoutputs[size],
				FFTW_FORWARD, FFTW_ESTIMATE);

		InverseProjectionPlans[size] = fftw_plan_dft_1d(fft_opt_width[size],
				Projectionsinputs[size],
				Projectionsoutputs[size],
				FFTW_BACKWARD, FFTW_ESTIMATE);

	}
	/* everything's fine */
	if(DEBUG) printf("Allocation succeeded\n");
	return(0);
}


/*
 * Cleaning FFTW plans, inputs and output vectors
 *
 */
int clean(void){
	/* declarations */
	int size , threadIdx;

	/* Declarations - end */
	if(DEBUG) printf(" Starting Cleaning stage\n");
	for (size=0; size < size_number ; size++){

		for(threadIdx=0; threadIdx<nb_threads; threadIdx++){
			/* de-allocate FFTW vectors*/
			if (NULL != ThreadPool_inputs[(threadIdx*size_number)+size]){
				fftw_free(ThreadPool_inputs[(threadIdx*size_number)+size]);
			}
			if (NULL != ThreadPool_outputs[(threadIdx*size_number)+size]){
				fftw_free(ThreadPool_outputs[(threadIdx*size_number)+size]);
			}


			/* destroy plans */
			if (NULL != ThreadPool_plans[(threadIdx*size_number)+size]){
				fftw_destroy_plan(ThreadPool_plans[(threadIdx*size_number)+size]);
			}

			if(DEBUG) printf(" De-aAllocated plan for size %d and Thread %d \n" ,  fft_sizes[size] , threadIdx);
		}

		/* Also deallocate fftw variables for local projections */
		if(DEBUG) printf(" Deallocating Non-Threaded Plans \n");
		if (NULL != Projectionsinputs[size]){
			fftw_free(Projectionsinputs[size]);
		}
		if (NULL != Projectionsoutputs[size]){
			fftw_free(Projectionsoutputs[size]);
		}
		if (NULL != ProjectionPlans[size]){
			fftw_destroy_plan(ProjectionPlans[size]);
		}

		if (NULL != InverseProjectionPlans[size]){
			fftw_destroy_plan(InverseProjectionPlans[size]);
		}
	}

	if(DEBUG) printf(" Deallocating fft_sizes \n");

	if(NULL != fft_sizes){
		free(fft_sizes);
	}
	if(NULL != fft_opt_width){
		free(fft_opt_width);
	}
	return(0);
}

int getSizeNumber(void){

	return size_number;

}

/*
 * Projection routine for simple MDCT. Inputs are:
 * cin_data : pointer to the raw data
 * cin_vecProj : pointer to an already allocated projection Score vector the same size
 * cout_scoreTree : vector that contains the best score of each frame for quick max search
 * in_vecPreTwid :  pre-twidlling coefficient vector (see Ravelli's PhD)
 * in_vecPostTwid:  post-twidlling coefficient vector (see Ravelli's PhD)
 * start/end : indexes of first and last frame to compute
 * L : size of the transform
 * Ts : global offset to apply
 */
int projectMDCT(double * cin_data ,
		double * cin_vecProj , double * cout_scoreTree,
		fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
		int start , int end , int L, int Ts)
{
	/* declarations */
	int i,j,size;/*,n_data, n_frames; // usual iterator integers*/
	int K,T , threadIdx;
	double norm , max, realprod;

	//FILE *fp; // wisdom saving file
	//char * filename;

	fftw_complex *fftw_private_input, *fftw_private_output;  // FFTW input and output vectors
	fftw_plan fftw_private_plan;		// FFTW Plan

	/* declarations -  end */
	max = 0;
	realprod = 0;


	/* initialize some constants: half-window size and temporal offsets*/
	K = L/2;
	T = K/2 + Ts;
	norm = sqrt(2.00/((double)K));
	if(DEBUG)printf("DEBUG :  Current number of initialized sizes is %d\n",getSizeNumber());
	/* MultiThreaded version -  instantiate a pool of vectors */
	/* Retrieve inputs and outputs vector from global variables */
	size = -1;
	for (i=0 ; i < size_number ; i++){
		if (L == fft_sizes[i])
		{	size =i;
		if(DEBUG>1) printf("DEBUG :  found value %d in %d\n",L , i);
		break;
		}/*
		else{	printf("DEBUG :  value %d is not in %d = %d\n",L , i , fft_sizes[i]);
		}*/
	}
	if (size <0) {
		printf("ERROR : size %d not recognized in pre-built dictionary \n %d sizes availables : parallelFFT.c line 260", L , size_number);
		return(-1);
	}

	/* LOOP ON signal frames */
#pragma omp parallel for private(i,max,realprod,threadIdx,j,fftw_private_input,fftw_private_output,fftw_private_plan) shared(start,end, cin_data ,K,T,L,cin_vecPreTwid ,cin_vecProj ,cin_vecPostTwid,norm,cout_scoreTree ,size, size_number , ThreadPool_inputs , ThreadPool_outputs , ThreadPool_plans)//,fftw_private_input,fftw_private_output,fftw_private_plan)
	for(i=start; i < end ; i++){

		threadIdx = omp_get_thread_num();

		if(DEBUG>1) printf("DEBUG :  iteration %d attributed to Thread %d\n",i,threadIdx);

		/* Allocate FFTW vector for current thread : Already instantiated and available in ThreadPools*/
		fftw_private_input = ThreadPool_inputs[(threadIdx*size_number)+size];
		fftw_private_output = ThreadPool_outputs[(threadIdx*size_number)+size];

		fftw_private_plan = ThreadPool_plans[(threadIdx*size_number)+size];

		/* populate input */
		for(j=0; j < L; j++){

			fftw_private_input[j][0] = (double) (cin_data[j + i*K - T]) * cin_vecPreTwid[j][0];
			fftw_private_input[j][1] = (double) (cin_data[j + i*K - T]) * cin_vecPreTwid[j][1];
		}

		// perform FFT
		fftw_execute(fftw_private_plan);

		// post-twiddle and assignement to projection vector
		// simultaneous max search and storage in the score Tree
		max = 0;
		realprod = 0;

		/* Loop on frequency indexes */
		//#pragma omp parallel for
		for(j=0; j < K; j++){

			/* Real part is all we care about */
			realprod = ((double) (fftw_private_output[j][0]* cin_vecPostTwid[j][0]) -
					 (double) (fftw_private_output[j][1]* cin_vecPostTwid[j][1]) );


			cin_vecProj[j + i*K] = (double) norm * realprod;
			if(DEBUG>1) printf("DEBUG : new score of %2.6f, %2.6f at %d\n",cin_vecProj[j + i*K],realprod,j);
			if (fabs(cin_vecProj[j + i*K]) > max) {
				max = fabs(cin_vecProj[j + i*K]);
				cout_scoreTree[i] = max;
				if(DEBUG>1) printf("DEBUG : new max found of %2.2f on frame %d at freq %d\n",max,i,j);
			}

		}/*END  Loop on frequency indexes */


	} /*END loop on frames */

	return(1);
}

/* CHANGES 28/10/11 : projection vector is made real and contains the modulus times the sign of the real part */
int projectMCLT(double * cin_data ,
		double * cin_vecProj ,
		double * cout_scoreTree,
		fftw_complex * cin_vecPreTwid ,
		fftw_complex * cin_vecPostTwid,
		int start , int end , int L){

	/* declarations */
	int i,j,size;/*,n_data, n_frames; // usual iterator integers*/
	int K,T , threadIdx;
	double norm , max;

	fftw_complex *fftw_private_input , *fftw_private_output;  // FFTW input and output vectors
	fftw_plan fftw_private_plan;						// FFTW Plan
	fftw_complex prod;
	/*initialize some constants*/
	K = L/2;
	T = K/2;
	norm = (double)(sqrt(2.0/((double)K)));

	/*  Retrieve inputs and outputs vector from global variables*/
	size = -1;

	for (i=0 ; i < size_number ; i++){
		if (L == fft_sizes[i])
		{	size =i;
		break;
		}
	}
	if (size <0) {
		printf("Warning: FFTW has not been properly initiated\n");
		return(-1);
	}


	/* LOOP ON signal frames*/
#pragma omp parallel for default(none) private(i,prod,max,threadIdx,j,fftw_private_input,fftw_private_output,fftw_private_plan) shared(start,end, cin_data ,K,T,L,cin_vecPreTwid ,cin_vecProj ,cin_vecPostTwid,norm,cout_scoreTree ,size, size_number , ThreadPool_inputs , ThreadPool_outputs , ThreadPool_plans)
	for(i=start; i < end ; i++){

		threadIdx = omp_get_thread_num();
		/*Allocate FFTW vector for current thread : Already instantiated and available in ThreadPools*/
		fftw_private_input = ThreadPool_inputs[(threadIdx*size_number)+size];
		fftw_private_output = ThreadPool_outputs[(threadIdx*size_number)+size];

		fftw_private_plan = ThreadPool_plans[(threadIdx*size_number)+size];

		// Populate input data and pre-twiddle
		for(j=0; j < L; j++){

			/*C99*/
			fftw_private_input[j][0] = ( cin_data[j + i*K - T])* cin_vecPreTwid[j][0];
			fftw_private_input[j][1] = ( cin_data[j + i*K - T])* cin_vecPreTwid[j][1];
		}

		// perform FFT
		fftw_execute(fftw_private_plan);
		// post-twiddle and assignement to projection vector
		// simultaneous max search and storage in the score Tree
		max = 0;

		for(j=0; j < K; j++){

			/*not C99*/
			product(fftw_private_output[j], cin_vecPostTwid[j] , prod);

			/*cin_vecProj[j + i*K][0] = ((double) norm) * prod[0];
			cin_vecProj[j + i*K][1] = ((double) norm) * prod[1];*/

			cin_vecProj[j + i*K] = norm * modulus(prod);

			if ((cin_vecProj[j + (i*K)]) > max) {
				max = (cin_vecProj[j + (i*K)]);

				cout_scoreTree[i] = max;

			}

			/* Change the sign if real part is negative */
			if (prod[0]<0) cin_vecProj[j + i*K] *= -1;

/*			if (modulus(cin_vecProj[j + (i*K)]) > max) {
				max = modulus(cin_vecProj[j + (i*K)]);
				cout_scoreTree[i] = max;
			}*/

		}

	}/* END loop on signal frames*/
	return(1);
}

int projectSet(double * cin_data ,
		double * cin_vecProj ,
		double * cout_scoreTree,
		fftw_complex * cin_vecPreTwid ,
		fftw_complex * cin_vecPostTwid,
		int start , int end , int L, int type){

	int i,j;
	int K,T ,size , threadIdx;
	double norm ;
	fftw_complex newValue;
	fftw_complex *fftw_private_input , *fftw_private_output;  // FFTW input and output vectors
	fftw_plan fftw_private_plan;						// FFTW Plan
	double max ;
	// initialize some constants
	K = L/2;
	T = K/2;
	norm = (double)(sqrt(2.0/((double)K)));

	/*  Retrieve inputs and outputs vector from global variables*/
	size = -1;
	//printf("size number :  %d\n", size_number);
	for (i=0 ; i < size_number ; i++){
		//printf("Checking %d \n",  fft_sizes[i]);
		if (L == fft_sizes[i])
		{	size =i;
		break;
		}
	}
	if (size <0) {
		printf("Warning: FFTW has not been properly initiated\n");
		return -1;
	}
	/* LOOP ON signal frames*/
#pragma omp parallel for default(none) private(i,newValue,max,threadIdx,j,fftw_private_input,fftw_private_output,fftw_private_plan) shared(type,start,end, cin_data ,K,T,L,cin_vecPreTwid ,cin_vecProj ,cin_vecPostTwid,norm,cout_scoreTree ,size, size_number , ThreadPool_inputs , ThreadPool_outputs , ThreadPool_plans)
	for(i=start; i < end ; i++){

		threadIdx = omp_get_thread_num();
		/*Allocate FFTW vector for current thread : Already instantiated and available in ThreadPools*/
		fftw_private_input = ThreadPool_inputs[(threadIdx*size_number)+size];
		fftw_private_output = ThreadPool_outputs[(threadIdx*size_number)+size];

		fftw_private_plan = ThreadPool_plans[(threadIdx*size_number)+size];

		// Populate input data and pre-twiddle
		for(j=0; j < L; j++){
			/*C99*/
			fftw_private_input[j][0] = ( cin_data[j + i*K - T])*cin_vecPreTwid[j][0];
			fftw_private_input[j][1] = ( cin_data[j + i*K - T])*cin_vecPreTwid[j][1];
		}

		// perform FFT
		fftw_execute(fftw_private_plan);
		// post-twiddle and assignement to projection vector
		// simultaneous max search and storage in the score Tree
		max = 0;

		switch (type){
		/*TODO check if not faster to put the switch before*/
		/*Case 0 :  we sum all the absolute values of projections
		 BUGFIX : to accelerate we do not normalize the projections .. only in the best score tree*/
		case 0:
			for(j=0; j < K; j++){
				/*not C99*/
				product(fftw_private_output[j], cin_vecPostTwid[j] , newValue);

/*				cin_vecProj[j + i*K][0] += ((double) norm) * newValue[0];
				cin_vecProj[j + i*K][1] += ((double) norm) * newValue[1];

				if (modulus(cin_vecProj[j + (i*K)]) > max) {
					max = modulus(cin_vecProj[j + (i*K)]);
					cout_scoreTree[i] = max;
				}*/

				cin_vecProj[j + i*K] += ((double) norm) * modulus(newValue);

				if (cin_vecProj[j + i*K] > max) {
					max = cin_vecProj[j + i*K];
					cout_scoreTree[i] = max;
				}

				/*if (newValue[0]<0) cin_vecProj[j + i*K] *= -1;*/

			}
			break;
			/*case 1 :  we multiply all the absolute values of projections*/
		case 1:
			for(j=0; j < K; j++){

/*				not C99
				product(fftw_private_output[j], cin_vecPostTwid[j] , newValue);

				cin_vecProj[j + i*K][0] *= ((double) norm) * newValue[0];
				cin_vecProj[j + i*K][1] *= ((double) norm) * newValue[1];

				if (modulus(cin_vecProj[j + (i*K)]) > max) {
					max = modulus(cin_vecProj[j + (i*K)]);
					cout_scoreTree[i] = max;
				}*/
				product(fftw_private_output[j], cin_vecPostTwid[j] , newValue);

				cin_vecProj[j + i*K] *= ((double) norm) * modulus(newValue);

				if (cin_vecProj[j + i*K] > max) {
					max = cin_vecProj[j + i*K];
					cout_scoreTree[i] = max;
				}


			}

			break;
			/*case 3 : we take the new value if it is smaller*/
		case 2:
			for(j=0; j < K; j++){

				/*not C99*/
				product(fftw_private_output[j], cin_vecPostTwid[j] , newValue);
				newValue[0] *=  norm;
				newValue[1] *=  norm;

				if ( modulus(newValue)  < (cin_vecProj[j + (i*K)]) )
				{
					cin_vecProj[j + i*K] =  modulus(newValue);

				}

				if (cin_vecProj[j + i*K] > max) {
					max = cin_vecProj[j + i*K];
					cout_scoreTree[i] = max;
				}

			}

			break;
		}

/*		for(j=0; j < K; j++){
			if (cin_vecProj[j + i*K] > max) {
				max = cin_vecProj[j + i*K];
				cout_scoreTree[i] = max;
			}
		}*/


/*		for(j=0; j < K; j++){
			if (cabs(cin_vecProj[j + (i*K)]) > max) {
				max = (double)(cabs(cin_vecProj[j + (i*K)]));
				cout_scoreTree[i] = max;
			}
		}*/


	} /*END loop on signal frames*/

	return 0;
}

int projectAtom(double *cin_sigData,
		double *cin_atomData ,
		double *cout_score,
		int L, int scale){

	/* Declaration */
	fftw_complex * cin_sigfft, *cin_atomfft ; // fftw variables

	int out_atomTS , j , halfOffsetWidth;
	double atomEnergy , fact ;
	fftw_complex prod;
	fftw_complex *fftw_input , *fftw_output;  // FFTW input and output vectors
	fftw_plan fftw_p , fftw_ip;						// FFTW Plan

	int size , i;
	int maxIndex;
	/* Declaration - End  */

	/* L is the length of the vectors, the size of the transform*/
/*	maxLag = L/4;
	scale = L/2;*/

	if(NULL == fft_sizes) {
		printf("FATAL ERROR : wrong plan initialization \n");
		return -1;
	}


	/* Check that input data length half exists in the available plans*/
	size = -1;
	//printf("size number :  %d\n", size_number);
	for (i=0 ; i < size_number ; i++){
		//printf("Checking %d \n",  fft_sizes[i]);
		if (L == fft_opt_width[i])
		{	size =i;

		break;
		}
	}
	if (size <0) {
		printf("Warning: FFTW has not been properly initiated\n");
		return -1;
	}

	/*We have retrieved the correct total length, the corresponding atom length is scale*/

	/*scale = fft_sizes[size];*/

	halfOffsetWidth = (int) (L - scale)/2;

	if(DEBUG>1)printf("Scale found of %d hal offset of %d , Total L : %d\n", scale,halfOffsetWidth,L);
	/* allocate FFTW vectors*/
	/* Faster: grab an available plan from the threadpool*/
	fftw_input = Projectionsinputs[size];
	fftw_output = Projectionsoutputs[size];

	fftw_p = ProjectionPlans[size];
	fftw_ip = InverseProjectionPlans[size];

    // allocate fftw variables
    cin_sigfft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L);
    cin_atomfft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L);


	// Signal data FFT:
	//fftw_input =  (fftw_complex*) cin_sigData;
	//#pragma omp parallel for  private(j) shared(fftw_input , cin_sigData , L) default(none)
	for(j=0; j < L; j++){
		/*not C99*/
		fftw_input[j][0] = cin_sigData[j];
		fftw_input[j][1] = 0;
	}

	// perform FFT
	fftw_execute(fftw_p);

	for(j=0; j < L; j++){
		/* not C99*/
		cin_sigfft[j][0] = fftw_output[j][0];
		cin_sigfft[j][1] = fftw_output[j][1];
	}

	// Atom data FFT:
	atomEnergy = 0;
	//fftw_input = (fftw_complex*) cin_atomData;


	for(j=0; j < L; j++){

		/*not C99*/
		fftw_input[j][0] =  cin_atomData[j];
		fftw_input[j][1] =  0;
		// TODO check validity here
		atomEnergy +=  cin_atomData[j]*cin_atomData[j];
	}

	/*printf("initial energy of %2.6f \n" , atomEnergy);*/

	/* perform FFT*/
	fftw_execute(fftw_p);

	/* not C99*/
#pragma omp parallel for default(none) private(j,prod) shared(cin_atomfft,cin_sigfft, fftw_output, fftw_input,L)
	for(j=0; j < L; j++){

		cin_atomfft[j][0] = fftw_output[j][0];
		// TAKING CONJUGATE
		cin_atomfft[j][1] = - fftw_output[j][1];

		product(cin_sigfft[j] , cin_atomfft[j] , prod);

		fftw_input[j][0] = prod[0];
		fftw_input[j][1] = prod[1];
	}

	maxIndex = 0;
	out_atomTS = findMaxXcorr(fftw_ip, cout_score, maxIndex, halfOffsetWidth, fftw_output, scale, L);

	if (NULL == cout_score){
		printf("ERROR: parproj.c 659 :Fatalitas! cross correlation kind of failed!\n ");
		return(-1);
	}

	/*printf("PARALLELFFT : %2.6f   \n",out_atomScore);*/
	/*product(cin_sigfft[maxIndex] , cin_atomfft[maxIndex] , prod);*/
	/*printf("PARALLELFFT : %2.6f _ %2.6f _  %2.6f _ %2.6f\n",cin_sigfft[maxIndex][1], cin_atomfft[maxIndex][1],prod[0],prod[1]);*/
	//printf("found correlation max of %2.6f at %d , true value : %2.4f , %2.4f\n" , out_atomScore , out_atomTS , creal(fftw_output[maxIndex]) , cimag(fftw_output[maxIndex]));

	// now let us re-project the atom on the signal to adjust it's energy: Only if no pathological case
	if (cout_score[0] <0){
		fact = -sqrt( -cout_score[0]/atomEnergy);
	}else{
		fact = sqrt(cout_score[0]/atomEnergy);
	}
	/*printf("PARALLELFFT : %2.2f \n",fact);*/
	// perform calculus on the atom waveform
	/*printf("PARALLELFFT : %2.6f   \n",fact);*/
	for(j=0; j<L; j++){
		cin_atomData[j] *= fact;
	}

	/*if (abs(out_atomTS) >= halfOffsetWidth) printf("Ooops! found value %d, way over %d \n", out_atomTS , halfOffsetWidth);*/

	 fftw_free(cin_sigfft);
     fftw_free(cin_atomfft);

	// assign projection score value to output
/*	cout_score[0] = out_atomScore;*/

	/*printf("sent %d  - %d  - %d\n", out_atomTS , halfOffsetWidth , scale);*/
	return out_atomTS;
}


int findMaxXcorr(fftw_plan fftw_ip, double *cout_atomScore, int maxIndex,
		int halfOffsetWidth, fftw_complex *fftw_output, int scale, int L)
{

	int j, out_atomTS;
	double out_atomScore;

    // perform FFT
    /*printf("Executing backward transform\n");*/
    fftw_execute(fftw_ip);
    // max correlation search
    out_atomScore = 0;

    for(j = 0;j < halfOffsetWidth;j++){
        if(fabs(fftw_output[j][0]) > fabs(out_atomScore)){
            out_atomScore = fftw_output[j][0];
            maxIndex = j;
        }
    }

    // search positive offsets
    for(j = (halfOffsetWidth + scale) + 1;j < L;j++){
        if(fabs(fftw_output[j][0]) > fabs(out_atomScore)){
            out_atomScore = fftw_output[j][0];
            maxIndex = j;
        }
    }

    // normalizing  inverse fft
    out_atomScore *= (1.0 / (((double)L)));
    // circle permutation
    if(maxIndex > halfOffsetWidth){
        out_atomTS = maxIndex - L;
    }else{
        out_atomTS = maxIndex;
    }

    cout_atomScore[0] = out_atomScore;

    return(out_atomTS);

}

/* Need to provide some doc on this
 * same as projectAtom but avoid recomputing the atom's waveform fft
 * useful for multi-dimensionnal datas and lists of arrays
 */
int reprojectAtom(double *cin_sigData,
		double *cin_atomData,
		fftw_complex *cin_atomfft ,
		double *cout_score,
		int L, int scale,
		int sigIdx, int maj){

	/* Declaration */
	fftw_complex * cin_sigfft; // fftw variables

	int out_atomTS , j , halfOffsetWidth;
	double fact , atomEnergy;
	fftw_complex prod;
	fftw_complex *fftw_input , *fftw_output;  // FFTW input and output vectors
	fftw_plan fftw_p , fftw_ip;						// FFTW Plan

	int size , i;
	int maxIndex;
	/* Declaration - End  */

	if(NULL == fft_sizes) {
		printf("FATAL ERROR : wrong plan initialization \n");
		return -1;
	}
	/* Check that input data length half exists in the available plans*/
	size = -1;
	//printf("size number :  %d\n", size_number);
	for (i=0 ; i < size_number ; i++){
		//printf("Checking %d \n",  fft_sizes[i]);
		if (L == fft_opt_width[i])
		{	size =i;

		break;
		}
	}
	if (size <0) {
		printf("Warning: FFTW has not been properly initiated\n");
		return -1;
	}

	/*We have retrieved the correct total length, the corresponding atom length is scale*/
	halfOffsetWidth = (int) (L - scale)/2;

	if(DEBUG>1)printf("Scale found of %d hal offset of %d , Total L : %d\n", scale,halfOffsetWidth,L);
	/* allocate FFTW vectors*/
	/* Faster: grab an available plan from the threadpool*/
	fftw_input = Projectionsinputs[size];
	fftw_output = Projectionsoutputs[size];

	fftw_p = ProjectionPlans[size];
	fftw_ip = InverseProjectionPlans[size];

    // allocate fftw variables
	if(DEBUG) printf("Allocating fft for signal \n");

	cin_sigfft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L);


	// Signal data FFT:
	//fftw_input =  (fftw_complex*) cin_sigData;
	//#pragma omp parallel for  private(j) shared(fftw_input , cin_sigData , L) default(none)
	for(j=0; j < L; j++){
		/*not C99*/
		fftw_input[j][0] = cin_sigData[j];
		fftw_input[j][1] = 0;
	}

	// perform FFT
	fftw_execute(fftw_p);

	for(j=0; j < L; j++){
		/* not C99*/
		cin_sigfft[j][0] = fftw_output[j][0];
		cin_sigfft[j][1] = fftw_output[j][1];
	}
	atomEnergy = 0;
	if (sigIdx<1){
		// Atom data FFT:
		if(DEBUG) printf("First signal : Computing fft of atom wf \n");
		//fftw_input = (fftw_complex*) cin_atomData;

		for(j=0; j < L; j++){

			/*not C99*/
			fftw_input[j][0] =  cin_atomData[j];
			fftw_input[j][1] =  0;
			// TODO check validity here
			atomEnergy +=  cin_atomData[j]*cin_atomData[j];
		}

		if(DEBUG) printf("initial energy of %2.6f \n" , atomEnergy);
		// perform FFT

		fftw_execute(fftw_p);

		if(DEBUG) printf("Populating output \n");
		for(j = 0;j < L;j++){

			cin_atomfft[j][0] = fftw_output[j][0];
			// TAKING CONJUGATE
			cin_atomfft[j][1] = - fftw_output[j][1];

			product(cin_sigfft[j], cin_atomfft[j], prod);

			fftw_input[j][0] = prod[0];
			fftw_input[j][1] = prod[1];
		}
	}
	else{
		if(DEBUG) printf("Not First signal : fft of atom wf already computed \n");
	    for(j = 0;j < L;j++){
	        product(cin_sigfft[j], cin_atomfft[j], prod);
	        fftw_input[j][0] = prod[0];
	        fftw_input[j][1] = prod[1];

	        atomEnergy +=  cin_atomData[j]*cin_atomData[j];
	    }

	}
	if(DEBUG) printf("Looking for max Xcorr \n");
    maxIndex = 0;
    out_atomTS = findMaxXcorr(fftw_ip, cout_score, maxIndex, halfOffsetWidth, fftw_output, scale, L);
    if(DEBUG) printf("Found %d \n",out_atomTS);

    if (NULL == cout_score){
		printf("ERROR: parproj.c 659 :Fatalitas! cross correlation kinf of failed!\n ");
		return(-1);
	}
    /*printf("PARALLELFFT : %2.6f   \n",out_atomScore);*/
	/*product(cin_sigfft[maxIndex] , cin_atomfft[maxIndex] , prod);*/
	/*printf("PARALLELFFT : %2.6f _ %2.6f _  %2.6f _ %2.6f\n",cin_sigfft[maxIndex][1], cin_atomfft[maxIndex][1],prod[0],prod[1]);*/
	//printf("found correlation max of %2.6f at %d , true value : %2.4f , %2.4f\n" , out_atomScore , out_atomTS , creal(fftw_output[maxIndex]) , cimag(fftw_output[maxIndex]));

	/* now let us re-project the atom on the signal to adjust it's energy: Only if no pathological case
	 * Only if the MAJ flag is ON
	 */
	if (maj){
		if (cout_score[0] <0){
			fact = -sqrt( -cout_score[0]/atomEnergy);
		}else{
			fact = sqrt(cout_score[0]/atomEnergy);
		}
		/*printf("PARALLELFFT : %2.2f \n",fact);*/
		// perform calculus on the atom waveform
		/*printf("PARALLELFFT : %2.6f   \n",fact);*/
		for(j=0; j<L; j++){
			cin_atomData[j] *= fact;
		}
	}
	if(DEBUG) printf("Cleaning \n");
	/*if (abs(out_atomTS) >= halfOffsetWidth) printf("Ooops! found value %d, way over %d \n", out_atomTS , halfOffsetWidth);*/

	 fftw_free(cin_sigfft);

     /*fftw_free(cin_atomfft);*/
	 if(DEBUG) printf("Done \n");
	// assign projection score value to output
	/*cout_score[0] = out_atomScore;*/

	/*printf("sent %d  - %d  - %d\n", out_atomTS , halfOffsetWidth , scale);*/
	return out_atomTS;
}




int subprojectMDCT(double * cin_data ,
		double * cin_vecProj , double * cout_scoreTree,
		fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
		int start , int end , int L, int Ts,  int subFactor){

    int i,j; // usual iterator integers
    int K,T , threadIdx , size;
    double norm , max, realprod;
    fftw_complex *fftw_private_input, *fftw_private_output;  // FFTW input and output vectors
    fftw_plan fftw_private_plan;		// FFTW Plan

    /* declarations -  end*/
	max = 0;
	realprod = 0;

    // initialize some constants: half-window size and temporal offsets
    K = L/2;
    T = K/2 + Ts;
    norm = sqrt(2.00/((double)K));

	/*MultiThreaded version -  instantiate a pool of vectors
	           Retrieve inputs and outputs vector from global variables */
    size = -1;
    for (i=0 ; i < size_number ; i++){
    		if (L == fft_sizes[i])
    			{	size =i;
    				//printf("DEBUG :  found value %d in %d\n",L , i);
    				break;
    			}
/*    		else{	printf("DEBUG :  value %d is not in %d = %d\n",L , i , fft_sizes[i]);
    			}*/
    }
    if (size <0) {
    	printf("ERROR : size %d not recognized in pre-built dictionary \n %d sizes availables : parallelFFT.c line 260", L , size_number);
    	return -1;
    }

		/* LOOP ON signal frames*/
		//private(max,realprod,j,fftw_private_input,fftw_private_output,fftw_private_plan)
		// ADDITIONAL ENTRY :  only compute a subset of frames
		#pragma omp parallel for private(i,max,realprod,threadIdx,j,fftw_private_input,fftw_private_output,fftw_private_plan) shared(start,end, cin_data ,K,T,L,cin_vecPreTwid ,cin_vecProj ,cin_vecPostTwid,norm,cout_scoreTree ,size, size_number , ThreadPool_inputs , ThreadPool_outputs , ThreadPool_plans)//,fftw_private_input,fftw_private_output,fftw_private_plan)
		for(i=start; i < end ; i+=subFactor){

			threadIdx = omp_get_thread_num();

			 /*Allocate FFTW vector for current thread : Already instantiated and available in ThreadPools*/
			fftw_private_input = ThreadPool_inputs[(threadIdx*size_number)+size];
			fftw_private_output = ThreadPool_outputs[(threadIdx*size_number)+size];

			fftw_private_plan = ThreadPool_plans[(threadIdx*size_number)+size];

			for(j=0; j < L; j++){

				// UPDATE with no C99 complex type : need to separate real and imaginary parts
				fftw_private_input[j][0] = (double) (cin_data[j + i*K - T]) * cin_vecPreTwid[j][0];
				fftw_private_input[j][1] = (double) (cin_data[j + i*K - T]) * cin_vecPreTwid[j][1];

				// C99 version
				/*fftw_private_input[j] = ((fftw_complex) cin_data[j + i*K - T])*cin_vecPreTwid[j];*/
			}

			// perform FFT
			fftw_execute(fftw_private_plan);

			// post-twiddle and assignement to projection vector
			// simultaneous max search and storage in the score Tree
			max = 0;
			realprod = 0;

			 /*Loop on frequency indexes*/
			for(j=0; j < K; j++){

				/* Real part is all we care about */
				realprod = ((double) (fftw_private_output[j][0]* cin_vecPostTwid[j][0]) -
						 (double) (fftw_private_output[j][1]* cin_vecPostTwid[j][1]) );


				cin_vecProj[j + i*K] = (double) norm * realprod;

				if (fabs(cin_vecProj[j + i*K]) > max) {
					max = fabs(cin_vecProj[j + i*K]);
					cout_scoreTree[i] = max;
				}

			}/*END  Loop on frequency indexes */

		} /*END loop on frames*/


	return 0;
}

/* Gabor Dictionary projections computed using FFTS */
int projectMaskedGabor(double * cin_data,
                 double * cout_scoreTree,
                 fftw_complex * cin_vecProj,
                 double *penalty_mask,
                 int   start,
                 int   end,
                 int L) {

    /* Declarations */
    int i, j;  /* Loop indexes*/
    int K, T , blockIndex, threadIdx;
    double norm;

    /*omp_set_num_threads(omp_get_max_threads());*/


    fftw_complex *fftw_private_input , *fftw_private_output;  /* FFTW input and output vectors*/
    fftw_plan fftw_private_plan;						/* FFTW Plan*/
    double max;
    /* Declarations -  end */

    /* initialize some constants*/
    K = L/2;
    T = 0;

	/*  Retrieve inputs and outputs vector from global variables*/
    blockIndex = -1;

	for (i=0 ; i < size_number ; i++){
		if (L == fft_sizes[i])
		{	blockIndex =i;
		break;
		}
	}
	if (blockIndex <0) {
		printf("Warning: FFTW has not been properly initiated\n");
		return(-1);
	}

/*    window = (double*) mxMalloc(sizeof(double) * L);
    for(i=0; i<L; i++){
        window[i] = 0.5*(1 - cos( ((double)2.0*PI*j) / ((double)L) ));
    }*/

    /* Norm compensation of the signal windowing*/
    norm = (double)(sqrt(8.0/(3.0*(double)K)));

    if(DEBUG) printf("DEBUG : Projecting frames from %d to %d \n" , start, end);
    /* LOOP ON signal frames */
/*     #pragma omp parallel for private(i, max, threadIdx, j, fftw_private_input, fftw_private_output, fftw_private_plan) shared(absoluteMax,start, end, cin_data , K, T, L, cin_vecProj , norm, cout_scoreTree , blockIndex, size_number , ThreadPool_inputs , ThreadPool_outputs , ThreadPool_plans)*/

     #pragma omp parallel for default(none) private(i,j,fftw_private_input,fftw_private_output, fftw_private_plan, threadIdx,max) shared(cout_scoreTree,penalty_mask,cin_data,K,T,L,cin_vecProj,norm,start,end,ThreadPool_inputs,ThreadPool_outputs,ThreadPool_plans,size_number,blockIndex)
    for(i=start; i < end+1 ; i++){

        threadIdx = omp_get_thread_num();

        /* Allocate FFTW vector for current thread : Already instantiated and available in ThreadPools*/
        fftw_private_input = ThreadPool_inputs[(threadIdx*size_number)+blockIndex];
        fftw_private_output = ThreadPool_outputs[(threadIdx*size_number)+blockIndex];

        fftw_private_plan = ThreadPool_plans[(threadIdx*size_number)+blockIndex];

        /* Populate input data and windowing        */
        for(j=0; j < L; j++){

            /* UPDATE with no C99 complex type : need to separate real and imaginary parts +  windowing of the signal*/
				fftw_private_input[j][0] = (double) (cin_data[j + i*K - T]) *(0.5*(1 - cos( ((double)2.0*PI*j) / ((double)L) )));
				fftw_private_input[j][1] = 0 ;

        }

        /* perform FFT*/
        fftw_execute(fftw_private_plan);
        /* post-twiddle and assignement to projection vector
        // simultaneous max search and storage in the score Tree*/


        /* BUGFIX : do not allow for zero frequency*/
        cin_vecProj[i*K][0] = 0;
        cin_vecProj[i*K][1] = 0;
        max = 0.0;
        for(j=1; j < K; j++){

            /* NEW VERSION : not C99 compliant*/
				cin_vecProj[j + i*K][0] = penalty_mask[j + i*K] * norm * fftw_private_output[j][0];
				cin_vecProj[j + i*K][1] = penalty_mask[j + i*K] * norm * fftw_private_output[j][1];

                /* BUGFIX here - search is conducted on real values*/
                /* CHANGES : penalization by current mask before selecting the maximum*/
                if (( modulus(cin_vecProj[j + i*K])) > max) {
					max = ( modulus(cin_vecProj[j + i*K]));
					cout_scoreTree[i] = max;
                    if(DEBUG) printf("DEBUG found new value of %2.2f at frame %d located at %d \n" , max , i , j);
				}

        }
        /*if(max > absoluteMax){
            absoluteMax = max;
        }*/
    }/* END loop on signal frames */
    /*mxFree(window);*/


    return (1);

}

/* Gabor Dictionary projections computed using FFTS */
int projectPenalizedMDCT(double * cin_data,
                 double * cout_scoreTree,
                 double * cin_vecProj,
                 double *penalty_mask,
                 fftw_complex * cin_vecPreTwid , fftw_complex * cin_vecPostTwid,
                 int   start,
                 int   end,
                 int L, double lambda) {

    /* Declarations */
    int i, j;  /* Loop indexes*/
    int K, T , blockIndex, threadIdx;
    double norm, realprod;

    /*omp_set_num_threads(omp_get_max_threads());*/


    fftw_complex *fftw_private_input , *fftw_private_output;  /* FFTW input and output vectors*/
    fftw_plan fftw_private_plan;						/* FFTW Plan*/
    double max, penalty;
    /* Declarations -  end */

    /* initialize some constants*/
    K = L/2;
    T = K/2;
    norm = sqrt(2.00/((double)K));
	/*  Retrieve inputs and outputs vector from global variables*/
    blockIndex = -1;

	for (i=0 ; i < size_number ; i++){
		if (L == fft_sizes[i])
		{	blockIndex =i;
		break;
		}
	}
	if (blockIndex <0) {
		printf("Warning: FFTW has not been properly initiated\n");
		return(-1);
	}


    if(DEBUG) printf("DEBUG : Projecting frames from %d to %d \n" , start, end);
    /* LOOP ON signal frames */
     #pragma omp parallel for default(none) private(i,j,realprod,penalty,fftw_private_input,fftw_private_output, fftw_private_plan, threadIdx,max) shared(cout_scoreTree,penalty_mask,cin_vecPostTwid,cin_vecPreTwid,cin_data,K,T,L,cin_vecProj,norm,start,end,lambda,ThreadPool_inputs,ThreadPool_outputs,ThreadPool_plans,size_number,blockIndex)
    for(i=start; i < end+1 ; i++){

        threadIdx = omp_get_thread_num();

        /* Allocate FFTW vector for current thread : Already instantiated and available in ThreadPools*/
        fftw_private_input = ThreadPool_inputs[(threadIdx*size_number)+blockIndex];
        fftw_private_output = ThreadPool_outputs[(threadIdx*size_number)+blockIndex];

        fftw_private_plan = ThreadPool_plans[(threadIdx*size_number)+blockIndex];

        /* populate input */
		for(j=0; j < L; j++){

			fftw_private_input[j][0] = (double) (cin_data[j + i*K - T]) * cin_vecPreTwid[j][0];
			fftw_private_input[j][1] = (double) (cin_data[j + i*K - T]) * cin_vecPreTwid[j][1];
		}

        /* perform FFT*/
        fftw_execute(fftw_private_plan);
        /* post-twiddle and assignement to projection vector
        // simultaneous max search and storage in the score Tree*/


/*         BUGFIX : do not allow for zero frequency
        cin_vecProj[i*K][0] = 0;
        cin_vecProj[i*K][1] = 0;*/
        max = 0.0;
        realprod = 0.0;
        for(j=1; j < K; j++){

        	/* Real part is all we care about */
			realprod = ((double) (fftw_private_output[j][0]* cin_vecPostTwid[j][0]) -
					 (double) (fftw_private_output[j][1]* cin_vecPostTwid[j][1]) );


			cin_vecProj[j + i*K] = (double) norm * realprod;
			/* BUGFIX here - search is conducted on real values*/
			/* CHANGES : penalization by current mask before selecting the maximum*/
			penalty = lambda * penalty_mask[j + i*K];
			if (( fabs(cin_vecProj[j + i*K]) - penalty) > max) {
				max = fabs(cin_vecProj[j + i*K]) - penalty;
				cout_scoreTree[i] = max;
				if(DEBUG) printf("DEBUG found new value of %2.2f (penalty of %2.3f) at frame %d located at %d \n" , max , penalty,i , j);
			}

        }
        /*if(max > absoluteMax){
            absoluteMax = max;
        }*/
    }/* END loop on signal frames */
    /*mxFree(window);*/


    return (1);

}


/* Routine for complex product calculation with fftw_complex types */
void product( fftw_complex c1 , fftw_complex c2 , fftw_complex prod){
	/* fills real and imaginary parts of the product */
	prod[0] = c1[0]*c2[0] - c1[1]*c2[1];
	prod[1] = c1[1]*c2[0] + c1[0]*c2[1];
}

/* Routine for complex modulus calculation : no C99 */
double modulus( fftw_complex c ){
	return sqrt(c[0]*c[0] + c[1]*c[1]);
}
