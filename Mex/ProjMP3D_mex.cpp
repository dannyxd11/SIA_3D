#include "mex.h"
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "ProjMP3D.cpp"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int* hDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[2], hDim);
    double* h = mxGetPr(prhs[0]);
   
  
    //
	
    int* reDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], mxGetDimensions(prhs[1])[2], reDim);
    double* re = mxGetPr(prhs[1]);
   

    //

    int* v1Dim = new int[3];
    setDimensions(mxGetDimensions(prhs[2])[0], mxGetDimensions(prhs[2])[1], 1, v1Dim);
    double* v1 = mxGetPr(prhs[2]);


    //

    int* v2Dim = new int[3];
    setDimensions(mxGetDimensions(prhs[3])[0], mxGetDimensions(prhs[3])[1], 1, v2Dim);
    double* v2 = mxGetPr(prhs[3]);

    //


    int* v3Dim = new int[3];
    setDimensions(mxGetDimensions(prhs[4])[0], mxGetDimensions(prhs[4])[1], 1, v3Dim);
    double* v3 = mxGetPr(prhs[4]);
	
    //


    int* cDim = new int[3];
    setDimensions(mxGetDimensions(prhs[5])[0], mxGetDimensions(prhs[5])[1], 1, cDim);
    double* c = mxGetPr(prhs[5]);


    //

	double toln = mxGetPr(prhs[6])[0];
	double max = mxGetPr(prhs[7])[0];

    nlhs = 3;

	ProjMP3d( 	h, 
				re, reDim, 
				v1, v1Dim, 
				v2, v2Dim, 
				v3, v3Dim,
				c,  cDim, 				
				toln,
				max
			);

    plhs[0] = mxDuplicateArray(prhs[0]);
    plhs[1] = mxDuplicateArray(prhs[1]);
    plhs[2] = mxDuplicateArray(prhs[5]);

    delete [] hDim;
	delete [] reDim;
    delete [] cDim;
    delete [] v1Dim;
    delete [] v2Dim;
    delete [] v3Dim;
}