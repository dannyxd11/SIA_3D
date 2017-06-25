#include "mex.h"
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "ProjMP3D.cpp"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int hDim[] = {(int)mxGetDimensions(prhs[0])[0], (int)mxGetDimensions(prhs[0])[1], (int)mxGetDimensions(prhs[0])[2]};
    double* h = mxGetPr(prhs[0]);
   
  
    //
	
    int reDim[] = {(int)mxGetDimensions(prhs[1])[0], (int)mxGetDimensions(prhs[1])[1], (int)mxGetDimensions(prhs[1])[2]};
    double* re = mxGetPr(prhs[1]);
   

    //

    int v1Dim[] = {(int)mxGetDimensions(prhs[2])[0], (int)mxGetDimensions(prhs[2])[1], 1};
    double* v1 = mxGetPr(prhs[2]);


    //

    int v2Dim[] = {(int)mxGetDimensions(prhs[3])[0], (int)mxGetDimensions(prhs[3])[1], 1};
    double* v2 = mxGetPr(prhs[3]);

    //


    int v3Dim[] = {(int)mxGetDimensions(prhs[4])[0], (int)mxGetDimensions(prhs[4])[1], 1};
    double* v3 = mxGetPr(prhs[4]);
	
    //


    int cDim[] = {(int)mxGetDimensions(prhs[5])[0], (int)mxGetDimensions(prhs[5])[1], 1};
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

}