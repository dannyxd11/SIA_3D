//
// Created by Dan on 12/12/2016.
//
#include "mex.h"
#include "commonOps.cpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	int QDim[] = {(int)mxGetDimensions(prhs[0])[0], (int)mxGetDimensions(prhs[0])[1], 1};
	int newAtomDim[] = {(int)mxGetDimensions(prhs[1])[0], (int)mxGetDimensions(prhs[1])[1], 1};

    double* newAtom = mxGetPr(prhs[1]);

    nlhs = 1;

    plhs[0] = mxDuplicateArray(prhs[0]);

    double* Q;
    double *ptr;
    double *newptr;

    size_t nbytes =  QDim[0] * (QDim[1]+1) * sizeof(*ptr);

    ptr = mxGetPr(plhs[0]);
    newptr = (double*)mxRealloc(ptr, nbytes);
    mxSetPr(plhs[0], newptr);
    Q = mxGetPr(plhs[0]);
    mxSetN(plhs[0],QDim[1] + 1);

    orthogonalize(Q, QDim, newAtom, newAtomDim);
}

