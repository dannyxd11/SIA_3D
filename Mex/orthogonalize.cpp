//
// Created by Dan on 12/12/2016.
//
#include "mex.h"
#include "commonOps.cpp"

/*
 * k=size(Q,2) +1;
 *    for p=1:k -1
 *        % Orthogonalization
 *
 *        Q(:,k) =    new_atom    -   (Q(:,p)' * new_atom)  *   Q(:,p);
 *     end
 *
 */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	int* QDim = new int[3];
	setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], 1, QDim);

	int* newAtomDim = new int[3];
	setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], 1, newAtomDim);
	double* newAtom = mxGetPr(prhs[1]);

    nlhs = 1;

//    plhs[0] = mxCreateDoubleMatrix(QDim[0], QDim[1] + 1, mxREAL);
//
//	double* Q = (double*)mxCalloc(QDim[0] * (QDim[1] + 1), sizeof(double));
//	orthogonalize(Q, QDim, newAtom, newAtomDim);
//	mxSetPr(plhs[0], Q);


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



	delete [] QDim;
	delete [] newAtomDim;







}

