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

    int* betaDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], 1, betaDim);

    int* qkDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], 1, qkDim);
    double* qk = mxGetPr(prhs[1]);

    int* newAtomDim = new int[3];
    setDimensions(mxGetDimensions(prhs[2])[0], mxGetDimensions(prhs[2])[1], 1, newAtomDim);
    double* newAtom = mxGetPr(prhs[2]);

    double nork = mxGetPr(prhs[3])[0];

    nlhs = 1;

    plhs[0] = mxDuplicateArray(prhs[0]);
    double* beta = mxGetPr(plhs[0]);
    biorthogonalize(beta, betaDim, qk, qkDim, newAtom, newAtomDim, nork);

    delete [] betaDim;
    delete [] qkDim;
    delete [] newAtomDim;

}

