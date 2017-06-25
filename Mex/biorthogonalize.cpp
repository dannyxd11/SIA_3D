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

    int betaDim[] = {(int)mxGetDimensions(prhs[0])[0], (int)mxGetDimensions(prhs[0])[1], 1};
    int qkDim[] = {(int)mxGetDimensions(prhs[1])[0], (int)mxGetDimensions(prhs[1])[1], 1};

    double* qk = mxGetPr(prhs[1]);

    int newAtomDim[] = {(int)mxGetDimensions(prhs[2])[0], (int)mxGetDimensions(prhs[2])[1], 1};
    double* newAtom = mxGetPr(prhs[2]);

    double nork = mxGetPr(prhs[3])[0];

    nlhs = 1;

    plhs[0] = mxDuplicateArray(prhs[0]);
    double* beta = mxGetPr(plhs[0]);
    biorthogonalize(beta, betaDim, qk, qkDim, newAtom, newAtomDim, nork);

}