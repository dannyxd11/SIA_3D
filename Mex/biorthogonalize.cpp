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

void biorthogonalize(double* beta, int* betaDim, double* qk, int* qkDim, double* newAtom, int* newAtomDim, double nork){

    // beta = beta - Qk * (new_atom'*beta) / nork;
    // Width of second, height of first

    double* multResult = new double[betaDim[1] * newAtomDim[0]];
    int* multResultDim = new int[3];
    setDimensions(betaDim[0],newAtomDim[0],1, multResultDim);

    double* finalResult = new double[multResultDim[1]*qkDim[0]];
    int* finalResultDim = new int[3];
    setDimensions(qkDim[0], multResultDim[1], 1, finalResultDim);

    double* transposed = new double[newAtomDim[0] * newAtomDim[1]];
    int* transposedDim = new int[3];
    transpose(newAtom, newAtomDim, transposed, transposedDim);

    blasMultiply(transposed, transposedDim, beta, betaDim, multResult, multResultDim);
    blasMultiply(qk, qkDim, multResult, multResultDim, finalResult, finalResultDim);

    for (int n = 0; n < betaDim[0] * betaDim[1]; n++){
        beta[n] -= finalResult[n] / nork;
    }

    delete [] multResult;
    delete [] multResultDim;
    delete [] finalResult;
    delete [] finalResultDim;
    delete [] transposed;
    delete [] transposedDim;
}



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

