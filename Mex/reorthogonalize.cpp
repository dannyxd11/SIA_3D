//
// Created by Dan on 12/12/2016.
//
#include "mex.h"
#include "commonOps.cpp"

/*
 *   function [Q]=Reorthogonalize(Q,zmax);
 *
 * k=size(Q,2);
 *   for zi=1:zmax
 *      for p=1:k-1
 *          Q(:,k)=Q(:,k)-(Q(:,p)'*Q(:,k))*Q(:,p);
 *      end
 *   end
 *
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    int* QDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], 1, QDim);

    int zmax = mxGetPr(prhs[1])[0];
    //setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], 1, newAtomDim);
    //double* newAtom = mxGetPr(prhs[1]);

    nlhs = 1;

//    plhs[0] = mxCreateDoubleMatrix(QDim[0], QDim[1] + 1, mxREAL);
//
//	double* Q = (double*)mxCalloc(QDim[0] * (QDim[1] + 1), sizeof(double));
//	orthogonalize(Q, QDim, newAtom, newAtomDim);
//	mxSetPr(plhs[0], Q);


    plhs[0] = mxDuplicateArray(prhs[0]);
    double* Q = mxGetPr(plhs[0]);
    reorthogonalize(Q, QDim, zmax);



    delete [] QDim;
}

