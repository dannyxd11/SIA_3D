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

    int QDim[] = {mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], 1};

    int zmax = mxGetPr(prhs[1])[0];

    nlhs = 1;


    plhs[0] = mxDuplicateArray(prhs[0]);
    double* Q = mxGetPr(plhs[0]);
    reorthogonalize(Q, QDim, zmax);

}

