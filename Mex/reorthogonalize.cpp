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

void reorthogonalize(double* Q, int* QDim, int zmax){

    // k=size(Q,2);
    int k = QDim[1];

    //
    for(int z = 0; z < zmax; z++){
        double multResult1 = 0;
        int* multResult1Dim = new int[3];
        setDimensions(1,1,1, multResult1Dim);

        for(int p = 0; p < k - 1; p++){


            double* qpTranspose = new double[QDim[0]];
            int* qpTransDim = new int[3];
            setDimensions(QDim[0], 1, 1, qpTransDim);
//
            double* qp = getCol(Q,QDim,p);
            int* qpDim = new int[3];
            setDimensions(QDim[0], 1, 1, qpDim);
//
            double* qk = getCol(Q, QDim, k-1);
            int* qkDim = new int[3];
            setDimensions(QDim[0], 1, 1, qkDim);

            transpose(qp, qpDim, qpTranspose, qpTransDim);
            blasMultiply(qpTranspose, qpTransDim, qk, qkDim, &multResult1, multResult1Dim);

            for (int n = 0; n < QDim[0]; n++){
                setElement(Q, QDim, k-1, n, qk[n] - (multResult1 * qp[n]));
            }

            delete [] qpTranspose;
            delete [] qpTransDim;
            delete [] qpDim;
            delete [] qkDim;
        }

        delete [] multResult1Dim;
    }


}



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

