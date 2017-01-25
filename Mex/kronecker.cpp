//
// Created by Dan on 25/01/2017.
//
#include "mex.h"
#include "commonOps.cpp"

#include <iostream>

void kroneckerProduct(double* leftMatrix, int* leftMatrixDim, double* rightMatrix, int* rightMatrixDim, double* result, int* resultDim){

    int newHeight = leftMatrixDim[0] * rightMatrixDim[0];
    int newWidth = rightMatrixDim[1] * rightMatrixDim[1];

    //result = new double[newHeight * newWidth];
    //setDimensions(newHeight, newWidth, 1, resultDim);




    for (int m = 0; m < leftMatrixDim[0]; m++){
        for (int n = 0; n < leftMatrixDim[1]; n++){
            for (int x = 0; x < rightMatrixDim[0]; x++){
                for (int y = 0; y < rightMatrixDim[1]; y++){
                    result[(n * rightMatrixDim[1] + y) * newHeight + m * rightMatrixDim[0] + x] = leftMatrix[n * leftMatrixDim[1] + m] * rightMatrix[y * rightMatrixDim[1] + x];
                }
            }
        }
    }

}

void mexFunction(int nlhs, mxArray *plhs[],
                int nrhs, const mxArray *prhs[]){

//    double a[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
//    int* aDim = new int[3];
//    aDim[0] = 4; aDim[1] = 4; aDim[2] = 1;
//
//    double b[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
//    int* bDim = new int[3];
//    bDim[0] = 4; bDim[1] = 4; bDim[2] = 1;
//
//    double* result;
//    int* resultDim = new int[3];
//
//    kroneckerProduct(a, aDim, b, bDim, result, resultDim);



    int* aDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[2], aDim);
    double* a = mxGetPr(prhs[0]);


    //

    int* bDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], mxGetDimensions(prhs[1])[2], bDim);
    double* b = mxGetPr(prhs[1]);

    //


    nlhs = 1;

    int* resultDim = new int[3];
    setDimensions(aDim[0] * bDim[0], aDim[1] * bDim[1], 1, resultDim);
    int ndim = 3, dims[3] = {resultDim[0], resultDim[1], resultDim[2]};

    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);

    kroneckerProduct(a, aDim, b, bDim, mxGetPr(plhs[0]), resultDim);

    delete [] aDim;
    delete [] bDim;
    delete [] resultDim;

}