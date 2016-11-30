#include "mex.h"
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "commonOps.cpp"




void hnew3d(double* cc, int* ccDimensions, double* v1, int* v1Dimensions, double* v2, int* v2Dimensions, double* v3, int* v3Dimensions, double* hnew, int *hnewDimensions){
    setDimensions(v1Dimensions[0],v2Dimensions[0], v3Dimensions[0], hnewDimensions);
    //hnew = new double[hnewDimensions[0] * hnewDimensions[1] * hnewDimensions[2]];

    for(int i = 0; i < hnewDimensions[0] * hnewDimensions[1] * hnewDimensions[2]; i++){
        hnew[i] = 0;
    }

    double *v2Trans = new double[v2Dimensions[0] * v2Dimensions[1]];
    int* v2TransDimensions = new int[3];

    int* aMatrixDimensions = new int[3];
    setDimensions(v1Dimensions[0], v2Dimensions[0], 1, aMatrixDimensions);
    double* aMatrix = new double[aMatrixDimensions[0] * aMatrixDimensions[1]];


    matrixScalarMultiplication(v1, v1Dimensions, cc[0]);
    transpose(v2, v2Dimensions, v2Trans, v2TransDimensions);
    blasMultiply(v1, v1Dimensions, v2Trans, v2TransDimensions, aMatrix, aMatrixDimensions);

    
        for(int zk = 0; zk < v3Dimensions[0] * v3Dimensions[1]; zk++){
            double* temp = new double[aMatrixDimensions[0] * aMatrixDimensions[1]];
            std::memcpy(temp, aMatrix, aMatrixDimensions[0] * aMatrixDimensions[1] * sizeof(double));
            /*
             *
             * for zk=1:L3
                 h_new(:,:,zk)=V1n1*ccn1*V2n1'*V3n1(zk);
               end
             *
             */
            matrixScalarMultiplication(temp, aMatrixDimensions, v3[zk]);
            setPlane(temp, aMatrixDimensions, hnew, hnewDimensions, zk);
            delete [] temp;
        }


    delete [] aMatrix;
    delete [] aMatrixDimensions;
    delete [] v2Trans;
    delete [] v2TransDimensions;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int* ccDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[2], ccDim);
    double* ccElements = mxGetPr(prhs[0]);
   

    //

    int* dxDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], 1, dxDim);
    double* dxElements = mxGetPr(prhs[1]);


    //

    int* dyDim = new int[3];
    setDimensions(mxGetDimensions(prhs[2])[0], mxGetDimensions(prhs[2])[1], 1, dyDim);
    double* dyElements = mxGetPr(prhs[2]);

    //


    int* dzDim = new int[3];
    setDimensions(mxGetDimensions(prhs[3])[0], mxGetDimensions(prhs[3])[1], 1, dzDim);
    double* dzElements = mxGetPr(prhs[3]);


    //

    int* hnewDim = new int[3];
    setDimensions(dxDim[0], dyDim[0], dzDim[0], hnewDim);
    nlhs = 1;
    int ndim = 3, dims[3] = {hnewDim[0], hnewDim[1], hnewDim[2]};
    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    double* hnew = mxGetPr(plhs[0]);

    hnew3d(ccElements, ccDim,
           dxElements, dxDim,
           dyElements, dyDim,
           dzElements, dzDim,
           hnew, hnewDim);

    delete [] ccDim;
    delete [] hnewDim;
    delete [] dxDim;
    delete [] dyDim;
    delete [] dzDim;
}