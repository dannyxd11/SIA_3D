#ifndef IPSP3D
#define IPSP3D

#include "commonOps.cpp"

void IPSP3d(double* re, int* reDim, double* v1, int* v1Dim, double* v2, int* v2Dim, double* v3, int* v3Dim, double* cc, int* ccDim){
    int n1 = v1Dim[1];
    int l3 = v3Dim[0];

    int aMatrixDim[] = {1, reDim[1], 1};
    double* aMatrix = new double[aMatrixDim[0] * aMatrixDim[1]]();

    for(int i = 0; i < n1; i++){
        cc[i] = 0;
        for(int j = 0; j < l3; j++){
            MatrixMultiplyBLAS(getCol(v1,v1Dim,i), getPlane(re, reDim, j), aMatrix, v1Dim[0], 1, aMatrixDim[1], 'T', 'N');
            MatrixMultiplyBLAS(aMatrix, getCol(v2, v2Dim, i), &cc[i], aMatrixDim[0], aMatrixDim[1], 1, 'N', 'N', getElement(v3, v3Dim, i, j), 1);
        }
    }

    delete [] aMatrix;
}

#endif