#ifndef IP3D
#define IP3D

#include "commonOps.cpp"

void IP3d(double* re, int* reDimensions, double* v1, int* v1Dimensions, double* v2, int* v2Dimensions, double* v3, int* v3Dimensions, double* cc, int *ccDimensions) {

    for (int i = 0; i < ccDimensions[0] * ccDimensions[1] * ccDimensions[2]; i++) { cc[i] = 0; }

    int *aMatrixDimensions = new int[3];
    setDimensions(v1Dimensions[1], reDimensions[1], 1, aMatrixDimensions);
    double *aMatrix = new double[aMatrixDimensions[0] * aMatrixDimensions[1]];

    for (int m3 = 0; m3 < v3Dimensions[1]; m3++) {
        for (int zk = 0; zk < v3Dimensions[0]; zk++) {
            MatrixMultiplyBLAS(v1, getPlane(re, reDimensions, zk), aMatrix, v1Dimensions[0], v1Dimensions[1],
                               aMatrixDimensions[1], 'T', 'N');
            MatrixMultiplyBLAS(aMatrix, v2, getPlane(cc, ccDimensions, m3), aMatrixDimensions[0], aMatrixDimensions[1],
                               v2Dimensions[1], 'N', 'N', getElement(v3, v3Dimensions, m3, zk), 1);
        }
    }

    delete[] aMatrix;
    delete[] aMatrixDimensions;
}

#endif




