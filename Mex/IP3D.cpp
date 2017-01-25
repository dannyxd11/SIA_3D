#ifndef IP3D
#define IP3D

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "commonOps.cpp"

void IP3d(double* re, int* reDimensions, double* v1, int* v1Dimensions, double* v2, int* v2Dimensions, double* v3, int* v3Dimensions, double* cc, int *ccDimensions){

    //Declaration & Initialization of variables
    for(int i = 0; i < ccDimensions[0] * ccDimensions[1] * ccDimensions[2]; i++){
        cc[i] = 0;
    }

    double *v1Trans = new double[v1Dimensions[0] * v1Dimensions[1]];
    int* v1TransDimensions = new int[3];
    transpose(v1, v1Dimensions, v1Trans, v1TransDimensions);

    double *b1;
    int* b1Dimensions = new int[3];
    setDimensions(reDimensions[0], reDimensions[1], 1, b1Dimensions);


    int* aMatrixDimensions = new int[3];
    setDimensions(v1TransDimensions[0], b1Dimensions[1], 1, aMatrixDimensions);
    double* aMatrix = new double[aMatrixDimensions[0] * aMatrixDimensions[1]];

    int* calcMatrixDimensions = new int[3];
    setDimensions(aMatrixDimensions[0], v2Dimensions[1], 1, calcMatrixDimensions);
    double* calcMatrix = new double[calcMatrixDimensions[0] * calcMatrixDimensions[1]];


    int* eleMultMatrixDimensions = new int[3];
    setDimensions(calcMatrixDimensions, eleMultMatrixDimensions);
    double* eleMultMatrix = new double[eleMultMatrixDimensions[0] * eleMultMatrixDimensions[1]];

    double* eleAddMatrix;
    int* eleAddMatrixDimensions = new int[3];
    setDimensions(ccDimensions[0], ccDimensions[1], 1, eleAddMatrixDimensions);

    for(int m3 = 0; m3 < v3Dimensions[1]; m3++){
        double* ccPlane  = new double[ccDimensions[0] * ccDimensions[1]];
        for (int i = 0; i < ccDimensions[0] * ccDimensions[1]; i++){
            ccPlane[i] = 0;
        }
        int* ccPlaneDimensions = new int[3];
        setDimensions(ccDimensions[0], ccDimensions[1], 1, ccPlaneDimensions);

        for(int zk = 0; zk < v3Dimensions[0]; zk++){
            b1 = getPlane(re, reDimensions, zk);
            blasMultiply(v1Trans, v1TransDimensions, b1, b1Dimensions, aMatrix, aMatrixDimensions);
            blasMultiply(aMatrix, aMatrixDimensions, v2, v2Dimensions, calcMatrix, calcMatrixDimensions);
            matrixScalarMultiplication(calcMatrix, calcMatrixDimensions, getElement(v3,v3Dimensions,m3,zk));
            eleAddMatrix = getPlane(cc, ccDimensions, m3);
            //setDimensions(ccDimensions[0], ccDimensions[1], 1, eleAddMatrixDimensions);
            elementAddition(eleAddMatrix, eleAddMatrixDimensions, calcMatrix, calcMatrixDimensions, ccPlane, ccPlaneDimensions);
        }
        setPlane(ccPlane, ccPlaneDimensions, cc, ccDimensions, m3);
        delete [] ccPlane;
        delete [] ccPlaneDimensions;
    }

    delete [] aMatrix;
    delete [] aMatrixDimensions;
    // delete [] b1;
    delete [] b1Dimensions;
    delete [] eleMultMatrix;
    delete [] eleMultMatrixDimensions;
    delete [] eleAddMatrixDimensions;
    delete [] calcMatrix;
    delete [] calcMatrixDimensions;
    delete [] v1Trans;
    delete [] v1TransDimensions;
}
#endif
