#ifndef HNEW3D
#define HNEW3D

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "commonOps.cpp"

void hnew3d(double* cc, int* ccDimensions, double* v1, int* v1Dimensions, double* v2, int* v2Dimensions, double* v3, int* v3Dimensions, double* hnew, int *hnewDimensions){

    /*
     *
     * for zk=1:L3
         h_new(:,:,zk)=V1n1*ccn1*V2n1'*V3n1(zk);
       end
     *
     */

    for(int zk = 0; zk < v3Dimensions[0] * v3Dimensions[1]; zk++){
        MatrixMultiplyBLAS(v1, v2, getPlane(hnew, hnewDimensions, zk), v1Dimensions[0], v1Dimensions[1], v2Dimensions[0], 'N', 'T', v3[zk] * cc[0], 0);
    }

}

#endif