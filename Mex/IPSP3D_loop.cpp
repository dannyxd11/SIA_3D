#ifndef IPSP3D
#define IPSP3D

#include <iostream>
#include <stdlib.h>
#include <cstring>
//#include "commonOps.cpp"

void setDimensions(int height, int width, int depth, int* dim){
    dim[0] = height;
    dim[1] = width;
    dim[2] = depth;
}

void setDimensions(int* newDim, int* dim){
    setDimensions(newDim[0], newDim[1], newDim[2], dim);
}

double getElement(double* matrix, int* matrixDimensions,  int col, int row){
    return matrix[col * matrixDimensions[0] + row];
}

void setElement(double matrix[], int* matrixDimensions, int col, int row, double value){
    matrix[col * matrixDimensions[0] + row] = value;
}

double get3DElement(double* matrix, int* dim, int row, int col, int depth){
    return matrix[depth * dim[0] * dim[1] + col * dim[0] + row];
}

void transpose(double* matrix, int* matrixDimensions, double* newMx, int* newMxDimensions){
    setDimensions(matrixDimensions[1],matrixDimensions[0], 1, newMxDimensions);
    for(int i = 0; i < matrixDimensions[0]; i++) {
        for(int j = 0; j < matrixDimensions[1]; j++){
            setElement(newMx, newMxDimensions, i, j, getElement(matrix, matrixDimensions, j, i));
        }}
}

void IPSP3d_loop(double* re, int* reDim, double* v1, int* v1Dim, double* v2, int* v2Dim, double* v3, int* v3Dim, double* cc, int* ccDim){
    int n1 = v1Dim[1];
    int l1 = v1Dim[0];
    int l2 = v2Dim[0];
    int l3 = v3Dim[0];


    double* v1Trans = new double[v1Dim[0] * v1Dim[1]];
    int* v1TransDim = new int[3];
    transpose(v1, v1Dim, v1Trans, v1TransDim);

    for(int k = 0; k < n1; k++){
        cc[k] = 0;
        for(int zk = 0; zk < l3; zk++){
            for(int i = 0; i < l1; i++){
                for(int j = 0; j < l2; j++){
                    cc[k] +=  getElement(v1, v1Dim, k, i) * get3DElement(re, reDim, j, i, zk) * getElement(v2, v2Dim, k, j) * getElement(v3, v3Dim, k, zk) ;
                }
            }
        }
        //std::cout << k << "/" << n1 << "\tcc[k] = " << cc[k] << std::endl;
    }

    delete [] v1Trans;
    delete [] v1TransDim;
}

int main() {

    double dxElements[]  = {0.35355,0.49759,0.49039,0.47847,0.46194,0.44096,0.41573,0.38651,0.35355,0.3172,0.27779,0.2357,0.19134,0.14514,0.097545,0.049009,0,0.049009,0.097545,0.14514,0.19134,0.2357,0.27779,0.3172,0.35355,0.38651,0.41573,0.44096,0.46194,0.47847,0.49039,0.49759,1,0,0,0,0,0,0,0,
                            0.35355,0.47847,0.41573,0.3172,0.19134,0.049009,-0.097545,-0.2357,-0.35355,-0.44096,-0.49039,-0.49759,-0.46194,-0.38651,-0.27779,-0.14514,0,0.14514,0.27779,0.38651,0.46194,0.49759,0.49039,0.44096,0.35355,0.2357,0.097545,-0.049009,-0.19134,-0.3172,-0.41573,-0.47847,0,1,0,0,0,0,0,0,
                            0.35355,0.44096,0.27779,0.049009,-0.19134,-0.38651,-0.49039,-0.47847,-0.35355,-0.14514,0.097545,0.3172,0.46194,0.49759,0.41573,0.2357,0,0.2357,0.41573,0.49759,0.46194,0.3172,0.097545,-0.14514,-0.35355,-0.47847,-0.49039,-0.38651,-0.19134,0.049009,0.27779,0.44096,0,0,1,0,0,0,0,0,
                            0.35355,0.38651,0.097545,-0.2357,-0.46194,-0.47847,-0.27779,0.049009,0.35355,0.49759,0.41573,0.14514,-0.19134,-0.44096,-0.49039,-0.3172,0,0.3172,0.49039,0.44096,0.19134,-0.14514,-0.41573,-0.49759,-0.35355,-0.049009,0.27779,0.47847,0.46194,0.2357,-0.097545,-0.38651,0,0,0,1,0,0,0,0,
                            0.35355,0.3172,-0.097545,-0.44096,-0.46194,-0.14514,0.27779,0.49759,0.35355,-0.049009,-0.41573,-0.47847,-0.19134,0.2357,0.49039,0.38651,0,0.38651,0.49039,0.2357,-0.19134,-0.47847,-0.41573,-0.049009,0.35355,0.49759,0.27779,-0.14514,-0.46194,-0.44096,-0.097545,0.3172,0,0,0,0,1,0,0,0,
                            0.35355,0.2357,-0.27779,-0.49759,-0.19134,0.3172,0.49039,0.14514,-0.35355,-0.47847,-0.097545,0.38651,0.46194,0.049009,-0.41573,-0.44096,0,0.44096,0.41573,-0.049009,-0.46194,-0.38651,0.097545,0.47847,0.35355,-0.14514,-0.49039,-0.3172,0.19134,0.49759,0.27779,-0.2357,0,0,0,0,0,1,0,0,
                            0.35355,0.14514,-0.41573,-0.38651,0.19134,0.49759,0.097545,-0.44096,-0.35355,0.2357,0.49039,0.049009,-0.46194,-0.3172,0.27779,0.47847,0,0.47847,0.27779,-0.3172,-0.46194,0.049009,0.49039,0.2357,-0.35355,-0.44096,0.097545,0.49759,0.19134,-0.38651,-0.41573,0.14514,0,0,0,0,0,0,1,0,
                            0.35355,0.049009,-0.49039,-0.14514,0.46194,0.2357,-0.41573,-0.3172,0.35355,0.38651,-0.27779,-0.44096,0.19134,0.47847,-0.097545,-0.49759,0,0.49759,0.097545,-0.47847,-0.19134,0.44096,0.27779,-0.38651,-0.35355,0.3172,0.41573,-0.2357,-0.46194,0.14514,0.49039,-0.049009,0,0,0,0,0,0,0,1
    };

    double dyElements[]  = {0.35355,0.49759,0.49039,0.47847,0.46194,0.44096,0.41573,0.38651,0.35355,0.3172,0.27779,0.2357,0.19134,0.14514,0.097545,0.049009,0,0.049009,0.097545,0.14514,0.19134,0.2357,0.27779,0.3172,0.35355,0.38651,0.41573,0.44096,0.46194,0.47847,0.49039,0.49759,1,0,0,0,0,0,0,0,
                            0.35355,0.47847,0.41573,0.3172,0.19134,0.049009,-0.097545,-0.2357,-0.35355,-0.44096,-0.49039,-0.49759,-0.46194,-0.38651,-0.27779,-0.14514,0,0.14514,0.27779,0.38651,0.46194,0.49759,0.49039,0.44096,0.35355,0.2357,0.097545,-0.049009,-0.19134,-0.3172,-0.41573,-0.47847,0,1,0,0,0,0,0,0,
                            0.35355,0.44096,0.27779,0.049009,-0.19134,-0.38651,-0.49039,-0.47847,-0.35355,-0.14514,0.097545,0.3172,0.46194,0.49759,0.41573,0.2357,0,0.2357,0.41573,0.49759,0.46194,0.3172,0.097545,-0.14514,-0.35355,-0.47847,-0.49039,-0.38651,-0.19134,0.049009,0.27779,0.44096,0,0,1,0,0,0,0,0,
                            0.35355,0.38651,0.097545,-0.2357,-0.46194,-0.47847,-0.27779,0.049009,0.35355,0.49759,0.41573,0.14514,-0.19134,-0.44096,-0.49039,-0.3172,0,0.3172,0.49039,0.44096,0.19134,-0.14514,-0.41573,-0.49759,-0.35355,-0.049009,0.27779,0.47847,0.46194,0.2357,-0.097545,-0.38651,0,0,0,1,0,0,0,0,
                            0.35355,0.3172,-0.097545,-0.44096,-0.46194,-0.14514,0.27779,0.49759,0.35355,-0.049009,-0.41573,-0.47847,-0.19134,0.2357,0.49039,0.38651,0,0.38651,0.49039,0.2357,-0.19134,-0.47847,-0.41573,-0.049009,0.35355,0.49759,0.27779,-0.14514,-0.46194,-0.44096,-0.097545,0.3172,0,0,0,0,1,0,0,0,
                            0.35355,0.2357,-0.27779,-0.49759,-0.19134,0.3172,0.49039,0.14514,-0.35355,-0.47847,-0.097545,0.38651,0.46194,0.049009,-0.41573,-0.44096,0,0.44096,0.41573,-0.049009,-0.46194,-0.38651,0.097545,0.47847,0.35355,-0.14514,-0.49039,-0.3172,0.19134,0.49759,0.27779,-0.2357,0,0,0,0,0,1,0,0,
                            0.35355,0.14514,-0.41573,-0.38651,0.19134,0.49759,0.097545,-0.44096,-0.35355,0.2357,0.49039,0.049009,-0.46194,-0.3172,0.27779,0.47847,0,0.47847,0.27779,-0.3172,-0.46194,0.049009,0.49039,0.2357,-0.35355,-0.44096,0.097545,0.49759,0.19134,-0.38651,-0.41573,0.14514,0,0,0,0,0,0,1,0,
                            0.35355,0.049009,-0.49039,-0.14514,0.46194,0.2357,-0.41573,-0.3172,0.35355,0.38651,-0.27779,-0.44096,0.19134,0.47847,-0.097545,-0.49759,0,0.49759,0.097545,-0.47847,-0.19134,0.44096,0.27779,-0.38651,-0.35355,0.3172,0.41573,-0.2357,-0.46194,0.14514,0.49039,-0.049009,0,0,0,0,0,0,0,1
    };

    double dzElements[] = {0.57735,0.78868,0.70711,0.57735,0.40825,0.21132,0,0.21132,0.40825,0.57735,0.70711,0.78868,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0.57735,0.57735,4.9996e-17,-0.57735,-0.8165,-0.57735,0,0.57735,0.8165,0.57735,9.9992e-17,-0.57735,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0.57735,0.21132,-0.70711,-0.57735,0.40825,0.78868,0,0.78868,0.40825,-0.57735,-0.70711,0.21132,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    double reElements[] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6};

    int* dxDim = new int[3];
    int* dyDim = new int[3];
    int* dzDim = new int[3];
    int* reDim = new int[3];
    int* ccDim = new int[3];

    setDimensions(8, 40, 1, dxDim);
    setDimensions(8, 40, 1, dyDim);
    setDimensions(3, 40, 1, dzDim);
    setDimensions(8, 8, 3, reDim);
    setDimensions(1, dxDim[1], 1, ccDim);

    double* cc = new double[ccDim[1]];


    IPSP3d_loop(reElements, reDim, dxElements, dxDim, dyElements, dyDim, dzElements, dzDim, cc, ccDim);


    delete [] reDim;
    delete [] dxDim;
    delete [] dyDim;
    delete [] dzDim;
    delete [] cc;
    delete [] ccDim;
    return 0;
}

#endif