#include "mex.h"
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "MyBLAS.h"

using namespace std;


//#include "mex.h
//#include "MyBLAS.h"

void setDimensions(int width, int height, int depth, int* dim){
    dim[0] = width;
    dim[1] = height;
    dim[2] = depth;
}

void setDimensions(int* newDim, int* dim){
    setDimensions(newDim[0], newDim[1], newDim[2], dim);
}

void elementAddition(double* matrixA, int* aDim, double* matrixB, int* bDim, double* results, int* rDim){
    setDimensions(aDim[0], aDim[1], 1, rDim);
    for (int i = 0; i < aDim[0] * aDim[1]; i++){
        results[i] = results[i] + matrixA[i] + matrixB[i];
    }
}

void matrixScalarMultiplication(double* matrixA, int* dim, double scalar, double* results, int* resultsDim){

    setDimensions(dim, resultsDim);
    for (int i = 0; i < resultsDim[0] * resultsDim[1]; i++){
        results[i] = matrixA[i] * scalar;
    }
}

double getElement(double* matrix, int* matrixDimensions, int row, int col){
    return matrix[row*matrixDimensions[0]+col];
}

void setElement(double matrix[], int* matrixDimensions, int row, int col, double value){
    matrix[row*matrixDimensions[0]+col] = value;
}

double get3DElement(double* matrix, int* dim, int row, int col, int depth){
    return matrix[depth*dim[0]*dim[1]+row*dim[0]+col];
}

double* getPlane(double* matrix, int* matrixDimensions, int depth) {
    return matrix + (matrixDimensions[0] * matrixDimensions[1] * depth);
}

void setPlane(double* plane, int* planeDim, double* matrix3d, int* matrixDim, int dimension){
    std::memcpy(&matrix3d[matrixDim[0] * matrixDim[1] * dimension], plane, sizeof(double) * planeDim[0] * planeDim[1]);
}

void multiply(double* a, int* aDim, double* b, int* bDim, double* result, int* rDim){
    setDimensions(bDim[0], aDim[1], 1, rDim);
    for(int i = 0; i < rDim[1]; i++){
        for(int j = 0; j < rDim[0]; j++){
            double el=0;
            for(int n = 0; n < aDim[0]; n++){
                el += getElement(a,aDim,i,n) * getElement(b,bDim,n,j);
            }
            setElement(result,rDim,i,j,el);
        }
    }
}

void blasMultiply(double* a, int* aDim, double* b, int* bDim, double* result, int* rDim){
    MatrixMultiplyBLAS(a, b, result, aDim[1], aDim[0], bDim[0], 'T', 'N');
    //dgemm('N','N', aDim[1], bDim[0], bDim[1], 1, a, aDim[1], b, bDim[1], 0, result, rDim[1]);
}

void mlMultiply(mxArray* a, int* aDim, mxArray* b, int* bDim, mxArray* result, int* rDim){
    int nlhs = 1;
    int nrhs = 2;
    int aDims[2] = {aDim[1], aDim[0]};
    int bDims[2] = {bDim[1], bDim[0]};
    int rDims[2] = {aDim[1], bDim[0]};

    mxArray *prhs[2];
    prhs[0] = a;
    prhs[1] = b;

    mexCallMATLAB(0,NULL,1, &prhs[0], "disp");

    mxArray *plhs[1];
    plhs[0] = mxCreateNumericArray(2, rDims, mxDOUBLE_CLASS, mxREAL);

    mexCallMATLAB( nlhs, plhs,
                   nrhs, prhs,
                   "mtimes");

    result = plhs[0];
}

void printMatrix(double* matrix, int* dim){
    std::cout << "\n\nHeight: " << dim[1] << "\tWidth: " << dim[0] << "\n";
    if (dim[2] > 1){
        for(int z = 0; z < dim[2]; z++) {
            std::cout << "Dimension: " << z + 1 << "\n";
            for (int i = 0; i < dim[1]; i++) {
                for (int j = 0; j < dim[0]; j++) {
                    std::cout << get3DElement(matrix, dim, i, j, z);
                    std::cout << ", ";
                }
                std::cout << "\n";
            }
        }
    }else {
        for (int i = 0; i < dim[1]; i++) {
            for (int j = 0; j < dim[0]; j++) {
                std::cout << getElement(matrix, dim, i, j);
                std::cout << ", ";
            }
            std::cout << "\n";
        }
    }
}

void transpose(double* matrix, int* matrixDimensions, double* newMx, int* newMxDimensions){
    setDimensions(matrixDimensions[1],matrixDimensions[0], 1, newMxDimensions);
    for(int i = 0; i < matrixDimensions[0]; i++) {
        for(int j = 0; j < matrixDimensions[1]; j++){
            setElement(newMx, newMxDimensions, i, j, getElement(matrix, matrixDimensions, j, i));
        }}
}

double getElementUsingMajor(double* elements, int majorWidth, int x, int y){
    return elements[majorWidth * x + y];
}

void colMajorToRowMajor(double* elements, int* dim, double *newElements){
    if(dim[2] > 1){
        for (int z = 0; z < dim[2]; z++){
            int* tempDim = new int[3];
            setDimensions(dim[0], dim[1], 1, tempDim);
            double* temp = new double[tempDim[0] * tempDim[1]];
            colMajorToRowMajor(&elements[z * tempDim[0] * tempDim[1]], tempDim, temp);
            std::memcpy(&newElements[z * tempDim[0] * tempDim[1]], temp, tempDim[0] * tempDim[1] * sizeof(double));
            delete [] temp;
            delete [] tempDim;
        }
    }else {
        int newIndex = 0;
        for (int i = 0; i < dim[1]; i++) {
            for (int j = 0; j < dim[0]; j++) {
                newElements[newIndex] = getElementUsingMajor(elements, dim[1], j, i);
                //mexPrintf("%d", (height * i + width * j));
                newIndex++;
            }
        }
    }
}

void printMex(double* elements, int* dim){
    mexPrintf("\n\nHeight: %d \tWidth: %d \tDepth: %d\n",dim[1],dim[0],dim[2]);
    if (dim[2] > 1){
        for(int z = 0; z < dim[2]; z++) {
            mexPrintf("Dimension: %d\n",z+1);
            for (int i = 0; i < dim[1]; i++) {
                for (int j = 0; j < dim[0]; j++) {
                    mexPrintf("%f, ", get3DElement(elements, dim, i, j, z));
                }
                mexPrintf("\n");
            }
        }
    }else {
        for (int i = 0; i < dim[1]; i++) {
            for (int j = 0; j < dim[0]; j++) {
                mexPrintf("%f, ", getElement(elements, dim, i, j));
            }
            mexPrintf("\n");
        }
    }
}
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int* reDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[2], reDim);
    double* reElements = new double[reDim[0] * reDim[1] * reDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[0]), reDim, reElements);


    //

    int* dxDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[1], mxGetDimensions(prhs[1])[0], 1, dxDim);
    double* dxElements = new double[dxDim[0] * dxDim[1] * dxDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[1]), dxDim, dxElements);


    //

    int* dyDim = new int[3];
    setDimensions(mxGetDimensions(prhs[2])[1], mxGetDimensions(prhs[2])[0], 1, dyDim);
    double* dyElements = new double[dyDim[0] * dyDim[1] * dyDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[2]), dyDim, dyElements);

    //


    int* dzDim = new int[3];
    setDimensions(mxGetDimensions(prhs[3])[1], mxGetDimensions(prhs[3])[0], 1, dzDim);
    double* dzElements = new double[dzDim[0] * dzDim[1] * dzDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[3]), dzDim, dzElements);


    //

    int* ccDim = new int[3];
    setDimensions(reDim[1], dxDim[0], 1, ccDim);

    nlhs = 1;
    int ndim = 3, dims[3] = {ccDim[1], ccDim[0], ccDim[2]};
    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    double* cc = mxGetPr(plhs[0]);

    multiply(getPlane(reElements,reDim, 1), reDim, dxElements,dxDim, cc, ccDim);
    printMex(cc, ccDim);
    blasMultiply(getPlane(mxGetPr(prhs[0]),reDim, 1), reDim, mxGetPr(prhs[1]),dxDim, cc, ccDim);
    printMex(cc, ccDim);
//    IP3d(reElements, reDim,
//         dxElements, dxDim,
//         dyElements, dyDim,
//         dzElements, dzDim,
//         cc, ccDim);

    delete [] reElements;
    delete [] dxElements;
    delete [] dyElements;
    delete [] dzElements;

    delete [] ccDim;
    delete [] reDim;
    delete [] dxDim;
    delete [] dyDim;
    delete [] dzDim;
}