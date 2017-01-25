#ifndef COMMON_OPS
#define COMMON_OPS

#include "MyBLAS.h"
#include <stdlib.h>
#include <iostream>

void setDimensions(int height, int width, int depth, int* dim){
    dim[0] = height;
    dim[1] = width;
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

void matrixScalarMultiplication(double* matrixA, int* dim, double scalar){
    for (int i = 0; i < dim[0] * dim[1]; i++){
        matrixA[i] = matrixA[i] * scalar;
    }
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

double* getPlane(double* matrix, int* matrixDimensions, int depth) {
    return matrix + (matrixDimensions[0] * matrixDimensions[1] * depth);
}

void setPlane(double* plane, int* planeDim, double* matrix3d, int* matrixDim, int dimension){
    std::memcpy(&matrix3d[matrixDim[0] * matrixDim[1] * dimension], plane, sizeof(double) * planeDim[0] * planeDim[1]);
}

void multiply(double* a, int* aDim, double* b, int* bDim, double* result, int* rDim){
    setDimensions(aDim[0], bDim[1], 1, rDim);
    for(int j = 0; j < rDim[1]; j++){
        for(int i = 0; i < rDim[0]; i++){
            double el=0;
            for(int n = 0; n < aDim[1]; n++){
                el += getElement(a,aDim,n,i) * getElement(b,bDim,j,n);
            }
            setElement(result,rDim,j,i,el);
        }
    }
}

void blasMultiply(double* a, int* aDim, double* b, int* bDim, double* result, int* rDim){
    MatrixMultiplyBLAS(a, b, result, aDim[0], aDim[1], rDim[1], 'N', 'N');
}

void printMatrix(double* matrix, int* dim){
    std::cout << "\n\nHeight: " << dim[0] << "\tWidth: " << dim[1] << "\n";
    if (dim[2] > 1){
        for(int z = 0; z < dim[2]; z++) {
            std::cout << "Dimension: " << z + 1 << "\n";
            for (int i = 0; i < dim[1]; i++) {
                for (int j = 0; j < dim[0]; j++) {
                    std::cout << get3DElement(matrix, dim, j, i, z);
                    std::cout << ", ";
                }
                std::cout << "\n";
            }
        }
    }else {
        for (int i = 0; i < dim[1]; i++) {
            for (int j = 0; j < dim[0]; j++) {
                std::cout << getElement(matrix, dim, j, i);
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

void getRow(double* matrix, int* matrixDim, int row, double* rowMatrix, int* rowDim){
    setDimensions(1, matrixDim[1], 1, rowDim);
    for(int i = 0; i < matrixDim[1]; i++){
        setElement(rowMatrix, rowDim, i, 0, getElement(matrix, matrixDim, i, row));
    }
}

double* getCol(double* matrix, int* matrixDim, int col){
    return &matrix[matrixDim[0] * col];
}


#endif



