#ifndef COMMON_OPS
#define COMMON_OPS

#include "MyBLAS.h"
#include <stdlib.h>
#include <iostream>
#include <cstring>

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

int getElement(int* matrix, int* matrixDimensions,  int col, int row){
    return matrix[col * matrixDimensions[0] + row];
}

void setElement(int matrix[], int* matrixDimensions, int col, int row, int value){
    matrix[col * matrixDimensions[0] + row] = value;
}

double get3DElement(double* matrix, int* dim, int row, int col, int depth){
    return matrix[depth * dim[0] * dim[1] + col * dim[0] + row];
}

void set3DElement(double* matrix, int* dim, int row, int col, int depth, double value){
    matrix[depth * dim[0] * dim[1] + col * dim[0] + row] = value;
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

void transpose(int* matrix, int* matrixDimensions, int* newMx, int* newMxDimensions){
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

int* getCol(int* matrix, int* matrixDim, int col){
    return &matrix[matrixDim[0] * col];
}

void reorthogonalize(double* Q, int* QDim, int zmax){


    int k = QDim[1];

    for(int z = 0; z < zmax; z++){

        double multResult1 = 0;
        int multResult1Dim[] = {1,1,1};

        for(int p = 0; p < k - 1; p++){
            double* qp = getCol(Q,QDim,p);
            double* qk = getCol(Q, QDim, k-1);

            MatrixMultiplyBLAS(getCol(Q,QDim,p), getCol(Q, QDim, k-1), &multResult1, QDim[0], 1, 1, 'T', 'N', 1, 0);


            for (int n = 0; n < QDim[0]; n++){
                setElement(Q, QDim, k-1, n, qk[n] - (multResult1 * qp[n]));
            }
        }
    }
}

void biorthogonalize(double* beta, int* betaDim, double* qk, int* qkDim, double* newAtom, int* newAtomDim, double nork){

    // beta = beta - Qk * (new_atom'*beta) / nork;
    // Width of second, height of first

    double* multResult = new double[betaDim[1] * newAtomDim[0]];
    int multResultDim[] = {newAtomDim[1],betaDim[1],1};

    MatrixMultiplyBLAS(newAtom, beta, multResult, newAtomDim[0], newAtomDim[1], betaDim[1], 'T', 'N', 1, 0);
    MatrixMultiplyBLAS(qk, multResult, beta, qkDim[0], qkDim[1], multResultDim[1], 'N', 'N', -1/nork, 1);

    delete [] multResult;
}


void kroneckerProduct(double* leftMatrix, int* leftMatrixDim, double* rightMatrix, int* rightMatrixDim, double* result, int* resultDim){

    int newHeight = leftMatrixDim[0] * rightMatrixDim[0];
    int newWidth = leftMatrixDim[1] * rightMatrixDim[1];

    //result = new double[newHeight * newWidth];
    setDimensions(newHeight, newWidth, 1, resultDim);




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

void orthogonalize(double* Q, int* QDim, double* newAtom, int* newAtomDim){
// Set K as width + 1 (Size is increasing)
    int k = QDim[1] + 1;
    int* origDim = new int[3];

    setDimensions(QDim, origDim);
    QDim[1] += 1;

    double multResult1 = 0;
    int multResult1Dim[] = {1,1,1};

    for(int p = 0; p < k - 1; p++){
        MatrixMultiplyBLAS(getCol(Q, origDim, p), newAtom, &multResult1, origDim[0], 1, 1, 'T', 'N', 1, 0);

        double* col = getCol(Q, QDim, p);

        for (int n = 0; n < QDim[0]; n++){
            setElement(Q, QDim, k-1, n, newAtom[n] - (multResult1) * col[n]);
        }

    }

    delete [] origDim;
}

#endif
