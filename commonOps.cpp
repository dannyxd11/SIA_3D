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

