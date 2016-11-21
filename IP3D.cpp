#include <iostream>
#include <stdlib.h>
#include <cstring>


void setDimensions(int width, int height, int depth, int* dim){

    dim[0] = width;
    dim[1] = height;
    dim[2] = depth;
}

void setDimensions(int* newDim, int* dim){
    setDimensions(newDim[0], newDim[1], newDim[2], dim);
}

//todo Assertion
//void elementMultiplication(Matrix* matrixA, Matrix* matrixB, Matrix* results){
//    results->height = matrixA->height;
//    results->width = matrixA->width;
//    results->depth = 1;
//    results->elements = new double[matrixA->width * matrixA->height];
//    for (int i = 0; i < matrixA->width * matrixA->height; i++){
//        results->elements[i] = matrixA->elements[i] * matrixB->elements[i];
//    }
//}
//
void elementAddition(double* matrixA, int* aDim, double* matrixB, int* bDim, double* results, int* rDim){
    setDimensions(aDim[0], aDim[1], 1, rDim);
    for (int i = 0; i < aDim[0] * aDim[1]; i++){
        results[i] = results[i] + matrixA[i] + matrixB[i];
    }
}
//
void matrixScalarMultiplication(double* matrixA, int* dim, double scalar, double* results, int* resultsDim){
    setDimensions(dim, resultsDim);
    //results = new double[resultsDim[0] * resultsDim[1]];
    for (int i = 0; i < resultsDim[0] * resultsDim[1]; i++){
        results[i] = matrixA[i] * scalar;
    }
}
//

double getElement(double* matrix, int* matrixDimensions, int row, int col){
//    std::cout << row << "\t" << matrixDimensions[0]<< "\t" << col << "\t" << row*matrixDimensions[0]+col << std::endl;
//    std::cout << matrix[0];
//    std::cout << "Here";
    return matrix[row*matrixDimensions[0]+col];
}

void setElement(double matrix[], int* matrixDimensions, int row, int col, double value){
    matrix[row*matrixDimensions[0]+col] = value;
    // std::cout <<  matrix[row*matrixDimensions[0]+col] << std::endl;
}

//
double get3DElement(double* matrix, int* dim, int row, int col, int depth){
    return matrix[depth*dim[0]*dim[1]+row*dim[0]+col];
}
//
//void set3DElement(Matrix* matrix, int row, int col, int depth, double value){
//    matrix->elements[depth*matrix->height*matrix->width+row*matrix->width+col] = value;
//}
//
//void getRow(Matrix* matrix, int row, Matrix* rowMatrix){
//    rowMatrix->elements = new double[matrix->width];
//    rowMatrix->height = 1;
//    rowMatrix->width = matrix->width;
//
//    for(int i = 0; i < matrix->width; i++){
//        setElement(rowMatrix,0,i,getElement(matrix,row,i));
//    }
//}
//
//void getCol(Matrix* matrix, int col, Matrix* colMatrix){
//    //Matrix* colMatrix = new Matrix();
//    colMatrix->elements = new double[matrix->height];
//    colMatrix->width = 1;
//    colMatrix->height = matrix->height;
//    colMatrix->depth = 1;
//    for(int i = 0; i < matrix->height; i++){
//        setElement(colMatrix,i,0,getElement(matrix,i,col));
//    }
//}
//
double* getPlane(double* matrix, int* matrixDimensions, int depth) {
//    plane->elements = new double[plane->width * plane->height];
//    std::memcpy(plane->elements, &matrix->elements[matrix->width * matrix->height * depth], sizeof(double) * plane->height * plane->width);
    return matrix + (matrixDimensions[0] * matrixDimensions[1] * depth);
}
//
void setPlane(double* plane, int* planeDim, double* matrix3d, int* matrixDim, int dimension){
    std::memcpy(&matrix3d[matrixDim[0] * matrixDim[1] * dimension], plane, sizeof(double) * planeDim[0] * planeDim[1]);
}
//

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

void transpose(double* matrix, int* matrixDimensions, double* newMx, int* newMxDimensions){
    //newMx = new double[matrixDimensions[0] * matrixDimensions[1]];
    newMxDimensions[0] = matrixDimensions[1];
    newMxDimensions[1] = matrixDimensions[0];
    newMxDimensions[2] = 0;
    for(int i = 0; i < matrixDimensions[0]; i++) {
        for(int j = 0; j < matrixDimensions[1]; j++){
            setElement(newMx, newMxDimensions, i, j, getElement(matrix, matrixDimensions, j, i));
        }}
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

void IP3d(double* re, int* reDimensions, double* v1, int* v1Dimensions, double* v2, int* v2Dimensions, double* v3, int* v3Dimensions, double* cc, int *ccDimensions){
    for(int i = 0; i < ccDimensions[0] * ccDimensions[1] * ccDimensions[2]; i++){
        cc[i] = 0;
    }




    double *v1Trans = new double[v1Dimensions[0] * v1Dimensions[1]];
    int* v1TransDimensions = new int[3];
    transpose(v1, v1Dimensions, v1Trans, v1TransDimensions);

    //Declaration & Initialization of variables
    double *b1;
    int* b1Dimensions = new int[3];

    double* aMatrix = new double[b1Dimensions[0] * v1TransDimensions[1]];
    int* aMatrixDimensions = new int[3];

    double* calcMatrix = new double[v2Dimensions[0] * aMatrixDimensions[1]];
    int* calcMatrixDimensions = new int[3];

    double* eleMultMatrix = new double[v2Dimensions[0] * aMatrixDimensions[1]];
    int* eleMultMatrixDimensions = new int[3];

    double* eleAddMatrix;
    int* eleAddMatrixDimensions = new int[3];

    for(int m3 = 0; m3 < v3Dimensions[0]; m3++){
        double* ccPlane  = new double[ccDimensions[0] * ccDimensions[1]];
        for (int i = 0; i < ccDimensions[0] * ccDimensions[1]; i++){
            ccPlane[i] = 0;
        }
        int* ccPlaneDimensions = new int[3];
        ccPlaneDimensions[0] = ccDimensions[0];
        ccPlaneDimensions[1] = ccDimensions[1];
        ccPlaneDimensions[2] = 1;

        for(int zk = 0; zk < v3Dimensions[1]; zk++){
//            double *b1;
//            int* b1Dimensions = new int[3];
            b1 = getPlane(re, reDimensions, zk);
            setDimensions(reDimensions[0], reDimensions[1], 1, b1Dimensions);

//            double* aMatrix = new double[b1Dimensions[0] * v1TransDimensions[1]];
//            int* aMatrixDimensions = new int[3];

            multiply(v1Trans, v1TransDimensions, b1, b1Dimensions, aMatrix, aMatrixDimensions);



//            double* calcMatrix = new double[v2Dimensions[0] * aMatrixDimensions[1]];
//            int* calcMatrixDimensions = new int[3];
            multiply(aMatrix, aMatrixDimensions, v2, v2Dimensions, calcMatrix, calcMatrixDimensions);


//            printMatrix(calcMatrix,calcMatrixDimensions);
//            delete aMatrix;
//
//            double* eleMultMatrix = new double[v2Dimensions[0] * aMatrixDimensions[1]];
//            int* eleMultMatrixDimensions = new int[3];
            matrixScalarMultiplication(calcMatrix, calcMatrixDimensions, getElement(v3,v3Dimensions,zk,m3), eleMultMatrix, eleMultMatrixDimensions);
//
//            printMatrix(eleMultMatrix, eleMultMatrixDimensions);

//            double* eleAddMatrix; // = new double[ccPlaneDimensions[0] * ccPlaneDimensions[1]];
//            int* eleAddMatrixDimensions = new int[3];
            eleAddMatrix = getPlane(cc, ccDimensions, m3);
            setDimensions(ccDimensions[0], ccDimensions[1], 1, eleAddMatrixDimensions);


//            printMatrix(eleAddMatrix);
            elementAddition(eleAddMatrix, eleAddMatrixDimensions, eleMultMatrix, eleMultMatrixDimensions, ccPlane, ccPlaneDimensions);

//            delete [] aMatrix;
//            delete [] aMatrixDimensions;
//            //delete [] b1;
//            delete [] b1Dimensions;
//            delete [] eleMultMatrix;
//            delete [] eleMultMatrixDimensions;
//            //delete [] eleAddMatrix;
//            delete [] eleAddMatrixDimensions;
//            delete [] calcMatrix;
//            delete [] calcMatrixDimensions;


        }
        //  printMatrix(ccPlane, ccPlaneDimensions);
        setPlane(ccPlane, ccPlaneDimensions, cc, ccDimensions, m3);
        delete [] ccPlane;
        delete [] ccPlaneDimensions;
    }

    delete [] aMatrix;
    delete [] aMatrixDimensions;
    //delete [] b1;
    delete [] b1Dimensions;
    delete [] eleMultMatrix;
    delete [] eleMultMatrixDimensions;
    //delete [] eleAddMatrix;
    delete [] eleAddMatrixDimensions;
    delete [] calcMatrix;
    delete [] calcMatrixDimensions;

//    delete [] aMatrix;
//            delete eleMultMatrix;
//            delete eleAddMatrix;
    delete [] v1Trans;
    delete [] v1TransDimensions;
//    delete [] b1;
//    delete [] calcMatrix;
}

int main() {

    double vxElements[]  = {0.35355,0.49759,0.49039,0.47847,0.46194,0.44096,0.41573,0.38651,0.35355,0.3172,0.27779,0.2357,0.19134,0.14514,0.097545,0.049009,0,0.049009,0.097545,0.14514,0.19134,0.2357,0.27779,0.3172,0.35355,0.38651,0.41573,0.44096,0.46194,0.47847,0.49039,0.49759,1,0,0,0,0,0,0,0,
                            0.35355,0.47847,0.41573,0.3172,0.19134,0.049009,-0.097545,-0.2357,-0.35355,-0.44096,-0.49039,-0.49759,-0.46194,-0.38651,-0.27779,-0.14514,0,0.14514,0.27779,0.38651,0.46194,0.49759,0.49039,0.44096,0.35355,0.2357,0.097545,-0.049009,-0.19134,-0.3172,-0.41573,-0.47847,0,1,0,0,0,0,0,0,
                            0.35355,0.44096,0.27779,0.049009,-0.19134,-0.38651,-0.49039,-0.47847,-0.35355,-0.14514,0.097545,0.3172,0.46194,0.49759,0.41573,0.2357,0,0.2357,0.41573,0.49759,0.46194,0.3172,0.097545,-0.14514,-0.35355,-0.47847,-0.49039,-0.38651,-0.19134,0.049009,0.27779,0.44096,0,0,1,0,0,0,0,0,
                            0.35355,0.38651,0.097545,-0.2357,-0.46194,-0.47847,-0.27779,0.049009,0.35355,0.49759,0.41573,0.14514,-0.19134,-0.44096,-0.49039,-0.3172,0,0.3172,0.49039,0.44096,0.19134,-0.14514,-0.41573,-0.49759,-0.35355,-0.049009,0.27779,0.47847,0.46194,0.2357,-0.097545,-0.38651,0,0,0,1,0,0,0,0,
                            0.35355,0.3172,-0.097545,-0.44096,-0.46194,-0.14514,0.27779,0.49759,0.35355,-0.049009,-0.41573,-0.47847,-0.19134,0.2357,0.49039,0.38651,0,0.38651,0.49039,0.2357,-0.19134,-0.47847,-0.41573,-0.049009,0.35355,0.49759,0.27779,-0.14514,-0.46194,-0.44096,-0.097545,0.3172,0,0,0,0,1,0,0,0,
                            0.35355,0.2357,-0.27779,-0.49759,-0.19134,0.3172,0.49039,0.14514,-0.35355,-0.47847,-0.097545,0.38651,0.46194,0.049009,-0.41573,-0.44096,0,0.44096,0.41573,-0.049009,-0.46194,-0.38651,0.097545,0.47847,0.35355,-0.14514,-0.49039,-0.3172,0.19134,0.49759,0.27779,-0.2357,0,0,0,0,0,1,0,0,
                            0.35355,0.14514,-0.41573,-0.38651,0.19134,0.49759,0.097545,-0.44096,-0.35355,0.2357,0.49039,0.049009,-0.46194,-0.3172,0.27779,0.47847,0,0.47847,0.27779,-0.3172,-0.46194,0.049009,0.49039,0.2357,-0.35355,-0.44096,0.097545,0.49759,0.19134,-0.38651,-0.41573,0.14514,0,0,0,0,0,0,1,0,
                            0.35355,0.049009,-0.49039,-0.14514,0.46194,0.2357,-0.41573,-0.3172,0.35355,0.38651,-0.27779,-0.44096,0.19134,0.47847,-0.097545,-0.49759,0,0.49759,0.097545,-0.47847,-0.19134,0.44096,0.27779,-0.38651,-0.35355,0.3172,0.41573,-0.2357,-0.46194,0.14514,0.49039,-0.049009,0,0,0,0,0,0,0,1
    };

    double vyElements[]  = {0.35355,0.49759,0.49039,0.47847,0.46194,0.44096,0.41573,0.38651,0.35355,0.3172,0.27779,0.2357,0.19134,0.14514,0.097545,0.049009,0,0.049009,0.097545,0.14514,0.19134,0.2357,0.27779,0.3172,0.35355,0.38651,0.41573,0.44096,0.46194,0.47847,0.49039,0.49759,1,0,0,0,0,0,0,0,
                            0.35355,0.47847,0.41573,0.3172,0.19134,0.049009,-0.097545,-0.2357,-0.35355,-0.44096,-0.49039,-0.49759,-0.46194,-0.38651,-0.27779,-0.14514,0,0.14514,0.27779,0.38651,0.46194,0.49759,0.49039,0.44096,0.35355,0.2357,0.097545,-0.049009,-0.19134,-0.3172,-0.41573,-0.47847,0,1,0,0,0,0,0,0,
                            0.35355,0.44096,0.27779,0.049009,-0.19134,-0.38651,-0.49039,-0.47847,-0.35355,-0.14514,0.097545,0.3172,0.46194,0.49759,0.41573,0.2357,0,0.2357,0.41573,0.49759,0.46194,0.3172,0.097545,-0.14514,-0.35355,-0.47847,-0.49039,-0.38651,-0.19134,0.049009,0.27779,0.44096,0,0,1,0,0,0,0,0,
                            0.35355,0.38651,0.097545,-0.2357,-0.46194,-0.47847,-0.27779,0.049009,0.35355,0.49759,0.41573,0.14514,-0.19134,-0.44096,-0.49039,-0.3172,0,0.3172,0.49039,0.44096,0.19134,-0.14514,-0.41573,-0.49759,-0.35355,-0.049009,0.27779,0.47847,0.46194,0.2357,-0.097545,-0.38651,0,0,0,1,0,0,0,0,
                            0.35355,0.3172,-0.097545,-0.44096,-0.46194,-0.14514,0.27779,0.49759,0.35355,-0.049009,-0.41573,-0.47847,-0.19134,0.2357,0.49039,0.38651,0,0.38651,0.49039,0.2357,-0.19134,-0.47847,-0.41573,-0.049009,0.35355,0.49759,0.27779,-0.14514,-0.46194,-0.44096,-0.097545,0.3172,0,0,0,0,1,0,0,0,
                            0.35355,0.2357,-0.27779,-0.49759,-0.19134,0.3172,0.49039,0.14514,-0.35355,-0.47847,-0.097545,0.38651,0.46194,0.049009,-0.41573,-0.44096,0,0.44096,0.41573,-0.049009,-0.46194,-0.38651,0.097545,0.47847,0.35355,-0.14514,-0.49039,-0.3172,0.19134,0.49759,0.27779,-0.2357,0,0,0,0,0,1,0,0,
                            0.35355,0.14514,-0.41573,-0.38651,0.19134,0.49759,0.097545,-0.44096,-0.35355,0.2357,0.49039,0.049009,-0.46194,-0.3172,0.27779,0.47847,0,0.47847,0.27779,-0.3172,-0.46194,0.049009,0.49039,0.2357,-0.35355,-0.44096,0.097545,0.49759,0.19134,-0.38651,-0.41573,0.14514,0,0,0,0,0,0,1,0,
                            0.35355,0.049009,-0.49039,-0.14514,0.46194,0.2357,-0.41573,-0.3172,0.35355,0.38651,-0.27779,-0.44096,0.19134,0.47847,-0.097545,-0.49759,0,0.49759,0.097545,-0.47847,-0.19134,0.44096,0.27779,-0.38651,-0.35355,0.3172,0.41573,-0.2357,-0.46194,0.14514,0.49039,-0.049009,0,0,0,0,0,0,0,1
    };

    double vzElements[] = {0.57735,0.78868,0.70711,0.57735,0.40825,0.21132,0,0.21132,0.40825,0.57735,0.70711,0.78868,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0.57735,0.57735,4.9996e-17,-0.57735,-0.8165,-0.57735,0,0.57735,0.8165,0.57735,9.9992e-17,-0.57735,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0.57735,0.21132,-0.70711,-0.57735,0.40825,0.78868,0,0.78868,0.40825,-0.57735,-0.70711,0.21132,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    double reElements[] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6};

    int dxDimensions[] = {40, 8, 1};
    int dyDimensions[] = {40, 8, 1};
    int dzDimensions[] = {40, 3, 1};
    int reDimensions[] = {8, 8, 3};

//    Matrix* dx = new Matrix(40, 8, &vxElements[0]);
//    Matrix* dy = new Matrix(40, 8, &vyElements[0]);
//    Matrix* dz = new Matrix(40, 3, &vzElements[0]);
//    Matrix* re = new Matrix(8, 8, 3, &reElements[0]);

//    Matrix* cc = new Matrix();

    double* ccElements = new double[dxDimensions[0] * dyDimensions[0] * dzDimensions[0]];
    int ccDimensions[] = {dxDimensions[0], dyDimensions[0], dzDimensions[0]};

    IP3d(reElements,reDimensions,vxElements,dxDimensions,vyElements,dyDimensions,vzElements,dzDimensions,ccElements,ccDimensions);
    //printMatrix(ccElements, ccDimensions);

    double* firstPlane;
    int dim[] = {40,40,1};
    firstPlane = getPlane(ccElements,ccDimensions, 2);
    printMatrix(firstPlane,dim);
//
//    Matrix* val = new Matrix();
//    val->elements = 0;
//    matrixScalarMultiplication(dy, 2, val);
//    printMatrix(val);


//    delete val;
//    delete re;
//
//    delete dy;
//    delete dx;
//    delete dz;
    delete [] ccElements;
    return 0;
}

