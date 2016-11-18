#include <iostream>
#include <stdlib.h>
#include <cstring>

//typedef struct{
//    int width;
//    int height;
//    int depth;
//    double* elements;
//} Matrix;

class Matrix{
public:
    int width;
    int height;
    int depth;
    double* elements;

    Matrix(){   };

    Matrix(int newWidth, int newHeight, double *newElements) : width(newWidth), height(newHeight), elements(newElements), depth(1){    }

    Matrix(int newWidth, int newHeight, int newDepth, double *newElements): width(newWidth), height(newHeight), depth(newDepth), elements(newElements){    }

    ~Matrix(){
        if (elements){
            delete [] elements;
            elements = 0;
        }
    }
};

//todo Assertion
void elementMultiplication(Matrix* matrixA, Matrix* matrixB, Matrix* results){
    results->elements = new double[sizeof(matrixA->elements)];
    for (int i = 0; i < sizeof(matrixA->elements); i++){
        results->elements[i] = matrixA->elements[i] * matrixB->elements[i];
    }
}

void elementAddition(Matrix* matrixA, Matrix* matrixB, Matrix* results){
    results->elements = new double[sizeof(matrixA->elements)];
    for (int i = 0; i < sizeof(matrixA->elements); i++){
        results->elements[i] = matrixA->elements[i] + matrixB->elements[i];
    }
}

void matrixSclaraMultiplication(Matrix* matrixA, double scalar, Matrix* results){
    results->elements = new double[sizeof(matrixA->elements)];
    for (int i = 0; i < sizeof(matrixA->elements); i++){
        results->elements[i] = matrixA->elements[i] * scalar;
    }
}

double getElement(Matrix* matrix, int row, int col){
    return matrix->elements[row*matrix->width+col];
}

void setElement(Matrix* matrix, int row, int col, double value){
    matrix->elements[row*matrix->width+col] = value;
}

double get3DElement(Matrix matrix, int row, int col, int depth){
    return matrix.elements[depth*matrix.height*matrix.width+row*matrix.width+col];
}

void set3DElement(Matrix* matrix, int row, int col, int depth, double value){
    matrix->elements[depth*matrix->height*matrix->width+row*matrix->width+col] = value;
}

void getRow(Matrix* matrix, int row, Matrix* rowMatrix){
    rowMatrix->elements = new double[matrix->width];
    rowMatrix->height = 1;
    rowMatrix->width = matrix->width;

    for(int i = 0; i < matrix->width; i++){
        setElement(rowMatrix,0,i,getElement(matrix,row,i));
    }
}

void getCol(Matrix* matrix, int col, Matrix* colMatrix){
    //Matrix* colMatrix = new Matrix();
    colMatrix->elements = new double[matrix->height];
    colMatrix->width = 1;
    colMatrix->height = matrix->height;

    for(int i = 0; i < matrix->height; i++){
        setElement(colMatrix,i,0,getElement(matrix,i,col));
    }
}

void getPlane(Matrix* matrix, int depth, Matrix* plane) {
    plane->width = matrix->width;
    plane->height = matrix->height;
    plane->depth = 1;
//    plane->elements = new double[plane->width * plane->height];
//    std::memcpy(plane->elements, &matrix->elements[matrix->width * matrix->height * depth], sizeof(double) * plane->height * plane->width);
    plane->elements = &matrix->elements[matrix->width * matrix->height * depth];
}

void setPlane(Matrix* plane, Matrix* matrix3d, int dimension){
    std::memcpy(&matrix3d->elements[matrix3d->height * matrix3d->width * dimension], plane->elements,
                sizeof(double) * plane->height * plane->width);
}

void multiply(Matrix* a, Matrix* b, Matrix* result){
    result->height = a->height;
    result->width = b->width;
    result->elements = new double[result->width * result->height];
    for(int i = 0; i < a->height; i++){
        for(int j = 0; j < b->width; j++){
            double el=0;
            for(int n = 0; n < a->width; n++){
                el += getElement(a,i,n) * getElement(b,n,j);
            }
            setElement(result,i,j,el);
        }
    }
}

void transpose(Matrix* matrix, Matrix* newMx){
    newMx->elements = new double[matrix->width * matrix->height];
    newMx->width = matrix->height;
    newMx->height = matrix->width;
    for(int i = 0; i < matrix->width; i++) {
        for(int j = 0; j < matrix->height; j++){
            setElement(newMx, i, j, getElement(matrix, j, i));
        }}
}

void printMatrix(Matrix* matrix){

    std::cout << "\n\nHeight: " << matrix->height << "\tWidth: " << matrix->width << "\n";
    for(int i = 0; i < matrix->height; i++){
        for(int j = 0; j < matrix->width; j++){
            std::cout << getElement(matrix, i,j);
            std::cout << ", ";
        }
        std::cout << "\n";
    }
}

void IP3d(Matrix* re, Matrix* v1, Matrix* v2, Matrix* v3, Matrix* cc){
    int Nx = v1->width;
    int Lx = v1->height;
    int Ny = v2->width;
    int Ly = v2->height;
    int Nz = v3->width;
    int Lz = v3->height;

    cc->height = Nx;
    cc->width = Ny;
    cc->depth = Nz;
    cc->elements = new double[v1->width * v2->width * v3->width];
    for(int i = 0; i < Nx * Ny * Nz; i++){
        cc->elements[i] = 0;
    }
    std::cout << Nx << "\t" << Lx << "\t"  << Ny << "\t" << Ly << "\t" << Nz << "\t" << Lz << std::endl;
    for(int m3 = 0; m3 < Nz; m3++){
        Matrix* ccPlane = new Matrix();
        for(int zk = 0; zk < Lz; zk++){
            Matrix* calcMatrix = new Matrix();
            Matrix* aMatrix = new Matrix();
            Matrix* v1Trans = new Matrix();
            Matrix* a1 = new Matrix();
            Matrix* b1 = new Matrix();

            transpose(v1, v1Trans);
            getPlane(re, zk, b1);
            multiply(v1Trans, b1, aMatrix);
            multiply(aMatrix, v2,calcMatrix);

            Matrix* eleMultMatrix = new Matrix();
            matrixSclaraMultiplication(calcMatrix, getElement(v3,zk,m3), eleMultMatrix);

            Matrix* eleAddMatrix = new Matrix();
            getPlane(cc, m3, eleAddMatrix);
            elementAddition(eleAddMatrix, eleMultMatrix, ccPlane);

            delete eleMultMatrix;
            delete v1Trans;
            delete a1;
            delete b1;
            delete calcMatrix;
            delete aMatrix;
        }
        setPlane(ccPlane, cc, m3);
        delete ccPlane;
    }
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

//    dx.elements = &dxElements[0];
//    dy.elements = &dyElements[0];
//    dz.elements = &dzElements[0];
//    re.elements = &reElements[0];

    Matrix* dx = new Matrix(40, 8, dxElements);
    Matrix* dy = new Matrix(40, 8, dyElements);
    Matrix* dz = new Matrix(40, 3, dzElements);
    Matrix* re = new Matrix(8, 8, 3, reElements);

    Matrix* cc = new Matrix();
    IP3d(re,dx,dy,dz,cc);

    delete re;
    delete dx;
    delete dy;
    delete dz;
    delete cc;
    return 0;
}

