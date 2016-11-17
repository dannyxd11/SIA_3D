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
    plane->elements = new double[plane->width * plane->height];
    std::memcpy(plane->elements, &matrix->elements[matrix->width * matrix->height * depth],
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

void IPSP3d(Matrix* re, Matrix* v1, Matrix* v2, Matrix* v3, Matrix* cc){
    int n1 = v1->width;
    int l3 = v3->height;

    cc->height = cc->depth = 1;
    cc->width = n1;
    cc->elements = new double[cc->width];

    for(int i = 0; i < n1; i++){
        cc->elements[i] = 0;
        for(int j = 0; j < l3; j++){
            Matrix* calcMatrix = new Matrix();
            Matrix* aMatrix = new Matrix();

            Matrix* v1Trans = new Matrix();
            Matrix* a1 = new Matrix();
            Matrix* b1 = new Matrix();

            transpose(v1, v1Trans);
            getRow(v1Trans,i, a1);
            getPlane(re,j, b1);
            multiply(a1,b1,aMatrix);
        //    multiply(getRow(transpose(v1),i),getPlane(re,j), aMatrix);

            Matrix* b2 = new Matrix();
            getCol(v2,i,b2);
            //multiply(aMatrix,getCol(v2,i), calcMatrix);

            multiply(aMatrix,b2,calcMatrix);

            cc->elements[i] += calcMatrix->elements[0] * getElement(v3,j,i);

            //cc->elements[i] += multiply(multiply(getRow(transpose(v1),i),getPlane(re,j), calcMatrix),getCol(v2,i), calcMatrix).elements[0] * getElement(v3,j,i);
            std::cout << cc->elements[i] << std::endl;

            delete v1Trans;
            delete a1;
            delete b1;
            delete b2;
            delete calcMatrix;
            delete aMatrix;
        }
        std::cout << cc->elements[i] << "\n";
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
    IPSP3d(re,dx,dy,dz,cc);
    printMatrix(cc);

    delete re;
    delete dx;
    delete dy;
    delete dz;
    delete cc;
    return 0;
}

