#ifdef __CUDACC__
#define CUDA_HOST_DEV __host__ __device__
#else
#define CUDA_HOST_DEV
#endif


#include "h_Matrix.h"
#include <cstddef>

CUDA_HOST_DEV h_Matrix::h_Matrix() : height(1), width(1), depth(1) {elements = 0; devElements = 0; preventFree = 0;};

CUDA_HOST_DEV h_Matrix::h_Matrix(int height, int width, int depth) : height(height), width(width), depth(depth) {
    elements = new double[height*width*depth]();
    devElements = 0;
    preventFree = 0;
}

CUDA_HOST_DEV h_Matrix::h_Matrix(int height, int width, int depth, int preventFree) : height(height), width(width), depth(depth), preventFree(preventFree){
    elements = new double[height*width*depth]();
    devElements = 0;
}

CUDA_HOST_DEV h_Matrix::h_Matrix(double* elements, int height, int width, int depth) : height(height), width(width), depth(depth), elements(elements) {devElements = 0; preventFree = 0;};

CUDA_HOST_DEV h_Matrix::h_Matrix(double* elements, int height, int width, int depth, int preventFree) : height(height), width(width), depth(depth), elements(elements), preventFree(preventFree) {devElements = 0;};

CUDA_HOST_DEV int h_Matrix::numel(){
    return height * width * depth;
}

CUDA_HOST_DEV double* h_Matrix::getColDouble(int i){
    return &elements[height * i];
}
CUDA_HOST_DEV double* h_Matrix::getElement(int i, int j){
    return &elements[i * height + j];//    if(elements != NULL){
}

CUDA_HOST_DEV void h_Matrix::setElement(int i, int j, double value){
    elements[i * height + j] = value;
}

CUDA_HOST_DEV void h_Matrix::setElement(int i, double value){
    elements[i] = value;
}

CUDA_HOST_DEV double* h_Matrix::getElement(int i){
    return &elements[i];
}

CUDA_HOST_DEV h_Matrix h_Matrix::getCol(int i){
    h_Matrix newMatrix(getColDouble(i),height, 1, 1, -1);
    return newMatrix;
}

CUDA_HOST_DEV h_Matrix h_Matrix::getPlane(int i){
    h_Matrix newMatrix(&elements[height * width * i], height, width, 1, -1);
    return newMatrix;
}

CUDA_HOST_DEV void h_Matrix::addDoubleElementwise(double val){
	for(int i = 0; i < numel(); i++){
		elements[i] += val;
	}
}

CUDA_HOST_DEV void h_Matrix::multDoubleElementwise(double val){
	for(int i = 0; i < numel(); i++){
		elements[i] *= val;
	}
}

CUDA_HOST_DEV h_Matrix::~h_Matrix(){
	if(devElements){
		//delete [] devElements;
		if(devElements != 0){		
			cudaFree(devElements);
		}
	}
	if(elements && preventFree == 0){ //  && *elements != NULL){
		if(elements != 0){
			delete [] elements;
		}
	}
}

//todo sort this out..
CUDA_HOST_DEV void h_Matrix::multiply(h_Matrix* a, h_Matrix* b, h_Matrix* result){
    for(int j = 0; j < result->width; j++){
        for(int i = 0; i < result->height; i++){
            double el=0;
            for(int n = 0; n < a->width; n++){
            	el += a->getElement(n,i)[0] * b->getElement(j,n)[0];
            }
            result->setElement(j,i,el);
        }
    }
}
