#ifndef createDict
#define createDict

#include "h_Matrix.h"
#include <math.h>

using namespace cv;

void fillArray(double* array, int size){
	for(int i = 1; i <= size; i++ ){
		array[i-1] = i;
	}
}

void createCosineDict(h_Matrix* cosDict, int blockSize, int nFrequencies, int redundancy){
   // cosDict = new h_Matrix(blockSize, nFrequencies, 1);

    h_Matrix* kM =  new h_Matrix(blockSize, 1, 1);
    h_Matrix* nM =  new h_Matrix(1, nFrequencies, 1);

    fillArray(kM->elements, blockSize);
    fillArray(nM->elements, nFrequencies);



    int N = redundancy * blockSize;

    kM->h_Matrix::multDoubleElementwise(2);
    kM->h_Matrix::addDoubleElementwise(-1);
    nM->h_Matrix::addDoubleElementwise(-1);
    nM->h_Matrix::multDoubleElementwise(pow(2*N,-1));

    h_Matrix::multiply(kM, nM, cosDict);

    for(int i = 0; i < cosDict->numel(); i++){
    	cosDict->elements[i] = cos(M_PI * cosDict->elements[i]);
    }

    delete kM;
    delete nM;
}

void createSineDict(h_Matrix* sinDict, int blockSize, int nFrequencies, int redundancy){
    //sinDict = new h_Matrix(blockSize, nFrequencies, 1);

    h_Matrix* kM =  new h_Matrix(blockSize, 1, 1);
    h_Matrix* nM =  new h_Matrix(1, nFrequencies, 1);

    fillArray(kM->elements, blockSize);
    fillArray(nM->elements, nFrequencies);

    int N = redundancy * blockSize;

    kM->h_Matrix::multDoubleElementwise(2);
    kM->h_Matrix::addDoubleElementwise(-1);
    nM->h_Matrix::multDoubleElementwise(pow(2*N,-1));

    h_Matrix::multiply(kM, nM, sinDict);

    for(int i = 0; i < sinDict->numel(); i++){
    	sinDict->elements[i] = sin(M_PI * sinDict->elements[i]);
    }

    delete kM;
    delete nM;
}

h_Matrix* createIdentityMatrix(int height, int width){
    h_Matrix* identity =  new h_Matrix(height, width, 1);

    for(int i = 0; i < height && i < width; i++){
    	identity->setElement(i,i,1);
    }

    return identity;
}

void normaliseDict(h_Matrix* matrix){
	double nor = 0;
	for(int col = 0; col < matrix->width; col++){
		nor = 0;
		for(int row = 0; row < matrix->height; row++){
			nor += pow(matrix->getElement(col, row)[0], 2);
		}
		nor = sqrt(nor);
		if(nor > 1e-7){
			for(int row = 0; row < matrix->height; row++){
				matrix->setElement(col, row, matrix->getElement(col,row)[0] / nor);
			}
		}
	}
}

h_Matrix createStandardDict(){
    int blockWidth = 8;
    int XYdictSize = blockWidth * 2;
    int redundancy = 2;

    h_Matrix* cosDict = new h_Matrix(blockWidth, XYdictSize, 1);
    createCosineDict(cosDict, blockWidth, XYdictSize, redundancy);
    normaliseDict(cosDict);

    h_Matrix* sinDict = new h_Matrix(blockWidth, XYdictSize, 1);
    createSineDict(sinDict, blockWidth, XYdictSize, redundancy);
    normaliseDict(sinDict);

    h_Matrix* identityMatrix = createIdentityMatrix(blockWidth, blockWidth);

    h_Matrix matrix(blockWidth, XYdictSize * 2 + blockWidth, 1);
    std::copy(cosDict->elements, cosDict->elements + cosDict->numel(), matrix.elements);
    std::copy(sinDict->elements, sinDict->elements + sinDict->numel(), matrix.elements + cosDict->numel());
    std::copy(identityMatrix->elements, identityMatrix->elements + identityMatrix->numel(), matrix.elements + cosDict->numel() + sinDict->numel());

    delete cosDict;
    delete sinDict;
    delete identityMatrix;

    return matrix;
}

h_Matrix createDzDict(){
    int blockDepth = 3;
    int XYdictSize = blockDepth * 2;
    int redundancy = 2;

    h_Matrix* cosDict = new h_Matrix(blockDepth, XYdictSize, 1);
    createCosineDict(cosDict, blockDepth, XYdictSize, redundancy);
    normaliseDict(cosDict);

    h_Matrix* sinDict = new h_Matrix(blockDepth, XYdictSize, 1);
    createSineDict(sinDict, blockDepth, XYdictSize, redundancy);
    normaliseDict(sinDict);

    h_Matrix* identityMatrix = createIdentityMatrix(blockDepth, blockDepth);

    h_Matrix matrix(blockDepth, XYdictSize * 2 + blockDepth, 1);
    std::copy(cosDict->elements, cosDict->elements + cosDict->numel(), matrix.elements);
    std::copy(sinDict->elements, sinDict->elements + sinDict->numel(), matrix.elements + cosDict->numel());
    std::copy(identityMatrix->elements, identityMatrix->elements + identityMatrix->numel(), matrix.elements + cosDict->numel() + sinDict->numel());

    delete cosDict;
    delete sinDict;
    delete identityMatrix;

    return matrix;
}




#endif
