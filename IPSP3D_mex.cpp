#include "mex.h"
#include "IPSP3D.cpp"
using namespace std;


double getElement(double* elements, int majorWidth, int x, int y){
    return elements[majorWidth * x + y];
}

double* colMajorToRowMajor(double* elements, int width, int height){
    double* newElements = (double*)malloc(width * height * sizeof(double));
    int newIndex = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            newElements[newIndex] = getElement(elements, height, j, i);
            //mexPrintf("%d", (height * i + width * j));
            newIndex++;
        }
    }
    free(elements);
    return newElements;
}

double* colMajorToRowMajor3D(double* elements, int width, int height, int depth){
    double* newElements = (double*)malloc(width * height * depth * sizeof(double));
    for (int z = 0; z < depth; z++){
        std::memcpy(&newElements[z * width * height],colMajorToRowMajor(&elements[z * width * height], width, height), width * height * sizeof(double));
      // newElements[z * width * height] = colMajorToRowMajor(&elements[z * width * height], width, height);
    }
    return newElements;
}

void mexPrintMatrix(Matrix matrix){

    if(matrix.depth > 1){
        for(int k = 0; k < matrix.depth; k++){
            for(int i = 0; i < matrix.height; i++){
                for(int j = 0; j < matrix.width; j++){
                    mexPrintf("%f,",get3DElement(matrix,i,j,k));
                }
                mexPrintf("\n");
            }
        }
    }else{
        for(int i = 0; i < matrix.height; i++){
            for(int j = 0; j < matrix.width; j++){
                mexPrintf("%f,",getElement(matrix, i,j));
            }
            mexPrintf("\n");
        }
   // std::cout << "\n\nHeight: " << matrix.height << "\tWidth: " << matrix.width << "\n";
 mexPrintf("\n");
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    //Matrix re, dx, dy, dz;

    int height = mxGetDimensions(prhs[0])[0];
    int width = mxGetDimensions(prhs[0])[1];
    int depth = mxGetDimensions(prhs[0])[2];

    //re.elements = colMajorToRowMajor3D(mxGetPr(prhs[0]), re.width, re.height, re.depth);
    Matrix re = new Matrix(width, height, depth,colMajorToRowMajor3D(mxGetPr(prhs[0]), width, height, depth));

    //re.elements = mxGetPr(prhs[0]);
    //mexPrintMatrix(re);

    height = mxGetDimensions(prhs[1])[0];
    width = mxGetDimensions(prhs[1])[1];
    Matrix dx = new Matrix(width, height, colMajorToRowMajor(mxGetPr(prhs[1]), width, height));

    dy.height = mxGetDimensions(prhs[2])[0];
    dy.width = mxGetDimensions(prhs[2])[1];
    dy.elements = colMajorToRowMajor(mxGetPr(prhs[2]), dy.width, dy.height);


    dz.height = mxGetDimensions(prhs[3])[0];
    dz.width = mxGetDimensions(prhs[3])[1];
    dz.elements = colMajorToRowMajor(mxGetPr(prhs[3]), dz.width, dz.height);
    dx.depth = dy.depth = dz.depth = 1;

    Matrix cc = IPSP3d(re, dx, dy, dz);

    nlhs = 1;
    plhs[0] = mxCreateDoubleMatrix(cc.height, cc.width, mxREAL);
    memcpy(mxGetPr(plhs[0]), cc.elements, cc.height * cc.width * sizeof(double));
    delete re.elements, dx.elements, dy.elements, dz.elements;
}