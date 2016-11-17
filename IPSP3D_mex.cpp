#include "mex.h"
#include "IPSP3D.cpp"
using namespace std;


double getElement(double* elements, int majorWidth, int x, int y){
    return elements[majorWidth * x + y];
}

void colMajorToRowMajor(double* elements, int width, int height, double *newElements){
    int newIndex = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            newElements[newIndex] = getElement(elements, height, j, i);
            //mexPrintf("%d", (height * i + width * j));
            newIndex++;
        }
    }
    free(elements);
    //return newElements;
}

void colMajorToRowMajor3D(double* elements, int width, int height, int depth, double *newElements){

    for (int z = 0; z < depth; z++){
        double* temp = (double*)malloc(width * height * sizeof(double));
        colMajorToRowMajor(&elements[z * width * height], width, height, temp);
        std::memcpy(&newElements[z * width * height], temp, width * height * sizeof(double));
        free(temp);
      // newElements[z * width * height] = colMajorToRowMajor(&elements[z * width * height], width, height);
    }
    //return newElements;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    mexPrintf("No Errors: 1");
    int height = mxGetDimensions(prhs[0])[0];
    int width = mxGetDimensions(prhs[0])[1];
    int depth = mxGetDimensions(prhs[0])[2];
    double* reElements = (double*)malloc(width * height * depth * sizeof(double));
    mexPrintf("No Errors: 2");
    colMajorToRowMajor3D(mxGetPr(prhs[0]), width, height, depth, reElements);
    Matrix* re = new Matrix(width, height, depth, reElements);
    mexPrintf("No Errors: 3");

    //

    height = mxGetDimensions(prhs[1])[0];
    width = mxGetDimensions(prhs[1])[1];
    mexPrintf("No Errors: 4");
    double* dxElements = (double*)malloc(width * height * sizeof(double));
    mexPrintf("No Errors: 5");
    colMajorToRowMajor(mxGetPr(prhs[1]), width, height, dxElements);
    Matrix* dx = new Matrix(width, height, dxElements);
    mexPrintf("No Errors: 6");


    //

    height = mxGetDimensions(prhs[2])[0];
    width = mxGetDimensions(prhs[2])[1];
    double* dyElements = (double*)malloc(width * height * sizeof(double));
    colMajorToRowMajor(mxGetPr(prhs[2]), width, height, dyElements);
    Matrix* dy = new Matrix(width, height, dyElements);


    //

    height = mxGetDimensions(prhs[3])[0];
    width = mxGetDimensions(prhs[3])[1];
    double* dzElements = (double*)malloc(width * height * sizeof(double));
    colMajorToRowMajor(mxGetPr(prhs[3]), width, height, dzElements);
    Matrix* dz = new Matrix(width, height, dzElements);


    //
    mexPrintf("No Errors: 7");
    Matrix* cc = new Matrix();
    mexPrintf("No Errors: 8");
    IPSP3d(re, dx, dy, dz, cc);
    mexPrintf("No Errors: 9");

    nlhs = 1;
    plhs[0] = mxCreateDoubleMatrix(cc->height, cc->width, mxREAL);
    memcpy(mxGetPr(plhs[0]), cc->elements, cc->height * cc->width * sizeof(double));
    delete re;
    delete dx;
    delete dy;
    delete dz;
    delete cc;

    free(reElements);
    free(dxElements);
    free(dyElements);
    free(dzElements);
}