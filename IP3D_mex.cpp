#include "mex.h"
#include "IP3D_test.cpp"
using namespace std;


double getElementUsingMajor(double* elements, int majorWidth, int x, int y){
    return elements[majorWidth * x + y];
}

void colMajorToRowMajor(double* elements, int* dim, double *newElements){
    int newIndex = 0;
    mexPrintf("%d", dim[2]);
    if(dim[2] > 1){
        for (int z = 0; z < dim[2]; z++){
            int* tempDim = new int[3];
            setDimensions(dim[0], dim[1], 1, tempDim);
            double* temp = new double[tempDim[0] * tempDim[1]];
            colMajorToRowMajor(&elements[z * tempDim[0] * tempDim[1]], tempDim, temp);
            std::memcpy(&newElements[z * tempDim[0] * tempDim[1]], temp, tempDim[0] * tempDim[1] * sizeof(double));
            delete [] temp;
            delete [] tempDim;
        }
    }else {
        for (int i = 0; i < dim[1]; i++) {
            for (int j = 0; j < dim[2]; j++) {
                newElements[newIndex] = getElementUsingMajor(elements, dim[1], j, i);
                //mexPrintf("%d", (height * i + width * j));
                newIndex++;
            }
        }
    }
}

void printMex(double* elements, int* dim){
    mexPrintf("\n\nHeight: %d \tWidth: %d \tDepth: %d\n",dim[1],dim[0],dim[2]);
    if (dim[2] > 1){
        for(int z = 0; z < dim[2]; z++) {
            mexPrintf("Dimension: %d\n",z+1);
            for (int i = 0; i < dim[1]; i++) {
                for (int j = 0; j < dim[0]; j++) {
                    mexPrintf("%f, ", get3DElement(elements, dim, i, j, z));
                }
                mexPrintf("\n");
            }
        }
    }else {
        for (int i = 0; i < dim[1]; i++) {
            for (int j = 0; j < dim[0]; j++) {
                mexPrintf("%f, ", getElement(elements, dim, i, j));
            }
            mexPrintf("\n");
        }
    }
}
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int* reDim = new int[3];
    //colMajorToRowMajor(mxGetPr(prhs[1]), width, height, dxElements
    setDimensions(mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[2], reDim);
    double* reElements = new double[reDim[0] * reDim[1] * reDim[2]];
//    int height = mxGetDimensions(prhs[0])[0];
//    int width = mxGetDimensions(prhs[0])[1];
//    int depth = mxGetDimensions(prhs[0])[2];
//    double* reElements = (double*)malloc(width * height * depth * sizeof(double));
    colMajorToRowMajor(mxGetPr(prhs[0]), reDim, reElements);
    //Matrix* re = new Matrix(width, height, depth, reElements);


    //

    int* dxDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[1], mxGetDimensions(prhs[1])[0], 1, dxDim);
    double* dxElements = new double[dxDim[0] * dxDim[1] * dxDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[1]), dxDim, dxElements);

    /*
    height = mxGetDimensions(prhs[1])[0];
    width = mxGetDimensions(prhs[1])[1];
    double* dxElements = (double*)malloc(width * height * sizeof(double));
    );
    Matrix* dx = new Matrix(width, height, dxElements);
    */


    //

    int* dyDim = new int[3];
    setDimensions(mxGetDimensions(prhs[2])[1], mxGetDimensions(prhs[2])[0], 1, dyDim);
    double* dyElements = new double[dyDim[0] * dyDim[1] * dyDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[2]), dyDim, dyElements);

    /*
    height = mxGetDimensions(prhs[2])[0];
    width = mxGetDimensions(prhs[2])[1];
    double* dyElements = (double*)malloc(width * height * sizeof(double));
    colMajorToRowMajor(mxGetPr(prhs[2]), width, height, dyElements);
    Matrix* dy = new Matrix(width, height, dyElements);
    */

    //


    int* dzDim = new int[3];
    setDimensions(mxGetDimensions(prhs[3])[1], mxGetDimensions(prhs[3])[0], 1, dyDim);
    double* dzElements = new double[dyDim[0] * dyDim[1] * dyDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[3]), dzDim, dzElements);

    /*
    height = mxGetDimensions(prhs[3])[0];
    width = mxGetDimensions(prhs[3])[1];
    double* dzElements = (double*)malloc(width * height * sizeof(double));
    colMajorToRowMajor(mxGetPr(prhs[3]), width, height, dzElements);
    Matrix* dz = new Matrix(width, height, dzElements);
*/

    //

    double* cc = new double[dxDim[0] * dyDim[0] * dzDim[0]];
    int* ccDim = new int[3];
    setDimensions(dxDim[0], dyDim[0], dzDim[0], ccDim);

    //printMex(reElements, reDim);
    //printMex(dxElements, dxDim);

    IP3d(reElements, reDim,
         dxElements, dxDim,
         dyElements, dyDim,
         dzElements, dzDim,
         cc, ccDim);


    nlhs = 1;
    int ndim = 3, dims[3] = {ccDim[1], ccDim[0], ccDim[2]};
    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    mexPrintf("%f",cc[0]);
    memcpy(mxGetPr(plhs[0]), cc, ccDim[0] * ccDim[1] * ccDim[2] * sizeof(double));
//    delete re;
//    delete dx;
//    delete dy;
//    delete dz;
    delete [] cc;
    delete [] reElements;
    delete [] dxElements;
    delete [] dyElements;
    delete [] dzElements;

    delete [] ccDim;
    delete [] reDim;
    delete [] dxDim;
    delete [] dyDim;
    delete [] dzDim;
}