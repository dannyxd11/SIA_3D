#include "mex.h"
#include "IP3D.cpp"
using namespace std;

void printMex(double* elements, int* dim){
    mexPrintf("\n\nHeight: %d \tWidth: %d \tDepth: %d\n",dim[0],dim[1],dim[2]);
    if (dim[2] > 1){
        for(int z = 0; z < dim[2]; z++) {
            mexPrintf("Dimension: %d\n",z+1);
            for (int i = 0; i < dim[0]; i++) {
                for (int j = 0; j < dim[1]; j++) {
                    mexPrintf("%f, ", get3DElement(elements, dim, j, i, z));
                }
                mexPrintf("\n");
            }
        }
    }else {
        for (int i = 0; i < dim[0]; i++) {
            for (int j = 0; j < dim[1]; j++) {
                mexPrintf("%f, ", getElement(elements, dim, j, i));
            }
            mexPrintf("\n");
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int* reDim = new int[3];  
    setDimensions((int)mxGetDimensions(prhs[0])[0], (int)mxGetDimensions(prhs[0])[1], (int)mxGetDimensions(prhs[0])[2], reDim);
    double* reElements = mxGetPr(prhs[0]);


    //

    int* dxDim = new int[3];
    setDimensions((int)mxGetDimensions(prhs[1])[0], (int)mxGetDimensions(prhs[1])[1], 1, dxDim);
    double* dxElements = mxGetPr(prhs[1]);


    //

    int* dyDim = new int[3];
    setDimensions((int)mxGetDimensions(prhs[2])[0], (int)mxGetDimensions(prhs[2])[1], 1, dyDim);
    double* dyElements = mxGetPr(prhs[2]);

    //


    int* dzDim = new int[3];
    setDimensions((int)mxGetDimensions(prhs[3])[0], (int)mxGetDimensions(prhs[3])[1], 1, dzDim);
    double* dzElements = mxGetPr(prhs[3]);


    //

    int* ccDim = new int[3];
    setDimensions(dxDim[1], dyDim[1], dzDim[1], ccDim);
    nlhs = 1;
    int ndim = 3; 
    size_t dims[3] = {(size_t)ccDim[0], (size_t)ccDim[1], (size_t)ccDim[2]};
    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    double* cc = mxGetPr(plhs[0]);

    IP3d(reElements, reDim,
         dxElements, dxDim,
         dyElements, dyDim,
         dzElements, dzDim,
         cc, ccDim);

    delete [] ccDim;
    delete [] reDim;
    delete [] dxDim;
    delete [] dyDim;
    delete [] dzDim;
}