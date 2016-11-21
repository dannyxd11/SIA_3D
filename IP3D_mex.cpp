#include "mex.h"
#include "IP3D.cpp"
using namespace std;


double getElementUsingMajor(double* elements, int majorWidth, int x, int y){
    return elements[majorWidth * x + y];
}

void colMajorToRowMajor(double* elements, int* dim, double *newElements){
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
        int newIndex = 0;
        for (int i = 0; i < dim[1]; i++) {
            for (int j = 0; j < dim[0]; j++) {
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
    setDimensions(mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[2], reDim);
    double* reElements = new double[reDim[0] * reDim[1] * reDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[0]), reDim, reElements);

   // printMex(reElements, reDim);

    //

    int* dxDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[1], mxGetDimensions(prhs[1])[0], 1, dxDim);
    double* dxElements = new double[dxDim[0] * dxDim[1] * dxDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[1]), dxDim, dxElements);


    //

    int* dyDim = new int[3];
    setDimensions(mxGetDimensions(prhs[2])[1], mxGetDimensions(prhs[2])[0], 1, dyDim);
    double* dyElements = new double[dyDim[0] * dyDim[1] * dyDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[2]), dyDim, dyElements);

    //


    int* dzDim = new int[3];
    setDimensions(mxGetDimensions(prhs[3])[1], mxGetDimensions(prhs[3])[0], 1, dzDim);
    double* dzElements = new double[dzDim[0] * dzDim[1] * dzDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[3]), dzDim, dzElements);


    //

    //double* cc = new double[dxDim[0] * dyDim[0] * dzDim[0]];
    int* ccDim = new int[3];
    setDimensions(dxDim[0], dyDim[0], dzDim[0], ccDim);


    //printMex(reElements, reDim);
    //printMex(dxElements, dxDim);

    nlhs = 1;
    int ndim = 3, dims[3] = {ccDim[1], ccDim[0], ccDim[2]};
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


//    nlhs = 1;
//    int ndim = 3, dims[3] = {ccDim[1], ccDim[0], ccDim[2]};
//    plhs[0] = mxCreateNumericArray(ndim,
//                                   dims,
//                                   mxDOUBLE_CLASS,
//                                   mxREAL);
    //mexPrintf("\n%f",cc[0]);
    //memcpy(mxGetPr(plhs[0]), cc, ccDim[0] * ccDim[1] * ccDim[2] * sizeof(double));
    //ccPointer = mxGetPr(plhs[0]);


    //delete [] cc;
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