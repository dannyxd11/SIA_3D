#include "mex.h"
#include "IPSP3D.cpp"
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
                newIndex++;
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int* reDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[2], reDim);
    double* reElements = new double[reDim[0] * reDim[1] * reDim[2]];
    colMajorToRowMajor(mxGetPr(prhs[0]), reDim, reElements);

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

    int* ccDim = new int[3];
    setDimensions(dxDim[0], 1, 1, ccDim);

    nlhs = 1;
    int ndim = 3, dims[3] = {ccDim[1], ccDim[0], ccDim[2]};
    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    double* cc = mxGetPr(plhs[0]);

    IPSP3d(reElements, reDim,
           dxElements, dxDim,
           dyElements, dyDim,
           dzElements, dzDim,
           cc, ccDim);

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