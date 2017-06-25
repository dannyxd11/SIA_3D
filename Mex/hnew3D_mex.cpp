#include "mex.h"
#include "hnew3D.cpp"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int ccDim[] = {(int)mxGetDimensions(prhs[0])[0], (int)mxGetDimensions(prhs[0])[1], (int)mxGetDimensions(prhs[0])[2]};
    double* ccElements = mxGetPr(prhs[0]);


    //

    int dxDim[] = {(int)mxGetDimensions(prhs[1])[0], (int)mxGetDimensions(prhs[1])[1], 1};
    double* dxElements = mxGetPr(prhs[1]);


    //

    int dyDim[] = {(int)mxGetDimensions(prhs[2])[0], (int)mxGetDimensions(prhs[2])[1], 1};
    double* dyElements = mxGetPr(prhs[2]);

    //


    int dzDim[] = {(int)mxGetDimensions(prhs[3])[0], (int)mxGetDimensions(prhs[3])[1], 1};
    double* dzElements = mxGetPr(prhs[3]);


    //

    int hnewDim[] = {dxDim[0], dyDim[0], dzDim[0]};
    nlhs = 1;
    int ndim = 3;
    size_t dims[3] = {(size_t)hnewDim[0], (size_t)hnewDim[1], (size_t)hnewDim[2]};
    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    double* hnew = mxGetPr(plhs[0]);

    hnew3d(ccElements, ccDim,
           dxElements, dxDim,
           dyElements, dyDim,
           dzElements, dzDim,
           hnew, hnewDim);

}