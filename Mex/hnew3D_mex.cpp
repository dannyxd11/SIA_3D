#include "mex.h"
#include "hnew3D.cpp"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int ccDim[] = {mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[2]};
    double* ccElements = mxGetPr(prhs[0]);


    //

    int dxDim[] = {mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], 1};
    double* dxElements = mxGetPr(prhs[1]);


    //

    int dyDim[] = {mxGetDimensions(prhs[2])[0], mxGetDimensions(prhs[2])[1], 1};
    double* dyElements = mxGetPr(prhs[2]);

    //


    int dzDim[] = {mxGetDimensions(prhs[3])[0], mxGetDimensions(prhs[3])[1], 1};
    double* dzElements = mxGetPr(prhs[3]);


    //

    int hnewDim[] = {dxDim[0], dyDim[0], dzDim[0]};
    nlhs = 1;
    int ndim = 3, dims[3] = {hnewDim[0], hnewDim[1], hnewDim[2]};
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