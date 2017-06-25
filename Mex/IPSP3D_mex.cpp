#include "mex.h"
#include "IPSP3D.cpp"
using namespace std;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int reDim[] = {(int)mxGetDimensions(prhs[0])[0], (int)mxGetDimensions(prhs[0])[1], (int)mxGetDimensions(prhs[0])[2]};
    double* reElements = mxGetPr(prhs[0]);

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

    int ccDim[] = {1, dxDim[1], 1};

    nlhs = 1;
    int ndim = 3;
    size_t dims[3] = {(size_t)ccDim[0], (size_t)ccDim[1], (size_t)ccDim[2]};
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
}