#include "mex.h"
#include "IPSP3D.cpp"
using namespace std;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int reDim[] = {mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[2]};
    double* reElements = mxGetPr(prhs[0]);

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

    int ccDim[] = {1, dxDim[1], 1};

    nlhs = 1;
    int ndim = 3, dims[3] = {ccDim[0], ccDim[1], ccDim[2]};
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