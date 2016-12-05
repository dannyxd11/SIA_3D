#include "mex.h"
#include "IPSP3D_loop.cpp"
using namespace std;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    int* reDim = new int[3];
    setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[2], reDim);
    double* reElements = mxGetPr(prhs[0]);

    //

    int* dxDim = new int[3];
    setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], 1, dxDim);
    double* dxElements = mxGetPr(prhs[1]);


    //

    int* dyDim = new int[3];
    setDimensions(mxGetDimensions(prhs[2])[0], mxGetDimensions(prhs[2])[1], 1, dyDim);
    double* dyElements = mxGetPr(prhs[2]);

    //


    int* dzDim = new int[3];
    setDimensions(mxGetDimensions(prhs[3])[0], mxGetDimensions(prhs[3])[1], 1, dzDim);
    double* dzElements = mxGetPr(prhs[3]);


    //

    int* ccDim = new int[3];
    setDimensions(1, dxDim[1], 1, ccDim);

    nlhs = 1;
    int ndim = 3, dims[3] = {ccDim[0], ccDim[1], ccDim[2]};
    plhs[0] = mxCreateNumericArray(ndim,
                                   dims,
                                   mxDOUBLE_CLASS,
                                   mxREAL);
    double* cc = mxGetPr(plhs[0]);

    IPSP3d_loop(reElements, reDim,
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