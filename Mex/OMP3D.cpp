//
// Created by Dan on 26/01/2017.
//

#include "mex.h"
#include <cmath>
#include "commonOps.cpp"
#include "IP3D.cpp"


void changeDimensions(double *matrix, int *curDimensions, int height, int width, int depth) {
    double *newMatrix = new double[height * width * depth];
    memcpy(newMatrix, matrix, sizeof(double) * height * width * depth);

    delete[] matrix;
    matrix = newMatrix;
    setDimensions(height, width, depth, curDimensions);
}

int validateIndex(double indk, double *Di, int *DiDim) {
    for (int i = 0; i < DiDim[0] * DiDim[1] * DiDim[2]; i++) {
        if (Di[i] == indk) {
            return i;
        }
    }
    mexErrMsgTxt("Index not in dictionary");
    return 0;
}

int max(double *matrix, int *dimensions) {
    double maxValue = 0;
    int n1 = -1;
    for (int k = 0; k < dimensions[0] * dimensions[1] * dimensions[2]; k++) {
        if (std::abs(matrix[k]) > maxValue) {
            maxValue = std::abs(matrix[k]);
            n1 = k;
        }
    }
    return n1;
}

void initiateRangeVector(double *vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = i + 1;
    }
}

double sumOfSquares(double *matrix, int *dimensions) {
    double sum = 0;
    for (int i = 0; i < dimensions[0] * dimensions[1] * dimensions[2]; i++) {
        sum += matrix[i] * matrix[i];
    }
    return sum;
}

int numel(int *dimensions) {
    return dimensions[0] * dimensions[1] * dimensions[2];
}

double vectorNorm(double *matrix, int size) {
    double sum = 0;
    for (int n = 0; n < size; n++) {
        sum += matrix[n] * matrix[n];
    }
    return sqrt(sum);
}

void ind2sub(int *dimensions, int index, int *q) {
    int plane = dimensions[0] * dimensions[1];
    q[0] = index / plane;
    int rem = index % plane;
    q[1] = rem / dimensions[0];
    q[2] = rem % dimensions[1];
}

//void OMP3D(double* f, int* fDim, double* dx, int* dxDim, double* dy, int* dyDim, double* dz, int* dzDim, double tol, int No){

void
OMP3D(double *f, int *fDim, double *dx, int *dxDim, double *dy, int *dyDim, double *dz, int *dzDim, double tol, int No,
      double *h, int *hDim, double *Di1, int *Di1Dim, double *Di2, int *Di2Dim, double *Di3, int *Di3Dim, double *beta,
      int *betaDim,
      double *c, int *cDim, double *Q, int *QDim, double *noRe1) {

    std::string name = "OMP23_m";

    /*
    [Lx,Nx]=size(Dx);
    [Ly,Ny]=size(Dy);
    [Lz,Nz]=size(Dz);
    */
    int lx = dxDim[0];
    int nx = dxDim[1];
    int ly = dyDim[0];
    int ny = dyDim[1];
    int lz = dzDim[0];
    int nz = dzDim[1];

    /*
    delta=1/(Lx*Ly*Lz);
    N=Nx*Ny*Nz;
    */

    int delta = 1 / (lx * ly * lz);
    int N = nx * ny * nz;

    /*
    Dix=1:Nx;
    Diy=1:Ny;
    Diz=1:Nz;
    */

    double *Dix = new double[nx];
    double *Diy = new double[ny];
    double *Diz = new double[nz];

    int *DixDim = new int[3];
    int *DiyDim = new int[3];
    int *DizDim = new int[3];

    setDimensions(1, nx, 1, DixDim);
    setDimensions(1, ny, 1, DiyDim);
    setDimensions(1, nz, 1, DizDim);

    initiateRangeVector(Dix, nx);
    initiateRangeVector(Diy, ny);
    initiateRangeVector(Diz, nz);

    /*
    H1=zeros(size(f(:)'));
    */

    double *H1 = new double[fDim[0] * fDim[1] * fDim[2]]();
    int *H1Dim = new int[3];
    setDimensions(fDim, H1Dim);


    //todo
//    if(sumOfSquares(f, fDim) * delta < 1e-9) {
//        Di1 = [];
//        Di2 = [];
//        Di3 = [];
//        beta = [];
//        Q = [];
//        c = 0;
//        H = zeros(size(f));
//        return;
//    }

    int zmax = 1; // number of reorthogonalizations


    /*
    beta = []
    */
    //double* beta = new double[dxDim[0]]();
    //int* betaDim = new int[3];

    /*
    re=f;
    */

    double *re = new double[fDim[0] * fDim[1] * fDim[2]];
    int *reDim = new int[3];
    setDimensions(fDim, reDim);
    memcpy(re, f, fDim[0] * fDim[1] * fDim[2] * sizeof(double));


    /*//todo
    Validate inputs..
    if (nargin<9) | (isempty(indz)==1)  indz=[];end
    if (nargin<8) | (isempty(indy)==1)  indy=[];end
    if (nargin<7) | (isempty(indx)==1)  indx=[];end
    if (nargin<6) | (isempty(No)==1)    No=Lx*Ly*Lz;end
    if (nargin<5) | (isempty(tol)==1)   tol=6.5;end;
    */
    int *indx = new int[0];
    int *indy = new int[0];
    int *indz = new int[0];
    tol = 6.5;

    int numind = 0; // numel

    //atoms having smaller norm than tol1 are supposed be zero ones
    double tol1 = 1e-7;  //%1e-5
    double tol2 = 1e-10; //   %0.0001  %1e-5

// Main algorithm: at kth iteration
    /*
    min(No,N);
    */
    int H = No < N ? No : N; // maximal number of function in sub-dictionary
    H = N; // temp
    //double* Q = new double[dxDim[0] * dyDim[0] * dzDim[0]];
    //double* nore1 = new double[H];
    //double* Di1 = new double[H];
    //double* Di2 = new double[H];
    //double* Di3 = new double[H];

    for (int k = 0; k < H; k++) {

        double *cc = new double[nx * ny]();
        int *ccDim = new int[3];
        setDimensions(dxDim[1], dyDim[1], dzDim[1], ccDim);

        //int* QDim = new int[3];
        int *q = new int[3];
        if (k < numind) {

            int qx = validateIndex(indx[k], Dix, DixDim);
            int qy = validateIndex(indy[k], Diy, DiyDim);
            int qz = validateIndex(indz[k], Diz, DizDim);
            q[0] = indx[k];
            q[1] = indy[k];
            q[2] = indz[k];
        } else {

            IP3d(re, reDim, dx, dxDim, dy, dyDim, dz, dzDim, cc, ccDim);
            int maxind = max(cc, ccDim);
            ind2sub(ccDim, maxind, q);
			ccDim[0];
			ccDim[1];
			ccDim[2];
            if (cc[maxind] < tol2) {
                k = k - 1;
                mexPrintf("OMP3D stopped, max(|<f,q|/||q||) <= tol2 = %g\n", tol2);
                break;
            }
        }

        double *new_atom2 = new double[dyDim[0] * dxDim[0]]; // Only height since its 'Kronecking'  columns
        int *new_atom2Dim = new int[3];

        double *new_atom = new double[dzDim[0] * new_atom2Dim[0]];
        int *new_atomDim = new int[3];

        Di1[k] = q[0];
        Di2[k] = q[1];
        Di3[k] = q[2];
        if (k > 0) {
            double *new_atom2 = new double[dyDim[0] * dxDim[0]]; // Only height since its 'Kronecking'  columns
            int *new_atom2Dim = new int[3];

            int *tempDxColDim = new int[3];
            setDimensions(dxDim[0], 1, 1, tempDxColDim);
            int *tempDyColDim = new int[3];
            setDimensions(dyDim[0], 1, 1, tempDyColDim);
            int *tempDzColDim = new int[3];
            setDimensions(dyDim[0], 1, 1, tempDzColDim);

            kroneckerProduct(getCol(dy, dyDim, q[1]), tempDyColDim, getCol(dx, dxDim, q[0]), tempDxColDim, new_atom2,
                             new_atom2Dim);
            // new_atom2=kronecker(Dy(:,q(2)),Dx(:,q(1)));



            kroneckerProduct(getCol(dz, dzDim, q[2]), tempDzColDim, new_atom2, new_atom2Dim, new_atom, new_atomDim);
            //new_atom=kronecker(Dz(:,q(3)),new_atom2);
            //o_reorthogonalize(Q,new_atom,zmax);

            orthogonalize(Q, QDim, new_atom, new_atomDim); // Need to sort out resizing
            reorthogonalize(Q, QDim, zmax);

            delete[] new_atom2;
            delete[] new_atom2Dim;
            delete[] tempDxColDim;
            delete[] tempDyColDim;
            delete[] tempDzColDim;
            delete[] new_atom;
            delete[] new_atomDim;
        }

        if (k == 0) {


            int *tempDxColDim = new int[3];
            setDimensions(dxDim[0], 1, 1, tempDxColDim);
            int *tempDyColDim = new int[3];
            setDimensions(dyDim[0], 1, 1, tempDyColDim);
            int *tempDzColDim = new int[3];
            setDimensions(dyDim[0], 1, 1, tempDzColDim);

            kroneckerProduct(getCol(dy, dyDim, q[1]), tempDyColDim, getCol(dx, dxDim, q[0]), tempDxColDim, new_atom2,
                             new_atom2Dim);

            double *new_atom = new double[dzDim[0] * dyDim[0] * dyDim[0]];
            int *new_atomDim = new int[3];

            kroneckerProduct(getCol(dz, dzDim, q[2]), tempDzColDim, new_atom2, new_atom2Dim, new_atom, new_atomDim);
            memcpy(&Q, &new_atom, new_atomDim[0]);
        }
		mexPrintf("%f,1\n",QDim[0]);
		QDim[1];
		QDim[2];


        double nork = vectorNorm(getCol(Q, QDim, k), QDim[0]); // nork=norm(Q(:,k));
        for (int n = 0; n < QDim[0]; n++) { // Q(:,k) = Q(:,k) / nork;
            Q[k * QDim[0] + n] /= nork;
        }

        if (k > 0) {
            int *tempColDim = new int[3];
            setDimensions(QDim[0], 1, 1, tempColDim);
            biorthogonalize(beta, betaDim, getCol(Q, QDim, k), tempColDim, new_atom, new_atomDim, nork);
        }


        /*
        beta(:,k) =Q(:,k) / nork; // kth biorthogonal function
        */
        memcpy(getCol(beta, betaDim, k), getCol(Q, QDim, k), QDim[0] * sizeof(double));
        for (int i = 0; i < betaDim[0]; i++) {
            beta[i + k * betaDim[0]] /= nork;
        }

        /*
        h  = f(:)' * Q(:,k) * Q(:,k)' ;
        */

        int *tempFVectorDim = new int[3];
        setDimensions(1, fDim[0] * fDim[1] * fDim[2], 1, tempFVectorDim);

        int *tempQkDim = new int[3];
        setDimensions(QDim[0], 1, 1, tempQkDim);

        int *tempQkRowDim = new int[3];
        setDimensions(1, QDim[0], 1, tempQkRowDim);

        double *multresult = new double[QDim[0] * QDim[0]];
        int *multresultDim = new int[3];
        setDimensions(QDim[0], QDim[0], QDim[0], multresultDim);

        double *h = new double[fDim[0] * fDim[1] * fDim[2]];
        int *hDim = new int[3];
        blasMultiply(getCol(Q, QDim, k), tempQkDim, getCol(Q, QDim, k), tempQkRowDim, multresult, multresultDim);
        blasMultiply(f, tempFVectorDim, multresult, multresultDim, h, hDim);



        /*
        re = re(:)-h';
        H1 = H1+h;
        */

        for (int i = 0; i < numel(hDim); i++) {
            re[i] -= h[i];
            H1[i] += h[i];
        }

        /*
        nore1(k) = (norm(re))^2*(delta);
        */

        noRe1[k] = pow(vectorNorm(re, numel(hDim)), 2) * delta;

        /*
        re = reshape(re,Lx,Ly,Lz);
        */

        reDim[0] = lx;
        reDim[1] = ly;
        reDim[2] = lz;

        if (tol != 0 && (noRe1[k] < tol)) {
            break;
        }
    }

    /*
    c=f(:)'*beta;
    */

    int *tempFrowDim = new int[3];
    setDimensions(1, fDim[0] * fDim[1] * fDim[2], 1, tempFrowDim);

    double *cMultResult = new double[1];
    int *cMultResultDim = new int[3];
    setDimensions(1, 1, 1, cMultResultDim);

    blasMultiply(f, tempFrowDim, beta, betaDim, cMultResult, cMultResultDim);
    c = cMultResult;

    /*
    H=reshape(H1,Lx,Ly,Lz);
    */

    memcpy(h, H1, lx * ly * lz * sizeof(double));


}


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    // function [H,Di1,Di2,Di3, beta, c, Q, nore1]
    // OMP3D_m(f,Dx,Dy,Dz,tol, No, indx, indy,indz)


    //Assigning Input Variables
    double *f = mxGetPr(prhs[0]);
    double *dx = mxGetPr(prhs[1]);
    double *dy = mxGetPr(prhs[2]);
    double *dz = mxGetPr(prhs[3]);
    double tol = mxGetScalar(prhs[4]);
    //int No = (int)mxGetScalar(prhs[5]);
	

    int *fDim = new int[3];
    int *dxDim = new int[3];
    int *dyDim = new int[3];
    int *dzDim = new int[3];

    setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], mxGetDimensions(prhs[0])[2], fDim);
    setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], mxGetDimensions(prhs[1])[2], dxDim);
    setDimensions(mxGetDimensions(prhs[2])[0], mxGetDimensions(prhs[2])[1], mxGetDimensions(prhs[2])[2], dyDim);
    setDimensions(mxGetDimensions(prhs[3])[0], mxGetDimensions(prhs[3])[1], mxGetDimensions(prhs[3])[2], dzDim);

	int No = dxDim[0] * dxDim[1] * dxDim[2];

    //Preparing Output Variables
    int *hDim = new int[3];
    int *Di1Dim = new int[3];
    int *Di2Dim = new int[3];
    int *Di3Dim = new int[3];
    int *betaDim = new int[3];
    int *cDim = new int[3];
    int *qDim = new int[3];
    int *noRe1Dim = new int[3];

    setDimensions(fDim, hDim);
    setDimensions(1, dxDim[1] * dyDim[1] * dzDim[1], 1, Di1Dim);
    setDimensions(1, dxDim[1] * dyDim[1] * dzDim[1], 1, Di2Dim);
    setDimensions(1, dxDim[1] * dyDim[1] * dzDim[1], 1, Di3Dim);
    setDimensions(dxDim[0], 1, 1, betaDim);
    setDimensions(1, 1, 1, cDim);
    setDimensions(1, 1, 1, qDim);
    setDimensions(1, 1, 1, noRe1Dim);

    int HNdim = 3, HDims[3] = {hDim[0], hDim[1], hDim[2]};
    int Di1Ndim = 3, Di1Dims[3] = {Di1Dim[0], Di1Dim[1], Di1Dim[2]};
    int Di2Ndim = 3, Di2Dims[3] = {Di2Dim[0], Di2Dim[1], Di2Dim[2]};
    int Di3Ndim = 3, Di3Dims[3] = {Di3Dim[0], Di3Dim[1], Di3Dim[2]};
    int betaNdim = 3, betaDims[3] = {betaDim[0], betaDim[1], betaDim[2]};
    int cNdim = 3, cDims[3] = {cDim[0], cDim[1], cDim[2]};
    int qNdim = 3, qDims[3] = {qDim[0], qDim[1], qDim[2]};
    int noRe1NDim = 3, noRe1Dims[3] = {noRe1Dim[0], noRe1Dim[1], noRe1Dim[2]};

    plhs[0] = mxCreateNumericArray(HNdim, HDims, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(Di1Ndim, Di1Dims, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(Di2Ndim, Di2Dims, mxDOUBLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericArray(Di3Ndim, Di3Dims, mxDOUBLE_CLASS, mxREAL);
    plhs[4] = mxCreateNumericArray(betaNdim, betaDims, mxDOUBLE_CLASS, mxREAL);
    plhs[5] = mxCreateNumericArray(cNdim, cDims, mxDOUBLE_CLASS, mxREAL);
    plhs[6] = mxCreateNumericArray(qNdim, qDims, mxDOUBLE_CLASS, mxREAL);
    plhs[7] = mxCreateNumericArray(noRe1NDim, noRe1Dims, mxDOUBLE_CLASS, mxREAL);

    double *h = mxGetPr(plhs[0]);
    double *Di1 = mxGetPr(plhs[1]);
    double *Di2 = mxGetPr(plhs[2]);
    double *Di3 = mxGetPr(plhs[3]);
    double *beta = mxGetPr(plhs[4]);
    double *c = mxGetPr(plhs[5]);
    double *q = mxGetPr(plhs[6]);
    double *noRe1 = mxGetPr(plhs[7]);

    nlhs = 8;

    OMP3D(f, fDim, dx, dxDim, dy, dyDim, dz, dzDim, tol, No,
          h, hDim, Di1, Di1Dim, Di2, Di2Dim, Di3, Di3Dim, beta, betaDim, c, cDim, q, qDim, noRe1);


	delete [] fDim;
	delete [] dxDim;
	delete [] dyDim;
	delete [] dzDim;
	delete [] hDim;
	delete [] Di1Dim;
	delete [] Di2Dim;
	delete [] Di3Dim;
	delete [] betaDim;
	delete [] cDim;
	delete [] qDim;
	delete [] noRe1Dim;
}

