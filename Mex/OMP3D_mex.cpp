//
// Created by Dan on 26/01/2017.
//

#include "mex.h"
#include <cmath>
#include "commonOps.cpp"
#include "IP3D.cpp"
//
// Created by Dan on 26/01/2017.
//

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

int ismember(double* matrix, int size, double member){
    for (int i = 0; i < size; i++) {
        if (matrix[i] == member) {
            return 1;
        }
    }
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
    q[2] = index / plane;
    int rem = index % plane;
    q[1] = rem / dimensions[0];
    q[0] = rem % dimensions[1];
}

//void OMP3D(double* f, int* fDim, double* dx, int* dxDim, double* dy, int* dyDim, double* dz, int* dzDim, double tol, int No){

void
OMP3D(double *f, int *fDim, double *dx, int *dxDim, double *dy, int *dyDim, double *dz, int *dzDim, double tol, int No,
      double *H, int *HDim, double *Di1, int *Di1Dim, double *Di2, int *Di2Dim, double *Di3, int *Di3Dim, double *beta,
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

    double delta = 1.0 / (lx * ly * lz);
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
    int lim = No < N ? No : N; // maximal number of function in sub-dictionary
    lim = N; // temp
    //double* Q = new double[dxDim[0] * dyDim[0] * dzDim[0]];
    //double* nore1 = new double[H];
    //double* Di1 = new double[H];
    //double* Di2 = new double[H];
    //double* Di3 = new double[H];

    double* cc = new double[nx * ny * nz]();
    int ccDim[] = {nx, ny, nz};
    double* h = new double[fDim[0] * fDim[1] * fDim[2]]();

    int hDim[] = {reDim[0], reDim[1], reDim[2]};

    int *q = new int[3];


    for (int k = 0; k < lim; k++) {



        //int* QDim = new int[3];

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
            if (std::abs(cc[maxind]) < tol2) {
                //k = k - 1;

                mexPrintf("OMP3D stopped, max(|<f,q|/||q||) <= tol2 = %g\n", tol2);
                break;
            }
        }

        double *new_atom2 = new double[dyDim[0] * dxDim[0]]; // Only height since its 'Kronecking'  columns
        int *new_atom2Dim = new int[3];

        double *new_atom = new double[dzDim[0] * dyDim[0] * dxDim[0]];
        int *new_atomDim = new int[3];

        Di1[k] = q[0];
        Di2[k] = q[1];
        Di3[k] = q[2];

        Di1Dim[1] = k + 1;
        Di2Dim[1] = k + 1;
        Di3Dim[1] = k + 1;



        int *tempDxColDim = new int[3];
        setDimensions(dxDim[0], 1, 1, tempDxColDim);
        int *tempDyColDim = new int[3];
        setDimensions(dyDim[0], 1, 1, tempDyColDim);
        int *tempDzColDim = new int[3];
        setDimensions(dzDim[0], 1, 1, tempDzColDim);

        if (k > 0) {
            //double *new_atom2 = new double[dyDim[0] * dxDim[0]]; // Only height since its 'Kronecking'  columns
            //int *new_atom2Dim = new int[3];

            kroneckerProduct(getCol(dy, dyDim, q[1]), tempDyColDim, getCol(dx, dxDim, q[0]), tempDxColDim, new_atom2,
                             new_atom2Dim);
            // new_atom2=kronecker(Dy(:,q(2)),Dx(:,q(1)));



            kroneckerProduct(getCol(dz, dzDim, q[2]), tempDzColDim, new_atom2, new_atom2Dim, new_atom, new_atomDim);
            //new_atom=kronecker(Dz(:,q(3)),new_atom2);
            //o_reorthogonalize(Q,new_atom,zmax);


            orthogonalize(Q, QDim, new_atom, new_atomDim); // Need to sort out resizing			
            reorthogonalize(Q, QDim, zmax);


        }

        if (k == 0) {

            kroneckerProduct(getCol(dy, dyDim, q[1]), tempDyColDim, getCol(dx, dxDim, q[0]), tempDxColDim, new_atom2,
                             new_atom2Dim);

            kroneckerProduct(getCol(dz, dzDim, q[2]), tempDzColDim, new_atom2, new_atom2Dim, new_atom, new_atomDim);
            memcpy(&Q[0], &new_atom[0], new_atomDim[0] * sizeof(double));
        }

        double nork = vectorNorm(getCol(Q, QDim, k), QDim[0]); // nork=norm(Q(:,k));
        for (int n = 0; n < QDim[0]; n++) { // Q(:,k) = Q(:,k) / nork;
            Q[k * QDim[0] + n] /= nork;
        }


		//setDimensions(QDim, new_atomDim);
        
		if (k > 0) {
            int *tempColDim = new int[3];
            setDimensions(QDim[0], 1, 1, tempColDim);
            biorthogonalize(beta, betaDim, getCol(Q, QDim, k), tempColDim, new_atom, new_atomDim, nork);
            delete [] tempColDim;
        }

        /*
        beta(:,k) =Q(:,k) / nork; // kth biorthogonal function
        */
        betaDim[1] = QDim[1];


        //memcpy(getCol(beta, betaDim, k), getCol(Q, QDim, k), QDim[0] * sizeof(double));
        for (int i = 0; i < betaDim[0]; i++) {
            beta[i + k * betaDim[0]] = Q[k * QDim[0] + i] / nork;

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

        setDimensions(QDim[0], QDim[0], 1, multresultDim);
		


        blasMultiply(getCol(Q, QDim, k), tempQkDim, getCol(Q, QDim, k), tempQkRowDim, multresult, multresultDim);
		setDimensions(1, multresultDim[1], 1, hDim);

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

        delete [] tempFVectorDim;
        delete [] tempQkDim;
        delete [] tempQkRowDim;
        delete [] multresult;
        delete [] multresultDim;
        delete[] new_atom2;
        delete[] new_atom2Dim;
        delete[] tempDxColDim;
        delete[] tempDyColDim;
        delete[] tempDzColDim;
        delete[] new_atom;
        delete[] new_atomDim;

        if (tol != 0 && (noRe1[k] < tol)) {
            break;
        }

    }

    /*
    c=f(:)'*beta;
    */

    int *tempRowDim = new int[3];
    setDimensions(1, fDim[0] * fDim[1] * fDim[2], 1, tempRowDim);
    setDimensions(1, betaDim[1], 1, cDim);

    blasMultiply(f, tempRowDim, beta, betaDim, c, cDim);

    /*
    H=reshape(H1,Lx,Ly,Lz);
    */

    memcpy(H, H1, lx * ly * lz * sizeof(double));


    delete [] Dix;
    delete [] Diy;
    delete [] Diz;
    delete [] DixDim;
    delete [] DiyDim;
    delete [] DizDim;
    delete [] tempRowDim;
    delete [] H1;
    delete [] H1Dim;
    delete [] q;

    //  delete [] tempRowDim;
    //  delete [] cMultResult;
    //  delete [] cMultResultDim;

    delete [] cc;
    delete [] re;
    delete [] reDim;
    delete [] h;

    // indx/y/z not supported
    delete [] indx;
    delete [] indy;
    delete [] indz;
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


    setDimensions(fDim, hDim);
    setDimensions(1, dxDim[1] * dyDim[1] * dzDim[1], 1, Di1Dim);
    setDimensions(1, dxDim[1] * dyDim[1] * dzDim[1], 1, Di2Dim);
    setDimensions(1, dxDim[1] * dyDim[1] * dzDim[1], 1, Di3Dim);
    setDimensions(dxDim[0] * dyDim[0] * dzDim[0], 1, 1, betaDim);
    setDimensions(dxDim[0], dxDim[0], dxDim[0], cDim);
    setDimensions(dxDim[0] * dyDim[0] * dzDim[0], 1, 1, qDim); // gets changed by multiply at end of routine


    int HNdim = 3;
    int Di1Ndim = 3;
    int Di2Ndim = 3;
    int Di3Ndim = 3;
    int betaNdim = 3;
    int cNdim = 3;
    int qNdim = 3;
    int noRe1NDim = 3;


    double* h = new double[hDim[0] * hDim[1] * hDim[2]];
    double* Di1 = new double[dxDim[1] * dyDim[1] * dzDim[1]];
    double* Di2 = new double[dxDim[1] * dyDim[1] * dzDim[1]];
    double* Di3 = new double[dxDim[1] * dyDim[1] * dzDim[1]];
    double* beta = new double[dxDim[1] * dyDim[1] * dzDim[1]]();
    double* c = new double[dxDim[0] * dyDim[0] * dzDim[0]];
    double* q = new double[dxDim[1] * dyDim[1] * dzDim[1]];

    double* noRe1 = new double[dxDim[1] * dyDim[1] * dzDim[1]];



    nlhs = 8;

    OMP3D(f, fDim, dx, dxDim, dy, dyDim, dz, dzDim, tol, No,
          h, hDim, Di1, Di1Dim, Di2, Di2Dim, Di3, Di3Dim, beta, betaDim, c, cDim, q, qDim, noRe1);


    // Most output arguments sizes arent known until after the routine is completed, create MatLab arrays after routine has finished.



    int noRe1Dim[] = {1, Di1Dim[1], 1};

    plhs[0] = mxCreateNumericArray(HNdim, hDim, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(Di1Ndim, Di1Dim, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(Di2Ndim, Di2Dim, mxDOUBLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericArray(Di3Ndim, Di3Dim, mxDOUBLE_CLASS, mxREAL);
    plhs[4] = mxCreateNumericArray(betaNdim, betaDim, mxDOUBLE_CLASS, mxREAL);
    plhs[5] = mxCreateNumericArray(cNdim, cDim, mxDOUBLE_CLASS, mxREAL);
    plhs[6] = mxCreateNumericArray(qNdim, qDim, mxDOUBLE_CLASS, mxREAL);
    plhs[7] = mxCreateNumericArray(noRe1NDim, noRe1Dim, mxDOUBLE_CLASS, mxREAL);

    memcpy(mxGetPr(plhs[0]), h, sizeof(double) * hDim[0] * hDim[1] * hDim[2]);
    memcpy(mxGetPr(plhs[1]), Di1, sizeof(double) * Di1Dim[0] * Di1Dim[1] * Di1Dim[2]);
    memcpy(mxGetPr(plhs[2]), Di2, sizeof(double) * Di2Dim[0] * Di2Dim[1] * Di2Dim[2]);
    memcpy(mxGetPr(plhs[3]), Di3, sizeof(double) * Di3Dim[0] * Di3Dim[1] * Di3Dim[2]);
    memcpy(mxGetPr(plhs[4]), beta, sizeof(double) * betaDim[0] * betaDim[1] * betaDim[2]);
    memcpy(mxGetPr(plhs[5]), c, sizeof(double) * cDim[0] * cDim[1] * cDim[2]);
    memcpy(mxGetPr(plhs[6]), q, sizeof(double) * qDim[0] * qDim[1] * qDim[2]);
    memcpy(mxGetPr(plhs[7]), noRe1, sizeof(double) * noRe1Dim[1]); //todo Define numat





    delete [] h;
    delete [] Di1;
    delete [] Di2;
    delete [] Di3;
    delete [] beta;
    delete [] c;
    delete [] q;

	delete [] noRe1;





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



}

