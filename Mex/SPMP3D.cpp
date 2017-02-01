//
// Created by Dan on 30/01/2017.
//

#include "mex.h"
#include "commonOps.cpp"
#include "ProjMP3D.cpp"
#include "IP3D.cpp"
#include "hnew3d.cpp"
#include <cmath>

using namespace std;

double sumOfSquares(double *matrix, int *dimensions) {
    double sum = 0;
    for (int i = 0; i < dimensions[0] * dimensions[1] * dimensions[2]; i++) {
        sum += matrix[i] * matrix[i];
    }
    return sum;
}


int max(double *matrix, int *dimensions) {
    double maxValue = std::abs(matrix[0]);
    int n1 = -1;
    for (int k = 0; k < dimensions[0] * dimensions[1] * dimensions[2]; k++) {
        if (std::abs(matrix[k]) > maxValue) {
            maxValue = std::abs(matrix[k]);
            n1 = k;
        }
    }
    return n1;
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

void initiateRangeVector(double *vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = i + 1;
    }
}

int nonZeroNumel(double* m, int size){
    int numel = 0;
    for(int i = 0; i < size; i++){
        if(m[i] != 0) numel += 1;
    }
    return numel;
}

void ind2sub(int *dimensions, int index, int *q) {
    int plane = dimensions[0] * dimensions[1];
    q[0] = index / plane;
    int rem = index % plane;
    q[1] = rem / dimensions[0];
    q[2] = rem % dimensions[1];
}

void SPMP3D(double* f, int* fDim, double* Vx, int* VxDim, double* Vy, int* VyDim, double* Vz, int* VzDim,
        double tol, int No, double toln, int lstep, int Max, int Maxp, double* indx, double* indy, double* indz,
        double* h, double* c, double* Set_ind){
// h, c, set_ind

for(int i = 0; i < VxDim[0] * VyDim[0] * VzDim[0]; i++){
    c[0] = 0;
}

int Lx = VxDim[0];
int Nx = VxDim[1];
int Ly = VyDim[0];
int Ny = VyDim[1];
int Lz = VzDim[0];
int Nz = VzDim[1];

double delta = 1 / (Lx * Ly * Lz);
int Nxyz = Lx * Ly * Lz;

//if (nargin<12) | (isempty(indy)==1)  indz=[];end
//if (nargin<11) | (isempty(indy)==1)  indy=[];end
//if (nargin<10) | (isempty(indx)==1)  indx=[];end
//if (nargin<9) | (isempty(Maxp)==1)  Maxp=7000;end
//if (nargin<8)  | (isempty(Max)==1)   Max=7000;end
//if (nargin<7)  | (isempty(lstep)==1) lstep=0;end
//if (nargin<6)  | (isempty(toln)==1)  tolnu=1e-3;end
//if (nargin<5)  | (isempty(No)==1)    No=Nxyz;end
//if (nargin<4)  | (isempty(tol)==1)   tol=5e-4*sum(sum(sum(abs(f).^2)))*delta;end;

double* cp = new double[Nx * Ny * Nz]();
int cpDim[] = {Lx, Ly, Lz};
double* cc = new double[Nx * Ny * Nz]();
int ccDim[] = {Lx, Ly, Lz};
int cDim[] = {1, Lx * Ly * Lz, 1};

int MaxInd = max(f, fDim);
int MaxInt = f[MaxInd];
double* Di1; //= new double[ReDim[0] * ReDim[1]];
double* Di2; // = new double[ReDim[0] * ReDim[1]];
double* Di3; // = new double[ReDim[0] * ReDim[1]];
int numat = 0;
 Set_ind = new double[3 * Max];



double *Dix = new double[Nx];
double *Diy = new double[Ny];
double *Diz = new double[Nz];

int *DixDim = new int[3];
int *DiyDim = new int[3];
int *DizDim = new int[3];

setDimensions(1, Nx, 1, DixDim);
setDimensions(1, Ny, 1, DiyDim);
setDimensions(1, Nz, 1, DizDim);

initiateRangeVector(Dix, Nx);
initiateRangeVector(Diy, Ny);
initiateRangeVector(Diz, Nz);


int numind = 0; //todo numel(indx);
numind = 0;
double* h = new double[Lx * Ly * Lz];

if(sumOfSquares(f, fDim) * delta < 1e-9){
    c = new double[0];
    return;
}

double *Re = new double[fDim[0] * fDim[1] * fDim[2]];
int *ReDim = new int[3];
setDimensions(fDim, ReDim);
memcpy(Re, f, ReDim[0] * ReDim[1] * ReDim[2] * sizeof(double));

double tol2 = 1e-9;
int imp = 0;
int Maxit2 = 0;

if (lstep == -1)imp = 1;
if (imp == 1)lstep = 0;
if (lstep == 0) {
    Maxit2 = 1;
    lstep = Max;
}else {
    Maxit2 = Max / lstep;
}

/*
 *  Constant depending on dictionary size, move allocation outside loop to reduce unnessecary overhead
 */
int tempVxCol[] = {VxDim[0], 1, 1};
int tempVyCol[] = {VyDim[0], 1, 1};
int tempVzCol[] = {VzDim[0], 1, 1};
int h_newDim[] = {VxDim[0], VyDim[0], VzDim[0]};
int cscraDim[]= {1,1,1};
double *h_new = new double[h_newDim[0] * h_newDim[1] * h_newDim[2]];
int *q = new int[3];
double* Set_ind_trans;
int it;

    for (it = 0; it < Maxit2; it++) {
        for (int s = 0; s < lstep; s++) {

            if ((numat + 1) <= numind) {
                validateIndex(indx[numat + 1], Dix, DixDim);
                validateIndex(indy[numat + 1], Diy, DiyDim);
                validateIndex(indz[numat + 1], Diz, DizDim);

                q[0] = indx[numat + 1];
                q[1] = indy[numat + 1];
                q[2] = indz[numat + 1];

                set3DElement(cc, ccDim, q[1], q[2], q[3], 0) ;
                for (int zk = 0; zk < Lz; zk++) {
                    //todo    cc(q(1), q(2), q(3)) = cc(q(1), q(2), q(3)) + Vx(:,q(1))'*Re(:,:,zk)*Vy(:,q(2))*Vz(zk,q(3));
                }
            } else {
                IP3D(Re, ReDim, Vx, VxDim, Vy, VyDim, Vz, VzDim, cc, ccDim);
                int maxind = max(cc, ccDim);

                ind2sub(ccDim, maxind, q);

                if (cc[maxind] < tol2) {
                    mexPrintf("SPMP3D stopped, max(|<f,q|/||q||) <= tol2 = %g\n", tol2);
                    return;
                }
            }

            int vq[] = {q[1], q[2], q[3]};

            if (numat == 0) {
                Set_ind[0] = vq[0];
                Set_ind[1] = vq[1];
                Set_ind[2] = vq[2];
                numat += 1;
            } else {
                int exists = 0;
                /*
                 * [testq1, indq1] = ismember(vq, Set_ind, 'rows');
                 */
                for(int k = 0; k < numat; k++){
                    if(Set_ind[numat * 3] == vq[0] && Set_ind[numat * 3 + 1] == vq[2] && Set_ind[numat * 3 +2] == vq[2]){
                        exists = 1;
                    }
                }
                if (exists == 0) {
                    /*
                     * Set_ind = [Set_ind; vq];
                     */
                    Set_ind[(numat +1) * 3] = vq[0];
                    Set_ind[(numat +1) * 3 + 1] = vq[1];
                    Set_ind[(numat +1) * 3 + 1] = vq[2];
                    numat += 1;
                }
            }


            double cscra = get3DElement(cc, ccDim, q[1], q[2], q[3]);


            hnew3d(&cscra, cscraDim,
                       getCol(Vx, VxDim, q[0]), tempVxCol,
                       getCol(Vy, VyDim, q[1]), tempVyCol,
                       getCol(Vz, VzDim, q[2]), tempVzCol,
                       h_new, h_newDim);

            set3DElement(cp, cpDim, q[1], q[2], q[3], get3DElement(cp, cpDim, q[1], q[2], q[3]) + cscra);

            for(int k = 0; k < h_newDim[0] * h_newDim[1] * h_newDim[2]; k++) {
                h[k] += h_new[k];
                Re[k] -= h_new[k];
            }

            double nor_new = sumOfSquares(Re, ReDim) * delta;
            if (numat >= No || (nor_new < tol)) break;


        }

        // l = size(Set_ind, 1); Same as numat

        for (int n = 0; n < numat; n++) {
            c[n] = get3DElement(cp, cpDim, Set_ind[n * 3], Set_ind[n * 3 + 1], Set_ind[n * 3 + 2]);
        } // need to resize dimensions of c

        if (imp != 1) {
            delete [] Set_ind_trans;
            Set_ind_trans = new double[3 * numat];
            int Set_ind_trans_dim[] = {numat, 3, 1};
            int Set_ind_dim[] = {3, numat, 1};

            transpose(Set_ind, Set_ind_dim, Set_ind_trans, Set_ind_trans_dim);
            /*
            Di1 = Set_ind(:,1);
            Di2 = Set_ind(:,2);
            Di3 = Set_ind(:,3);
            */

            Di1 = getCol(Set_ind_trans, Set_ind_trans_dim, 0);
            Di2 = getCol(Set_ind_trans, Set_ind_trans_dim, 1);
            Di3 = getCol(Set_ind_trans, Set_ind_trans_dim, 2);

            double* VxTemp = new double[VxDim[0] * numat];
            double* VyTemp = new double[VyDim[0] * numat];
            double* VzTemp = new double[VzDim[0] * numat];

            int VxTempDim[] = {VxDim[0], numat, 1};
            int VyTempDim[] = {VyDim[0], numat, 1};
            int VzTempDim[] = {VzDim[0], numat, 1};

            for( int k = 0; k < numat; k++){
                for( int x = 0; x < VxDim[0]; x++) {
                    VxTemp[k * VxDim[0] + x] = getElement(Vx, VxDim, Di1[k], x);
                }
                for( int y = 0; y < VyDim[0]; y++) {
                    VyTemp[k * VyDim[0] + y] = getElement(Vy, VyDim, Di2[k], y);
                }
                for( int z = 0; z < VzDim[0]; z++) {
                    VzTemp[k * VzDim[0] + z] = getElement(Vz, VzDim, Di3[k], z);
                }
            }

            ProjMP3d(h, Re, ReDim, VxTemp, VxTempDim, VyTemp, VyTempDim, VzTemp, VzTempDim, c, cDim, toln, Maxp);
            int l = nonZeroNumel(c, VxDim[0] * VyDim[0] * VzDim[0]);

            for (int n = 0; n < l; n++) {
                set3DElement(cp, cpDim, Set_ind[n * 3], Set_ind[n * 3 +1], Set_ind[n * 3 + 2], c[n]);
            }
        }

        double nore = sumOfSquares(Re, ReDim) * delta;

        if (numat >= No || (nore < tol)) break;
    }
        if ((lstep != Max) && (it==Maxit2)){
            mexPrintf("%s Maximum number of iterations has been reached");
        }

}



void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {


    //todo validation
    double *f = mxGetPr(prhs[0]);
    double *vx = mxGetPr(prhs[1]);
    double *vy = mxGetPr(prhs[2]);
    double *vz = mxGetPr(prhs[3]);
    double tol = mxGetPr(prhs[4])[0];
    int No = (int)mxGetData(prhs[5])[0];
    double toln = mxGetPr(prhs[6])[0];
    int lstep = (int)mxGetData(prhs[7])[0];
    int Max = (int)mxGetData(prhs[8])[0];
    int Maxp = (int)mxGetData(prhs[9])[0];
    int* indx = (int*)mxGetData(prhs[10]);
    int* indy = (int*)mxGetData(prhs[10]);
    int* indz = (int*)mxGetData(prhs[10]);

    function[h,Set_ind,c]=SPMP3D(f,Vx,Vy,Vz,tol,No,toln,lstep,Max,Maxp,indx,indy,indz);
    m








}


