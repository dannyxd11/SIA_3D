#include "mex.h"
#include "commonOps.cpp"
#include "ProjMP3D.cpp"
#include "IP3D.cpp"
#include "hnew3D.cpp"
#include <cmath>

using namespace std;

int validateIndex(double indk, double *Di, int *DiDim) {
    for (int i = 0; i < DiDim[0] * DiDim[1] * DiDim[2]; i++) {
        if (Di[i] == indk) {
            return i;
        }
    }
    //mexErrMsgTxt("Index not in dictionary");
    return 0;
}


void SPMP3D(double* f, int* fDim, double* Vx, int* VxDim, double* Vy, int* VyDim, double* Vz, int* VzDim,
            double tol, double No, double toln, int lstep, int Max, int Maxp, int* indx, int* indy, int* indz,
            double* h, int* hDim, double* c, int* cDim, double* Set_ind, int* numat){

    for(int i = 0; i < VxDim[0] * VyDim[0] * VzDim[0]; i++){
        c[i] = 0;
    }

    /* MATLAB ≈
     *
     *  [Lx,Nx] = size(Vx);
     *  [Ly,Ny] = size(Vy);
     *  [Lz,Nz] = size(Vz);
     *  delta = 1 / (Lx*Ly*Lz);
     *  Nxyz = Lx*Ly*Lz;
     */

    int Lx = VxDim[0];
    int Nx = VxDim[1];
    int Ly = VyDim[0];
    int Ny = VyDim[1];
    int Lz = VzDim[0];
    int Nz = VzDim[1];

    int Nxyz = Lx * Ly * Lz;
    double delta = 1.0 / Nxyz;

    /*  MATLAB ≈
     *
     *      if sum(sum(sum(f).^2))*delta<1e-9;
     *          c=[];
     *          return
     *      end
     */

    if(sumOfSquares(f, fDim) * delta < 1e-9){
        c = new double[0];
        return;
    }


    //if (nargin<12) | (isempty(indy)==1)  indz=[];end
    //if (nargin<11) | (isempty(indy)==1)  indy=[];end
    //if (nargin<10) | (isempty(indx)==1)  indx=[];end
    //if (nargin<9) | (isempty(Maxp)==1)  Maxp=7000;end
    //if (nargin<8)  | (isempty(Max)==1)   Max=7000;end
    //if (nargin<7)  | (isempty(lstep)==1) lstep=0;end
    //if (nargin<6)  | (isempty(toln)==1)  tolnu=1e-3;end
    //if (nargin<5)  | (isempty(No)==1)    No=Nxyz;end
    //if (nargin<4)  | (isempty(tol)==1)   tol=5e-4*sum(sum(sum(abs(f).^2)))*delta;end;

    /* MATLAB ≈
     *
     *      cp=zeros(Nx,Ny,Nz);
     *      cc=zeros(Nx,Ny,Nz);
     */

    double* cc = new double[Nx * Ny * Nz]();
    int ccDim[] = {Nx, Ny, Nz};


    /* MATLAB ≈
     *
     *      Di1=[];
     *      Di2=[];
     *      Di3=[];
     *      numat=0;
     *      Set_ind=[];
     *      Dix=1:Nx;
     *      Diy=1:Ny;
     *      Diz=1:Nz;
     */

    double* Di1;
    double* Di2;
    double* Di3;
    numat[0] = 0;

    //Set_ind = new int[3 * Max]();

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

    /* MATLAB ≈
     *
     *      Re=f;
     *      tol2=1e-9;
     */

    double *Re = new double[fDim[0] * fDim[1] * fDim[2]];
    int *ReDim = new int[3];
    setDimensions(fDim, ReDim);
    memcpy(Re, f, ReDim[0] * ReDim[1] * ReDim[2] * sizeof(double));
    double tol2 = 1e-9;


    /*
     * MATLAB ≈
     *
     *      imp=0;
     *      if (lstep == -1) imp=1; end
     *      if (imp==1) lstep=0;end
     *      if (lstep == 0) Maxit2=1; lstep=Max;
     *      else Maxit2=Max/lstep; end
     */

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
     *  Constant depending on dictionary size, move allocation outside loop to reduce unnecessary overhead
     */

    int tempVxCol[] = {VxDim[0], 1, 1};
    int tempVyCol[] = {VyDim[0], 1, 1};
    int tempVzCol[] = {VzDim[0], 1, 1};
    int h_newDim[] = {VxDim[0], VyDim[0], VzDim[0]};
    int cscraDim[]= {1,1,1};
    double *h_new = new double[h_newDim[0] * h_newDim[1] * h_newDim[2]];
    int *q = new int[3];
    double* Set_ind_trans = new double[3];
    double* plane;
    double* col;
    double* Row = new double[VxDim[0]];
    int ColDim[] = {VxDim[0], 1, 1};
    int RowDim[] = {1, VxDim[0], 1};
    int planeDim[] = {ReDim[0], ReDim[1], 1};
    double* multResult1 = new double[VxDim[0]];
    int multResult1Dim[] = {1, VxDim[0], 1};



    /*
     * MATLAB ≈
     *
     *      for it=1:Maxit2;
     *          for s=1:lstep;
     *              if (numat+1)<=numind
     */

    int it;
    for (it = 0; it < Maxit2; it++) {
        for (int s = 0; s < lstep; s++) {
            if ((numat[0] + 1) <= numind) {

                /*
                 * MATLAB ≈
                 *
                 *          [testx,qx]=ismember(indx(numat+1),Dix);
                 *          [testy,qy]=ismember(indy(numat+1),Diy);
                 *          [testz,qz]=ismember(indz(numat+1),Diz);
                 *          if testx ~=1  error('Demanded index (x) %d is out of dictionary',indx(numat+1));end
                 *          if testy ~=1  error('Demanded index (y) %d is out of dictionary',indy(numat+1));end
                 *          if testz ~=1  error('Demanded index (z) %d is out of dictionary',indz(numat+1));end
                 */

                validateIndex(indx[numat[0] + 1], Dix, DixDim);
                validateIndex(indy[numat[0] + 1], Diy, DiyDim);
                validateIndex(indz[numat[0] + 1], Diz, DizDim);


                /*
                 * MATLAB ≈
                 *
                 *            q(1)=indx(numat+1);
                 *            q(2)=indy(numat+1);
                 *            q(3)=indz(numat+1);
                 */

                q[0] = indx[numat[0] + 1];
                q[1] = indy[numat[0] + 1];
                q[2] = indz[numat[0] + 1];


                /*
                 * MATLAB ≈
                 *
                 *      cc(q(1),q(2),q(3))=0;
                 *      for zk=1:Lz;
                 *          cc(q(1),q(2),q(3))= cc(q(1),q(2),q(3))+Vx(:,q(1))'*Re(:,:,zk)*Vy(:,q(2))*Vz(zk,q(3));
                 *      end
                 */

                set3DElement(cc, ccDim, q[0], q[1], q[2], 0) ;
                for (int zk = 0; zk < Lz; zk++) {

                    MatrixMultiplyBLAS(getCol(Vx,VxDim,q[0]), getPlane(Re, ReDim, zk), multResult1, VxDim[0], 1, multResult1[1], 'T', 'N');
                    MatrixMultiplyBLAS(multResult1, getCol(Vy, VyDim, q[1]), &cc[q[2] * ccDim[0] * ccDim[1] + q[1] * ccDim[0] + q[0]], multResult1Dim[0], multResult1Dim[1], 1, 'N', 'N', getElement(Vz, VzDim, q[2], zk), 1);

                }
            } else {

                /*
                 * MATLAB ≈
                 *
                 *        [cc] = IP3D_mex(Re,Vx,Vy,Vz);
                 *        [max_c,maxind] = max(abs(cc(:)));
                 *        [q(1),q(2),q(3)] = ind2sub(size(cc),maxind);
                 *
                 */

                IP3d(Re, ReDim, Vx, VxDim, Vy, VyDim, Vz, VzDim, cc, ccDim);
                int maxind = max(cc, ccDim);
                ind2sub(ccDim, maxind, q);

                /*
                 * MATLAB ≈
                 *          if max_c < tol2
                 *              fprintf('%s stopped, max(|<f,D>|)<= tol2=%g.\n',name,tol2);
                 *              return;
                 *          end
                 */

                if (abs(cc[maxind]) < tol2) {
                    mexPrintf("SPMP3D stopped, max(|<f,q|/||q||) <= tol2 ");

                    /* Clean up memory
                     *
                     *
                     */
                    delete [] Row;
                    delete [] multResult1;

                    delete [] cc;

                    delete [] Dix;
                    delete [] Diy;
                    delete [] Diz;

                    delete [] DixDim;
                    delete [] DiyDim;
                    delete [] DizDim;

                    delete [] Re;
                    delete [] ReDim;

                    delete [] h_new;
                    delete [] q;

                    delete [] Set_ind_trans;

                    return;
                }
            }

            /*
             * MATLAB ≈
             *
             *      vq=[q(1),q(2),q(3)];
             *      cscra=cc(q(1),q(2),q(3));
             *      
             *      if(isempty(Set_ind)==1)
             *      Set_ind=vq;
             *      numat=1;
             *      cscra=cc(q(1),q(2),q(3));
             */

            
            int vq[] = {q[0], q[1], q[2]};
            double cscra = get3DElement(cc, ccDim, q[0], q[1], q[2]);

            if (numat[0] == 0) {
                Set_ind[0] = vq[0];
                Set_ind[1] = vq[1];
                Set_ind[2] = vq[2]; 
                c[numat[0]] = cscra;
                numat[0] += 1;
            } else {

                /*
                 * MATLAB ≈
                 *
                 *      [testq1, indq1] = ismember(vq, Set_ind, 'rows');
                 */
            	
                int exists = 0;
                int index = 0;
                for(int k = 0; k < numat[0]; k++){
                    if(Set_ind[k * 3] == vq[0] && Set_ind[k * 3 + 1] == vq[1] && Set_ind[k * 3 +2] == vq[2]){
                        exists = 1;
                        index = k;
                    }
                }

                /*
                 * MATLAB ≈
                 *
                 *         if testq1==0
                 *         Set_ind =[Set_ind;vq];
                 *         numat=numat+1;
                 */
                if (exists == 0) {
                    Set_ind[(numat[0]) * 3] = vq[0];
                    Set_ind[(numat[0]) * 3 + 1] = vq[1];
                    Set_ind[(numat[0]) * 3 + 2] = vq[2];
                    c[numat[0]] = cscra;
                    numat[0] += 1;
                }else{
                	c[index] += cscra;                    
                }
            }

            /*
             * MATLAB ≈
             *
             *         h_new = hnew3D_mex(cscra, Vx(:,q(1)), Vy(:,q(2)), Vz(:, q(3)));
             *         cp(q(1),q(2),q(3))=cp(q(1),q(2),q(3))+cscra;  %add coefficients of identical atoms
             */




            hnew3d(&cscra, cscraDim,
                   getCol(Vx, VxDim, q[0]), tempVxCol,
                   getCol(Vy, VyDim, q[1]), tempVyCol,
                   getCol(Vz, VzDim, q[2]), tempVzCol,
                   h_new, h_newDim);

            //set3DElement(cp, cpDim, q[0], q[1], q[2], get3DElement(cp, cpDim, q[0], q[1], q[2]) + cscra);

            /*
             * MATLAB ≈
             *
             *       h=h+h_new; %Approximated Image
             *       Re=Re-h_new;%
             */

            for(int k = 0; k < h_newDim[0] * h_newDim[1] * h_newDim[2]; k++) {
                h[k] += h_new[k];
                Re[k] -= h_new[k];
            }

            /*
             * MATLAB ≈
             *
             *       nor_new=sum(sum(sum(abs(Re).^2)))*delta;
             *       if (numat>=No | (nor_new < tol)) break;end;
             */

            double nor_new = sumOfSquares(Re, ReDim) * delta;
            if (numat[0] >= No || (nor_new < tol)) break;


        }


        /*
         * l = size(Set_ind, 1); Unnecessary since same as numat
         *
         * MATLAB  ≈test_SPMP3D.cu(87): error
         *
         *      for n=1:l;
         *           c(n)=cp(Set_ind(n,1),Set_ind(n,2),Set_ind(n,3));
         *      end
         */

//        for (int n = 0; n < numat[0]; n++) {
//            c[n] = get3DElement(cp, cpDim, Set_ind[n * 3], Set_ind[n * 3 + 1], Set_ind[n * 3 + 2]);
//        }
        setDimensions(1, numat[0], 1, cDim); //todo need to resize dimensions of c

        /*
         * MATLAB ≈ if imp ~= 1
         */

        if (imp != 1) {
            /* Set_ind_trans is reused, delete prior data before reusing. */

            delete [] Set_ind_trans;
            Set_ind_trans = new double[3 * numat[0]];
            int Set_ind_trans_dim[] = {numat[0], 3, 1};
            int Set_ind_dim[] = {3, numat[0], 1};


            /* MATLAB implementation stores Set_ind as [x1, y1, z1; x2, y2, z2 ....]
             * C++ implementation stores Set_ind as [x1, x2 .... ; y1, y2 .... ; z1, z2 ....] since it is stored column
             * major and no need for resizing. Hence transpose for equivilent
             */

            transpose(Set_ind, Set_ind_dim, Set_ind_trans, Set_ind_trans_dim);

            /*
             * MATLAB ≈
             *
             *      Di1=Set_ind(:,1);
             *      Di2=Set_ind(:,2);
             *      Di3=Set_ind(:,3);
             */


            Di1 = getCol(Set_ind_trans, Set_ind_trans_dim, 0);
            Di2 = getCol(Set_ind_trans, Set_ind_trans_dim, 1);
            Di3 = getCol(Set_ind_trans, Set_ind_trans_dim, 2);

            double* VxTemp = new double[VxDim[0] * numat[0]];
            double* VyTemp = new double[VyDim[0] * numat[0]];
            double* VzTemp = new double[VzDim[0] * numat[0]];

            int VxTempDim[] = {VxDim[0], numat[0], 1};
            int VyTempDim[] = {VyDim[0], numat[0], 1};
            int VzTempDim[] = {VzDim[0], numat[0], 1};

            for( int k = 0; k < numat[0]; k++){
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




            /*
             * MATLAB ≈
             *
             *      [h,Re,c]=ProjMP3D(h,Re,Vx(:,Di1),Vy(:,Di2),Vz(:,Di3),c,toln,Maxp);
             *      l=numel(c);
             */

            ProjMP3d(h, Re, ReDim, VxTemp, VxTempDim, VyTemp, VyTempDim, VzTemp, VzTempDim, c, cDim, toln, Maxp);
            int l = nonZeroNumel(c, VxDim[0] * VyDim[0] * VzDim[0]);

            /*
             * MATLAB ≈
             *
             *      for n=1:l;
             *          cp(Set_ind(n,1),Set_indtest_SPMP3D.cu(87): error(n,2),Set_ind(n,3))=c(n);
             *      end
             *
             */

//            for (int n = 0; n < l; n++) {
//                set3DElement(cp, cpDim, Set_ind[n * 3], Set_ind[n * 3 +1], Set_ind[n * 3 + 2], c[n]);
//            }

            delete [] VxTemp;
            delete [] VyTemp;//h_Matrix h_cc(ccElements, 40, 40, 15);
            delete [] VzTemp;
        }

        /*
         * MATLAB ≈
         *
         *      nore=sum(sum(sum(abs(Re).^2)))*delta;
         *      if (numat>=No | (nore < tol)) break;end;
         */

        double nore = sumOfSquares(Re, ReDim) * delta;
        if (numat[0] >= No || (nore < tol)) break;
    }


    /*
     * MATLAB ≈ if (lstep ~=Max) & (it==Maxit2) fprintf('%s Maximum number of iterations has been reached\n',name);
     */

    if ((lstep != Max) && (it==Maxit2)){
        mexPrintf("Maximum number of iterations has been reached");
    }


    /*
     * Clean up Memory
     */

    delete [] Row;
    delete [] multResult1;

    //delete [] cp;
    delete [] cc;

    delete [] Dix;
    delete [] Diy;
    delete [] Diz;

    delete [] DixDim;
    delete [] DiyDim;
    delete [] DizDim;

    delete [] Re;
    delete [] ReDim;

    delete [] h_new;
    delete [] q;
    delete [] Set_ind_trans;
}


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    //
    //todo validation
    if (nrhs < 10){ mexErrMsgTxt("Not enough input arguments: <fs, Dx, Dy, Dz, tol, No, toln, lstep, Max, Maxp>"); return;}

    double *f = mxGetPr(prhs[0]);
    double *Vx = mxGetPr(prhs[1]);
    double *Vy = mxGetPr(prhs[2]);
    double *Vz = mxGetPr(prhs[3]);
    double tol = mxGetPr(prhs[4])[0];
    double No = mxGetPr(prhs[5])[0];
    double toln = mxGetPr(prhs[6])[0];
    int lstep = (int)(mxGetPr(prhs[7])[0]);
    int Max = (int)(mxGetPr(prhs[8])[0]);
    int Maxp = (int)(mxGetPr(prhs[9])[0]);

    //todo indx,indy,indz not yet supported.
    int *indx;
    int *indy;
    int *indz;

    if (nrhs > 10 &&  nrhs == 13) {
        indx = (int *) mxGetData(prhs[10]);
        indy = (int *) mxGetData(prhs[11]);
        indz = (int *) mxGetData(prhs[12]);
    }else if(nrhs > 10 && nrhs != 13){ mexErrMsgTxt("Not enough input arguments: <fs, Dx, Dy, Dz, tol, No, toln, lstep, Max, Maxp, indx, indy, indz>\nCustom Indexing not yet supported"); return; }
    int fDim[] =  {(int)mxGetDimensions(prhs[0])[0], (int)mxGetDimensions(prhs[0])[1], (int)mxGetDimensions(prhs[0])[2]};
    int VxDim[] = {(int)mxGetDimensions(prhs[1])[0], (int)mxGetDimensions(prhs[1])[1], (int)mxGetDimensions(prhs[1])[2]};
    int VyDim[] = {(int)mxGetDimensions(prhs[2])[0], (int)mxGetDimensions(prhs[2])[1], (int)mxGetDimensions(prhs[2])[2]};
    int VzDim[] = {(int)mxGetDimensions(prhs[3])[0], (int)mxGetDimensions(prhs[3])[1], (int)mxGetDimensions(prhs[3])[2]};


    int hDims = 3; int hDim[] = {VxDim[0], VyDim[0], VzDim[0]};
    size_t hDimMxArray[3] = {static_cast<size_t>(hDim[0]), static_cast<size_t>(hDim[1]), static_cast<size_t>(hDim[2])};
    plhs[0] = mxCreateNumericArray(hDims, hDimMxArray, mxDOUBLE_CLASS, mxREAL);
    double* h = mxGetPr(plhs[0]);



    double* Set_ind;
    double* c = new double[VxDim[0] * VyDim[0] * VzDim[0]];
    int cDim[3] = {VxDim[0], VyDim[0], VzDim[0]};
    int numat = 0;
	int cDims = 3;
	Set_ind = new double[3 * Max]();
        

	SPMP3D(f, fDim, Vx, VxDim, Vy, VyDim, Vz, VzDim, tol, No, toln, lstep, Max, Maxp, indx, indy, indz, h, hDim, c, cDim, Set_ind, &numat);

    int Set_ind_dims = 3; 

    // Dimensions for mxArray need to be size_t
    size_t Set_ind_dim[3] = {static_cast<size_t>(numat), 3, 1};
    size_t cDimMxArray[3] = {static_cast<size_t>(cDim[0]), static_cast<size_t>(cDim[1]), static_cast<size_t>(cDim[2])};

    // Create set_ind and c to be returned to MATLAB
    plhs[2] = mxCreateNumericArray(cDims, cDimMxArray, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(Set_ind_dims, Set_ind_dim, mxDOUBLE_CLASS, mxREAL);

    double* tempSetT = new double[numat * 3];
    int tempSetDimT[] = {numat, 3, 1};
    int tempSetDim[] = {3, numat, 1};

    for(int i = 0; i < numat * 3; i++){
        Set_ind[i] += 1;
    }

    transpose(Set_ind, tempSetDim, tempSetT, tempSetDimT);


    // Correct Indexing before returning to MATLAB
    memcpy(mxGetPr(plhs[2]), c, cDim[0] * cDim[1] * cDim[2] * sizeof(double));
    memcpy(mxGetPr(plhs[1]), tempSetT, (numat) * 3 * sizeof(double));

    delete [] c;
    delete [] Set_ind;
	delete [] tempSetT;
}


