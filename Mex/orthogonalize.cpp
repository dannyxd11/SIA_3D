//
// Created by Dan on 12/12/2016.
//
#include "mex.h"
#include "commonOps.cpp"

/*
 * k=size(Q,2) +1;
 *    for p=1:k -1
 *        % Orthogonalization
 *
 *        Q(:,k) =    new_atom    -   (Q(:,p)' * new_atom)  *   Q(:,p);
 *     end
 *
 */

void orthogonalize(double* Q, int* QDim, double* newAtom, int* newAtomDim){

    // Set K as width + 1 (Size is increasing)
    int k = QDim[1] + 1;
    int* origDim = new int[3];
    setDimensions(QDim, origDim);
    QDim[1] += 1;


    double multResult1 = 0;
    int* multResult1Dim = new int[3];
    setDimensions(1,1,1, multResult1Dim);



    for(int p = 0; p < k - 1; p++){
        int* tempColDim = new int[3];
        setDimensions(QDim[0], 1, 1,tempColDim);

        double* tempCol; // = new double[tempColDim[0]];
        tempCol = getCol(Q, origDim, p);


        double* colT = new double[tempColDim[0] * tempColDim[1]];
        int* colTDim = new int[3];
        setDimensions(1, QDim[1], 1, colTDim);
        transpose(tempCol, tempColDim, colT, colTDim);

        blasMultiply(colT, colTDim, newAtom, newAtomDim, &multResult1, multResult1Dim);

        double* col = getCol(Q, QDim, p);


        for (int n = 0; n < QDim[0]; n++){
            setElement(Q, QDim, k-1, n, newAtom[n] - (multResult1 * col[n]));
	}

        delete [] colT;
        delete [] colTDim;
        delete [] tempColDim;
        //delete [] tempCol;
    }

    delete [] origDim;
    delete [] multResult1Dim;








}







void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	int* QDim = new int[3];
	setDimensions(mxGetDimensions(prhs[0])[0], mxGetDimensions(prhs[0])[1], 1, QDim);

	int* newAtomDim = new int[3];
	setDimensions(mxGetDimensions(prhs[1])[0], mxGetDimensions(prhs[1])[1], 1, newAtomDim);
	double* newAtom = mxGetPr(prhs[1]);

    nlhs = 1;

//    plhs[0] = mxCreateDoubleMatrix(QDim[0], QDim[1] + 1, mxREAL);
//
//	double* Q = (double*)mxCalloc(QDim[0] * (QDim[1] + 1), sizeof(double));
//	orthogonalize(Q, QDim, newAtom, newAtomDim);
//	mxSetPr(plhs[0], Q);


    plhs[0] = mxDuplicateArray(prhs[0]);
    double* Q;
    double *ptr;
    double *newptr;
    size_t nbytes =  QDim[0] * (QDim[1]+1) * sizeof(*ptr);

    ptr = mxGetPr(plhs[0]);
    newptr = (double*)mxRealloc(ptr, nbytes);
    mxSetPr(plhs[0], newptr);
    Q = mxGetPr(plhs[0]);
    mxSetN(plhs[0],QDim[1] + 1);

    orthogonalize(Q, QDim, newAtom, newAtomDim);



	delete [] QDim;
	delete [] newAtomDim;







}

