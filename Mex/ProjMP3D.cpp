#ifndef PROJMP3D
#define PROJMP3D

#include "commonOps.cpp"
#include "IPSP3D.cpp"	
#include "hnew3D.cpp"
#include "mex.h"
#include <cmath>

using namespace std;

void ProjMP3d(double* h, double* re, int* reDim, double* v1, int* v1Dim, double* v2, int* v2Dim, double* v3, int* v3Dim, double* c, int* cDim, double toln, double max){

double delta 	= 	1 / (double)(v1Dim[0] * v2Dim[0] * v3Dim[0]);
double tol2		=	1e-11;

int* hnewDim = new int[3];
setDimensions(v1Dim[0], v2Dim[0], v3Dim[0], hnewDim);
double* hnew = new double[hnewDim[0] * hnewDim[1] * hnewDim[2]];	

int* v1ColDim = new int[3];
int* v2ColDim = new int[3]; 
int* v3ColDim = new int[3];
setDimensions(v1Dim[0], 1, 1, v1ColDim);
setDimensions(v2Dim[0], 1, 1, v2ColDim);
setDimensions(v3Dim[0], 1, 1, v3ColDim);

int* ccDim = new int[3];
setDimensions(1, v1Dim[1], 1, ccDim);



for(int it = 0; it < max; it++){
    double* cc = new double[ccDim[0] * ccDim[1]]();
	IPSP3d(re, reDim, v1, v1Dim, v2, v2Dim, v3, v3Dim, cc, ccDim);



	double maxValue = std::abs(cc[0]);
	int n1 = 0;
	for (int k = 0; k < ccDim[1]; k++){
		if(std::abs(cc[k]) > maxValue){
			maxValue = std::abs(cc[k]);
			n1 = k;
		}
	}

	if (maxValue < tol2){ break; }

	hnew3d(&cc[n1], ccDim, getCol(v1, v1Dim, n1), v1ColDim, getCol(v2, v2Dim, n1), v2ColDim, getCol(v3, v3Dim, n1), v3ColDim,  hnew, hnewDim);
	c[n1] += cc[n1];
    delete [] cc;
	double nornu = 0;

	for(int k = 0; k < reDim[0] * reDim[1] * reDim[2]; k++){
		h[k]  += hnew[k];
		re[k] -= hnew[k];
		nornu += hnew[k] * hnew[k];
	}
	nornu *= delta;
	if (nornu <= toln){ break; }

}


delete [] ccDim;
delete [] hnewDim;
delete [] hnew;
delete [] v1ColDim;
delete [] v2ColDim;
delete [] v3ColDim;
}

#endif