#ifndef IPSP3D
#define IPSP3D
#define BLOCK_WIDTH 8
#define SINGLE_THREAD if(threadIdx.x == 0 && threadIdx.y == 0)

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include <stdio.h>
#include "h_Matrix.h"

#ifndef CUDACHECK
#define CUDACHECK
#define cudaCheck(input){cudaAssert((input), __FILE__, __LINE__); } // http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void cudaAssert(cudaError_t code, const char *file, int line){
	if (code != cudaSuccess){ fprintf(stderr, "CudaAssert %s %s %d\n", cudaGetErrorString(code), file, line); exit(code);}
}
#endif































__device__ void d_ProjMP3d(h_Matrix* h, h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* c, double* toln, int* max){

	double delta 	= 	1 / (double)(v1->height * v2->height * v3->height);
	double tol2		=	1e-11;
	__shared__ h_Matrix* hnew;
	__shared__ h_Matrix* cc;
	__shared__ double* hnewElements;
	__shared__ double* ccElements;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		hnew = new h_Matrix(v1->height, v2->height, v3->height);
		hnewElements = new double[hnew->numel()]();
		hnew->elements = hnewElements;
		cc = new h_Matrix(1, v1->width, 1);
	}
	__syncthreads();

	for(int it = 0; it < max[0]; it++){
		SINGLE_THREAD{
			if(!ccElements){ delete [] ccElements; };
			ccElements = new double[cc->numel()]();
			cc->elements = ccElements;

			for(int i = 0; i < re->numel(); i++){
				printf("%f, ", re->elements[i]);
			}
		}

		__syncthreads();
		d_IPSP3d( re, v1, v2, v3, cc );
		__syncthreads();

		__shared__ int maxInd;
		__shared__ double maxVal;

		SINGLE_THREAD{
			maxInd = 0;
			maxVal = 0.0;
		}

		d_findMaxInd(cc, &maxInd, &maxVal);
		__syncthreads();
		printf("%f\n", maxVal);
		if (abs(maxVal) < tol2){ SINGLE_THREAD{printf("breaking maxVal- %g < %g, %d, %d, %d\n", maxVal, tol2, it, threadIdx.x, threadIdx.y);} break; }

		h_Matrix v1Col = v1->getCol(maxInd);
		h_Matrix v2Col = v2->getCol(maxInd);
		h_Matrix v3Col = v3->getCol(maxInd);

		__syncthreads();

		d_hnew3d(maxVal, &v1Col, &v2Col, &v3Col, hnew);
		c->elements[maxInd] += cc->elements[maxInd];

		__syncthreads();

		__shared__ double nornu;
		if(threadIdx.x == 0 && threadIdx.y == 0){
			nornu = 0;
			delete [] ccElements;
		}

		d_matrixAddition(h, hnew, h);
		d_matrixSubtraction(re, hnew, re);
		d_squaredSum(hnew, &nornu);

	__syncthreads();

	if (nornu * delta <= toln[0]){ SINGLE_THREAD{printf("nornu break");}; break; }

	}
}

__global__ void d_ProjMP3dKernel(h_Matrix* h, h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* c, double* toln, int* max){
	d_ProjMP3d(h, re, v1, v2, v3, c, toln, max);
}




//int main() {
//
//	double dxElements[] = {0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.4976, 0.4785, 0.4410, 0.3865, 0.3172, 0.2357, 0.1451, 0.0490, 0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904, 0.4785, 0.3172, 0.0490, -0.2357, -0.4410, -0.4976, -0.3865, -0.1451, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619, 0.4410, 0.0490, -0.3865, -0.4785, -0.1451, 0.3172, 0.4976, 0.2357, 0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157, 0.3865, -0.2357, -0.4785, 0.0490, 0.4976, 0.1451, -0.4410, -0.3172, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3172, -0.4410, -0.1451, 0.4976, -0.0490, -0.4785, 0.2357, 0.3865, 0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778, 0.2357, -0.4976, 0.3172, 0.1451, -0.4785, 0.3865, 0.0490, -0.4410, 0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1451, -0.3865, 0.4976, -0.4410, 0.2357, 0.0490, -0.3172, 0.4785, 0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975, 0.0490, -0.1451, 0.2357, -0.3172, 0.3865, -0.4410, 0.4785, -0.4976, 0.0490, 0.1451, 0.2357, 0.3172, 0.3865, 0.4410, 0.4785, 0.4976, 0.0975, 0.2778, 0.4157, 0.4904, 0.4904, 0.4157, 0.2778, 0.0975, 0.1451, 0.3865, 0.4976, 0.4410, 0.2357, -0.0490, -0.3172, -0.4785, 0.1913, 0.4619, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.2357, 0.4976, 0.3172, -0.1451, -0.4785, -0.3865, 0.0490, 0.4410, 0.2778, 0.4904, 0.0975, -0.4157, -0.4157, 0.0975, 0.4904, 0.2778, 0.3172, 0.4410, -0.1451, -0.4976, -0.0490, 0.4785, 0.2357, -0.3865, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3865, 0.2357, -0.4785, -0.0490, 0.4976, -0.1451, -0.4410, 0.3172, 0.4157, 0.0975, -0.4904, 0.2778, 0.2778, -0.4904, 0.0975, 0.4157, 0.4410, -0.0490, -0.3865, 0.4785, -0.1451, -0.3172, 0.4976, -0.2357, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1913, -0.4619, 0.4785, -0.3172, 0.0490, 0.2357, -0.4410, 0.4976, -0.3865, 0.1451, 0.4904, -0.4157, 0.2778, -0.0975, -0.0975, 0.2778, -0.4157, 0.4904, 0.4976, -0.4785, 0.4410, -0.3865, 0.3172, -0.2357, 0.1451, -0.0490, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000};
//	double dyElements[] = {0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.4976, 0.4785, 0.4410, 0.3865, 0.3172, 0.2357, 0.1451, 0.0490, 0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904, 0.4785, 0.3172, 0.0490, -0.2357, -0.4410, -0.4976, -0.3865, -0.1451, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619, 0.4410, 0.0490, -0.3865, -0.4785, -0.1451, 0.3172, 0.4976, 0.2357, 0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157, 0.3865, -0.2357, -0.4785, 0.0490, 0.4976, 0.1451, -0.4410, -0.3172, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3172, -0.4410, -0.1451, 0.4976, -0.0490, -0.4785, 0.2357, 0.3865, 0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778, 0.2357, -0.4976, 0.3172, 0.1451, -0.4785, 0.3865, 0.0490, -0.4410, 0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1451, -0.3865, 0.4976, -0.4410, 0.2357, 0.0490, -0.3172, 0.4785, 0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975, 0.0490, -0.1451, 0.2357, -0.3172, 0.3865, -0.4410, 0.4785, -0.4976, 0.0490, 0.1451, 0.2357, 0.3172, 0.3865, 0.4410, 0.4785, 0.4976, 0.0975, 0.2778, 0.4157, 0.4904, 0.4904, 0.4157, 0.2778, 0.0975, 0.1451, 0.3865, 0.4976, 0.4410, 0.2357, -0.0490, -0.3172, -0.4785, 0.1913, 0.4619, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.2357, 0.4976, 0.3172, -0.1451, -0.4785, -0.3865, 0.0490, 0.4410, 0.2778, 0.4904, 0.0975, -0.4157, -0.4157, 0.0975, 0.4904, 0.2778, 0.3172, 0.4410, -0.1451, -0.4976, -0.0490, 0.4785, 0.2357, -0.3865, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3865, 0.2357, -0.4785, -0.0490, 0.4976, -0.1451, -0.4410, 0.3172, 0.4157, 0.0975, -0.4904, 0.2778, 0.2778, -0.4904, 0.0975, 0.4157, 0.4410, -0.0490, -0.3865, 0.4785, -0.1451, -0.3172, 0.4976, -0.2357, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1913, -0.4619, 0.4785, -0.3172, 0.0490, 0.2357, -0.4410, 0.4976, -0.3865, 0.1451, 0.4904, -0.4157, 0.2778, -0.0975, -0.0975, 0.2778, -0.4157, 0.4904, 0.4976, -0.4785, 0.4410, -0.3865, 0.3172, -0.2357, 0.1451, -0.0490, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000};
//	double dzElements[] = {0.5774,0.5774,0.5774,0.7887,0.5774,0.2113,0.7071,0.0000,-0.7071,0.5774,-0.5774,-0.5774,0.4082,-0.8165,0.4082,0.2113,-0.5774,0.7887,0.2113,0.5774,0.7887,0.4082,0.8165,0.4082,0.5774,0.5774,-0.5774,0.7071,0.0000,-0.7071,0.7887,-0.5774,0.2113,0.5774,-0.5774,0.5774,1.0000,0,0,0,1.0000,0,0,0,1.0000};
//	double reElements[] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6};
//	double hElements[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
//	//double* ccElements = new double[ 40 * 40 * 15]();
//	double ccElements[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};
//
//	printDeviceDetails();
//
//	//Initialise matrix containers on host
//	std::cout << "Initializing.." << std::endl;
//	h_Matrix h_dx(dxElements, 8, 1, 1);
//	h_Matrix h_dy(dyElements, 8, 1, 1);
//	h_Matrix h_dz(dzElements, 3, 1, 1);
//	h_Matrix h_re(reElements, 8, 8, 3);
//	h_Matrix h_h(hElements, 8, 8, 3);
//	//h_Matrix h_cc(ccElements, 40, 40, 15);
//	h_Matrix h_cc(ccElements, 1, 40, 1);
//
//	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*32);
//
//	//Copy Matrix'sglobal to device
//	std::cout << "Copying th_newo Device.." << std::endl;
//	h_Matrix* d_dx = copyMatrixToDevice(&h_dx);
//	h_Matrix* d_dy = copyMatrixToDevice(&h_dy);
//	h_Matrix* d_dz = copyMatrixToDevice(&h_dz);
//	h_Matrix* d_re = copyMatrixToDevice(&h_re);
//	h_Matrix* d_cc = copyMatrixToDevice(&h_cc);
//	h_Matrix* d_h = copyMatrixToDevice(&h_h);
//
//	std::cout << "Starting.." <<std::endl;
//
//	double* cElements = new double[h_re.width]();
//	h_Matrix h_c(cElements, 1, 8, 1);
//	h_Matrix* d_c = copyMatrixToDevice(&h_c);
//
//
//	dim3 threadsPerBlock(8, 8);
//	// Input needs to be colsd_ProjMP3dKernel
//	//d_IPSP3d<<< 1, threadsPerBlock>>>(d_re, d_dx, d_dy, d_dz, d_cc);
//	//d_IP3d<<< 1, threadsPerBlock>>>(d_re, d_dx, d_dy, d_dz, d_cc);
//
//	//int* d_ind;
//	double* d_val;
//
//	//cudaCheck( cudaMalloc( &d_ind, sizeof(int) ) );
//	cudaCheck( cudaMalloc( &d_val, sizeof(double) ) );
//
//	//d_findMaxIndKernel<<< 1, threadsPerBlock>>>(d_re, d_ind, d_val);
//
//	h_Matrix calc(8,8,3);
//	h_Matrix* d_calc = copyMatrixToDevice(&calc);
//
//	double toln = 1e-8;
//	int max = 50000;
//
//	double* d_toln;
//	int* d_max;
//
//	cudaCheck( cudaMalloc( &d_toln, sizeof(double) ) );
//	cudaCheck( cudaMalloc( &d_max, sizeof(int) ) );
//
//	cudaCheck( cudaMemcpy( d_toln, &toln, sizeof(double), cudaMemcpyHostToDevice));
//	cudaCheck( cudaMemcpy( d_max, &max, sizeof(int), cudaMemcpyHostToDevice));
//
//	//d_SPMP3D(h_Matrix* f, h_Matrix* Vx, h_Matrix* Vy, h_Matrix* Vz,
//	//						double* tolPtr, double* NoPtr, double* tolnPtr, int* lstepPtr, int* MaxPtr, int* MaxpPtr,
//	//						h_Matrix* h, h_Matrix* c, double* Set_ind, numat){
//
//
//	//d_matrixSubtractionKernel<<< 1, threadsPerBlock>>>(d_re, d_re, d_calc);
//	//d_hnew3dKernel<<< 1, threadsPerBlock>>>(d_toln, d_dx, d_dy, d_dz, d_h);
//	//d_ProjMP3dKernel<<< 1, threadsPerBlock>>>(d_h, d_re, d_dx, d_dy, d_dz, d_cc, d_toln, d_max);
//
//	cudaCheck(cudaDeviceSynchronize());
//	//matrixMultiplyCudaKernel<<<1,threadsPerBlock>>>(d_dx, d_re, d_c, 1, h_re.width, h_dx.height, CUBLAS_OP_T, CUBLAS_OP_N, devScalar);
//
//	// Either works :)
//	//    h_Matrix results = 	copyMatrixToHost(d_cc);
//	//copyMatrixToHost(&h_cc, d_cc);
//	//printf("%d,%d,%d, %f\n", h_cc.height, h_cc.width, h_cc.depth, h_cc.elements[0]);
//
////	for(int i = 0; i < h_cc.height * h_cc.width * h_cc.depth; i++){
////		std::cout << h_cc.elements[i] << ", " << std::endl;
////	}
//
//	int ind;
//	double val;//	for(int i = 0; i < h_re.numel(); i++){
//	//		std::cout << h_re.elements[i] << std::endl;
//	//	}
//
//	//cudaCheck( cudaMemcpy( &ind, d_ind, sizeof(int), cudaMemcpyDeviceToHost));
//	//cudaCheck( cudaMemcpy( &val, d_val, sizeof(double), cudaMemcpyDeviceToHost));
//
//	copyMatrixToHost(&h_h, d_h);
//
////	for(int i = 0; i < h_h.numel(); i++){
////		std::cout << h_h.elements[i] << std::endl;
////	}
//
//	//std::cout << std::endl <<  val << std::endl;
//
//	cudaError_t err = cudaGetLastError();
//
//	printf("\nError: %s\n", cudaGetErrorString(err));
//
//	cudaFree(d_dx);
//	cudaFree(d_dy);
//	cudaFree(d_dz);
//	cudaFree(d_re);
//	cudaFree(d_cc);
//	return 0;
//}

#endif

