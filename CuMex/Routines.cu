#ifndef ROUTINES
#define ROUTINES

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <stdio.h>
#include "h_Matrix.h"
#include "common_ops.cu"
#include "cuda_helpers.cu"

#define BLOCK_WIDTH 8
#define SINGLE_THREAD if(threadIdx.x == 0 && threadIdx.y == 0)

#ifndef CUDACHECK
#define CUDACHECK
#define cudaCheck(input){cudaAssert((input), __FILE__, __LINE__); } // http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void cudaAssert(cudaError_t code, const char *file, int line){
	if (code != cudaSuccess){ fprintf(stderr, "CudaAssert %s %s %d\n", cudaGetErrorString(code), file, line); exit(code);}
}
#endif

__device__ void d_IPSP3d(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){
	double zero = 0;
	double scalar = 1;
	__shared__ int n1, l3;
	__shared__ h_Matrix aMatrix;
	__shared__ double *aMatrixElements;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		n1 = v1->width;
		l3 = v3->height;
		aMatrixElements = new double[re->width]();
		aMatrix.elements = aMatrixElements;
		aMatrix.width = v1->height; aMatrix.height = aMatrix.depth = 1;
	}
	__syncthreads();

	for(int i = 0; i < n1; i++){
		for(int j = 0; j < l3; j++){
			h_Matrix v1Col = v1->getCol(i);
			h_Matrix v2Col = v2->getCol(i);
			matrixMultiplyCuda(&v1Col, &re->getPlane(j), &aMatrix, 1, re->width, v1->height, CUBLAS_OP_T, CUBLAS_OP_N, &scalar, &zero);
			__syncthreads();
			matrixMultiplyCuda(&aMatrix, &v2Col, cc->getElement(i), 1, 1, aMatrix.width, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &scalar);
			__syncthreads();
		}
	}
	return;
}

__global__ void d_IPSP3dKernel(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){
	d_IPSP3d(re, v1, v2, v3, cc);
}


__device__ void d_IP3d_max(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc, double* maxVal, int* maxInd, h_Matrix* aMatrix){
	double zero = 0;
	double one = 1;
	__syncthreads();

	for(int i = 0; i < v3->width; i++){
		for(int j = 0; j < v3->height; j++){
			//SINGLE_THREAD{ printf("%d, %d, %d", re->height, re->width, re->depth); }
			//SINGLE_THREAD{ printf("%f, %f, %d, %d, %d, %f, %f ", v1->elements[1], re->getPlane(j).elements[0], v1->width, v1->height, re->width, re->getPlane(j).elements[0], v1->elements[0]); }
			__syncthreads();
			matrixMultiplyCuda(v1, &re->getPlane(j), aMatrix, v1->width, v1->height, re->width, CUBLAS_OP_T, CUBLAS_OP_N, &one);
			__syncthreads();
			if(j == 0){
			//	SINGLE_THREAD{ printf("%f,%f,%f,%f,%f", aMatrix->elements[0], v2->elements[0], cc->elements[0], v3->getElement(i, j)[0], zero); } 
				matrixMultiplyCuda(aMatrix, v2, cc, aMatrix->height, v2->width, v2->height, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &zero);
			}else{
				matrixMultiplyCuda(aMatrix, v2, cc, aMatrix->height, v2->width, v2->height, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &one);
			}
			__syncthreads();
			//SINGLE_THREAD{ printf("%f, %f, %f, %f, %d, %d\n", cc->elements[0], aMatrix->elements[0], maxVal[0], v2->elements[0], maxInd[0], j ); }
			d_findMaxInd(cc, maxInd, maxVal, i);
			
		}
	}
	__syncthreads();
//	SINGLE_THREAD{ delete aMatrix; }
	return;
}

__global__ void d_IP3d_maxKernel(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc, double* max, int* ind, h_Matrix* aMatrix){
	d_IP3d_max(re, v1, v2, v3, cc, max, ind, aMatrix);
}


__device__ void d_hnew3d(double cc, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* hnew){
	for(int zk = 0; zk < v3->numel(); zk++){
		matrixMultiplyCuda(v1, v2, &hnew->getPlane(zk), v1->height, v1->height, v2->width, CUBLAS_OP_N, CUBLAS_OP_T, v3->elements[zk] * cc, 0.0);
	}
}

__global__ void d_hnew3dKernel(double* cc, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* hnew){
	d_hnew3d(cc[0], v1, v2, v3, hnew);
}


__device__ void d_SPMP3D(h_Matrix* f, h_Matrix* Vx, h_Matrix* Vy, h_Matrix* Vz,
						double* tol, double* No, double* toln, int* lstep, int* Max, int* Maxp,
						h_Matrix* h, h_Matrix* c, int* Set_ind, int* numat){
		//all need to be shared

		for(int i = (c->numel() % (BLOCK_WIDTH*BLOCK_WIDTH) == 0)? 0 : -1 ; i < c->numel() / (BLOCK_WIDTH*BLOCK_WIDTH); i++){
			if(i * (threadIdx.x * c->height + threadIdx.y) < c->numel()){
					c->elements[i * (threadIdx.x * c->height + threadIdx.y)] = 0;
			}
		}

		__shared__ int Nxyz;
		__shared__ double delta;

		SINGLE_THREAD{
			Nxyz = Vx->height * Vy->height * Vz->height;
			delta = 1.0/ Nxyz;
			numat[0] = 0;
		}

		__shared__ double* Di1;
		__shared__ double* Di2;
		__shared__ double* Di3;

		// Not including Dix, Diy, Diz, numind since custom indexing is not priority

		__shared__ double sum;
		d_squaredSum(f, &sum);

		// Reorder so function doesn't return after allocation
		__syncthreads();
		if(sum * delta < 1e-9){
			c = new h_Matrix(0,0,0);
			return;
		}


		//todo Can cp be optimised so it can be stored on shared mem? (Currently too large 40*40*15!)
		__shared__ double* ccElements;
		__shared__ h_Matrix* cc;
		__shared__ h_Matrix* Re;
		__shared__ double* ReElements;
		__shared__ double tol2;
		__shared__ int imp;
		__shared__ int Maxit2;

		SINGLE_THREAD{
			ccElements = new double[Vx->width * Vy->width](); // Reduced size so it can be shared
			cc = new h_Matrix(ccElements, Vx->width, Vy->width,1);
			ReElements  = new double[f->numel()]();
			Re = new h_Matrix(ReElements, f->height, f->width, f->depth);

			tol2 = 1e-9;

			if(lstep[0] == -1){
				imp = 1;
				lstep[0] = Max[0];
				Maxit2 = 1;
			}else{
				Maxit2 = Max[0] / lstep[0];
			}
		}

		__syncthreads();
		makeCopyOfMatrixElements(Re, f);



		__shared__ h_Matrix* h_new;
		__shared__ double* h_new_elements;
		__shared__ int q[3];
		__shared__ double* multResultElements;
		__shared__ h_Matrix* multResult;

		__shared__ h_Matrix* Set_ind_trans;
		__shared__ double* Set_ind_trans_elements;
		__shared__ h_Matrix* Set_ind_container;

		SINGLE_THREAD{
			h_new_elements = new double[Vx->height * Vy->height * Vz->height]();
			h_new = new h_Matrix(h_new_elements, Vx->height, Vy->height, Vz->height);

			multResultElements = new double[Vx->height];
			multResult = new h_Matrix(multResultElements, 1, Vx->height, 1);

			Set_ind_trans = new h_Matrix(1,1,1);
		}


		__shared__ h_Matrix* aMatrix;
		__shared__ double* aMatrixElements;
		SINGLE_THREAD{
			aMatrixElements = new double[Vx->width * Re->width]();
			aMatrix = new h_Matrix(aMatrixElements, Vx->width, Re->width, 1);
		}


		__syncthreads();
		int it;
		for(it = 0; it < Maxit2; it++){
			__syncthreads();
			for(int s = 0; s < lstep[0]; s++){
				// Not including custom indexing processing since it is not priority
				__shared__ double maxVal;
				__shared__ int maxInd;

				SINGLE_THREAD{ maxVal = 0; maxInd = 0; }

				__syncthreads();

				d_IP3d_max(Re, Vx, Vy, Vz, cc, &maxVal, &maxInd, aMatrix); // Returns maxVal as normal form, i.e. not absolute. Hence cscra = maxVal, maxVal = abs(maxVal)

				SINGLE_THREAD{ ind2sub(cc->height, cc->width, cc->depth, maxInd, q); }

				__syncthreads();

				if(abs(maxVal) < tol2){
					SINGLE_THREAD{
						printf("SPMP3D stopped, max(|<f,q|/||q||) <= tol2 %g,%g,%g,%d,%d,%d\n", maxVal, tol2, cc->elements[0],it,s, maxInd);
						delete cc;
						delete multResult;
						delete h_new;
						delete Re;
						delete aMatrix;
					}
					__syncthreads();
					return;
				}


				// Has the indice been stored already, if not add it.
				SINGLE_THREAD{
					if (numat[0] == 0){
						Set_ind[0] = q[0];
						Set_ind[1] = q[1];
						Set_ind[2] = q[2];
						numat[0] += 1;
						c->elements[0] = maxVal;
					
					}else{					
						int exists = 0;
						int index = 0;
						for(int k = 0; k < numat[0]; k++){
							if(Set_ind[k * 3] == q[0] && Set_ind[k * 3 + 1] == q[1] && Set_ind[k * 3 + 2] == q[2]){
								exists = 1;
								index = k;
							}
						}

						if(exists == 0){
							Set_ind[(numat[0]) * 3] = q[0];
						 	Set_ind[(numat[0]) * 3 + 1] = q[1];
						 	Set_ind[(numat[0]) * 3 + 2] = q[2];
							numat[0] += 1;
							c->elements[numat[0]] = maxVal;
						}else{
							c->elements[index] += maxVal;
						}				
					}
				}

				h_Matrix VxCol = Vx->getCol(q[0]);
				h_Matrix VyCol = Vy->getCol(q[1]);
				h_Matrix VzCol = Vz->getCol(q[2]);
				
				d_hnew3d(maxVal, &VxCol, &VyCol, &VzCol, h_new);
				
				__syncthreads();

				d_matrixAddition(h, h_new, h);
				d_matrixSubtraction(Re, h_new, Re);

				__syncthreads();

				__shared__ double nor_new;
				
				d_squaredSum(Re, &nor_new);

				__syncthreads();

				if(numat[0] >= No[0] || ((nor_new * delta) < tol[0])){ 
					__syncthreads(); 
					break;				
				}

				__syncthreads();
			}

			__syncthreads();

			__shared__ double nore;
			SINGLE_THREAD{ nore = 0; }
			d_squaredSum(Re, &nore);

			if(numat[0] >= No[0] || ((nore * delta) < tol[0])){ break; }

		}

		if((lstep[0] != Max[0]) || it == Maxit2){ printf("Maximum Iterations has been reached"); }


	SINGLE_THREAD{
		delete cc;
		delete multResult;
		delete h_new;
		delete Re;
		delete aMatrix;
	}
}

__global__ void d_SPMP3DKernel(h_Matrix* f, h_Matrix* Vx, h_Matrix* Vy, h_Matrix* Vz,
						double* tol, double* No, double* toln, int* lstep, int* Max, int* Maxp,
						h_Matrix* h, h_Matrix* c, int* Set_ind, int* numat){
	if(threadIdx.z == 0){
	__shared__ h_Matrix* fBlock;
	__shared__ h_Matrix* VxShared;
	__shared__ h_Matrix* VyShared;
	__shared__ h_Matrix* VzShared;
	__shared__ h_Matrix* hShared;
	__shared__ h_Matrix* cShared;

	__shared__ double* fBlockElements;
	__shared__ double* VxSharedElements;
	__shared__ double* VySharedElements;
	__shared__ double* VzSharedElements;
	__shared__ double* hSharedElements;
	__shared__ double* cSharedElements;
	__shared__ int* Set_indShared;

	__shared__ double tolShared[1];
	__shared__ double NoShared[1];
	__shared__ double tolnShared[1];
	__shared__ int lstepShared[1];
	__shared__ int MaxShared[1];
	__shared__ int MaxpShared[1];
	__shared__ int numatShared[1];
	
	/*todo Need to create large space when creating arrays on global memory, and they customly pickout the bits to write to depending on the block size. Perhaps clash? */
	
	SINGLE_THREAD{
		fBlockElements = new double[Vx->height * Vy->height * Vz->height]();
		fBlock = new h_Matrix(fBlockElements, Vx->height, Vy->height, Vz->height);

		VxSharedElements = new double[Vx->height * Vx->width]();
		VxShared = new h_Matrix(VxSharedElements, Vx->height, Vx->width, Vx->depth);

		VySharedElements = new double[Vy->height * Vy->width]();
		VyShared = new h_Matrix(VySharedElements, Vy->height, Vy->width, Vy->depth);

		VzSharedElements = new double[Vz->height * Vz->width]();
		VzShared = new h_Matrix(VzSharedElements, Vz->height, Vz->width, Vz->depth);

		hSharedElements = new double[h->height * h->width * h->depth]();
		hShared = new h_Matrix(hSharedElements, h->height, h->width, h->depth);

		cSharedElements = new double[c->height * c->width * c->depth]();
		cShared = new h_Matrix(cSharedElements, c->height, c->width, c->depth);

		Set_indShared = new int[Vx->height * Vy->height * Vz->height]();

		tolShared[0] = tol[0];
		NoShared[0] = No[0];
		tolnShared[0] = toln[0];
		lstepShared[0] = lstep[0];
		MaxShared[0] = Max[0];
		MaxpShared[0] = Maxp[0];
		numatShared[0] = numat[0];
	}
	__syncthreads();
	
	moveBetweenShared(VxShared, Vx, Vx->height);
	moveBetweenShared(VyShared, Vy, Vy->height);
	moveBetweenShared(VzShared, Vz, Vz->height);
	moveBetweenShared(hShared, h, h->height);
	moveBetweenShared(cShared, c, c->height);
	
	extractBlock(fBlock, f, Vx->height);

	__syncthreads();

	d_SPMP3D(fBlock, VxShared, VyShared, VzShared, tolShared, NoShared, tolnShared, lstepShared, MaxShared, MaxpShared, hShared, cShared, Set_indShared, numatShared);

	__syncthreads();

	moveBetweenShared(h, hShared, h->height);
	moveBetweenShared(c, cShared, c->height);
	memcpy(Set_ind, Set_indShared, sizeof(int) * fBlock->height * fBlock->width * fBlock->depth);
	
	SINGLE_THREAD{printf("\nhshared: %f,\tc: %f\t%d,%d - fb: %f\tsmid: %d\n ", hShared->elements[0], cShared->elements[0], blockIdx.x, blockIdx.y, fBlock->elements[191], get_smid());}


	SINGLE_THREAD{
		delete fBlock;
		delete VxShared;
		delete VyShared;
		delete VzShared;
		delete hShared;
		delete cShared;
		delete [] Set_indShared;
	}
	}
}

#endif
