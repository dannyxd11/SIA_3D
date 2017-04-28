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
#define SINGLE_THREAD if(threadIdx.x == 0 && threadIdx.y == 0 )

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
			matrixMultiplyCuda(v1, &re->getPlane(j), aMatrix, v1->width, v1->height, re->width, CUBLAS_OP_T, CUBLAS_OP_N, &one, &zero);
			__syncthreads();
			if(j == 0){
			//	SINGLE_THREAD{ printf("%f,%f,%f,%f,%f", aMatrix->elements[0], v2->elements[0], cc->elements[0], v3->getElement(i, j)[0], zero); } 
				matrixMultiplyCuda(aMatrix, v2, cc, aMatrix->height, v2->width, v2->height, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &zero);
			}else{
				matrixMultiplyCuda(aMatrix, v2, cc, aMatrix->height, v2->width, v2->height, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &one);
			}
			//__syncthreads();
			//SINGLE_THREAD{ printf("%f, %f, %f, %f, %d, %d\n", cc->elements[0], aMatrix->elements[0], maxVal[0], v2->elements[0], maxInd[0], j ); }
//			d_findMaxInd(cc, maxInd, maxVal, i);			
		}
		//if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x ==2 && i == 6){printf("\tcc:%f\t",cc->elements[41]);}
		__syncthreads();
		d_findMaxInd(cc, maxInd, maxVal, i);
	}
	__syncthreads();
	return;
}

__global__ void d_IP3d_maxKernel(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc, double* maxVal, int* ind, h_Matrix* aMatrix){
	d_IP3d_max(re, v1, v2, v3, cc, maxVal, ind, aMatrix);
}

__global__ void d_IP3d_maxKernel2(h_Matrix* f, h_Matrix* Vx, h_Matrix* Vy, h_Matrix* Vz, double* maxVal, int* ind){
	if(threadIdx.z == 0){
	__shared__ h_Matrix* fBlock;
	__shared__ h_Matrix* VxShared;
	__shared__ h_Matrix* VyShared;
	__shared__ h_Matrix* VzShared;
	__shared__ h_Matrix* ccShared;
	__shared__ h_Matrix* aShared;

	__shared__ double* fBlockElements;
	__shared__ double* VxSharedElements;
	__shared__ double* VySharedElements;
	__shared__ double* VzSharedElements;
	__shared__ double* ccSharedElements;
	__shared__ double* aSharedElements;


	__shared__ double maxShared[1];
	__shared__ int indShared[1];

	

	extern __shared__ double sharedPool[];
	/*todo Need to create large space when creating arrays on global memory, and they customly pickout the bits to write to depending on the block size. Perhaps clash? */
	
	if(threadIdx.x == 0 && threadIdx.y == 0 ){
		/*fBlockElements = new double[Vx->height * Vy->height * Vz->height]();*/
		fBlockElements = &sharedPool[0];
		fBlock = new h_Matrix(fBlockElements, Vx->height, Vy->height, Vz->height, -1);		
	}	
	if(threadIdx.x == 1 && threadIdx.y == 0 ){	
		/*VxSharedElements = new double[Vx->height * Vx->width](); */
		VxSharedElements = &sharedPool[(Vx->height * Vy->height * Vz->height)];
		VxShared = new h_Matrix(VxSharedElements, Vx->height, Vx->width, Vx->depth, -1);
	}	
	if(threadIdx.x == 2 && threadIdx.y == 0 ){
		/*VySharedElements = new double[Vy->height * Vy->width]();*/
		VySharedElements = &sharedPool[(Vx->height * Vy->height * Vz->height) + Vx->numel()];
		VyShared = new h_Matrix(VySharedElements, Vy->height, Vy->width, Vy->depth, -1);
	}	
	if(threadIdx.x == 3 && threadIdx.y == 0 ){
		/* VzSharedElements = new double[Vz->height * Vz->width](); */
		VzSharedElements = &sharedPool[(Vx->height * Vy->height * Vz->height) + Vx->numel() + Vy->numel()];
		VzShared = new h_Matrix(VzSharedElements, Vz->height, Vz->width, Vz->depth, -1);
	}	
	if(threadIdx.x == 4 && threadIdx.y == 0 ){
		//hSharedElements = new double[Vx->height * Vy->height * Vz->height]();
		ccSharedElements = &sharedPool[(Vx->height * Vy->height * Vz->height) + Vx->numel() + Vy->numel() + Vz->numel()];
		ccShared = new h_Matrix(ccSharedElements, Vx->width, Vy->width, 1);		
	}	
	if(threadIdx.x == 5 && threadIdx.y == 0 ){
		//cSharedElements = new double[Vx->height * Vy->height * Vz->height]();
		aSharedElements = &sharedPool[(Vx->height * Vy->height * Vz->height)*2 + Vx->numel() + Vy->numel() + Vz->numel()];
		aShared = new h_Matrix(aSharedElements, Vx->width, Vx->height, 1);
	}	
	if(threadIdx.x == 6 && threadIdx.y == 0 ){
		maxShared[0] = 0;
		indShared[0] = 0;
	}
	__syncthreads();

	moveBetweenShared(VxShared, Vx, Vx->height);
	moveBetweenShared(VyShared, Vy, Vy->height);
	moveBetweenShared(VzShared, Vz, Vz->height);
	
	extractBlock(fBlock, f, Vx->height);

	d_clearMatrix(ccShared);
	d_clearMatrix(aShared);

	__syncthreads();

	d_IP3d_max(fBlock, VxShared, VyShared, VzShared, ccShared, maxShared, indShared, aShared);

	//memcpy(Set_ind, Set_indShared, sizeof(int) * fBlock->height * fBlock->width * fBlock->depth);
	SINGLE_THREAD{printf("\nmax: %f,\tind: %d\tblock:%d,%d\tsmid:%d\tf:%f", maxShared[0], indShared[0],blockIdx.x, blockIdx.y, get_smid(),fBlock->elements[0]);}
	//SINGLE_THREAD{printf("\nhshared: %f,\tc: %f\t%d,%d - fb: %f\tsmid: %d\t gridDim: %d\n ", hShared->elements[0], cShared->elements[0], blockIdx.x, blockIdx.y, fBlock->elements[0], get_smid(), gridDim.x);}
	

	SINGLE_THREAD{	
		maxVal[0] = maxShared[0];
		ind[0] = indShared[0];
	}


}

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
						//h_Matrix* h, h_Matrix* c, int* Set_ind, int* numat, size_t sharedInd){
						h_Matrix* h, double* c, int* Set_ind, int* numat, size_t sharedInd){

		for(int i = (f->numel()*2 % (BLOCK_WIDTH*BLOCK_WIDTH) == 0)? 0 : -1 ; i < f->numel() / (BLOCK_WIDTH*BLOCK_WIDTH); i++){
			//if(i * (threadIdx.x * c->height + threadIdx.y) < c->numel()){
			if(i * (threadIdx.x * f->height + threadIdx.y) < f->numel()){
					c[i * (threadIdx.x * f->height + threadIdx.y)] = 0;
			}
		}

		int Nxyz = Vx->height * Vy->height * Vz->height;
		double delta =  1.0/ Nxyz;
		numat[0] = 0;

		__shared__ double* Di1;
		__shared__ double* Di2;
		__shared__ double* Di3;

		// Not including Dix, Diy, Diz, numind since custom indexing is not priority

		__shared__ double sum;
		d_squaredSum(f, &sum);

		if(sum * delta < 1e-9){
			//c = new h_Matrix(0,0,0);
			c = new double[0];
			return;
		}


		__shared__ double* ccElements;
		__shared__ h_Matrix* cc;
		__shared__ h_Matrix* Re;
		__shared__ double* ReElements;
		__shared__ double tol2;
		__shared__ int imp;
		__shared__ int Maxit2;

		SINGLE_THREAD{
			ccElements = new double[Vx->width * Vy->width]();
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

		extern __shared__ double sharedPool[];
		__shared__ h_Matrix* aMatrix;
		__shared__ double* aMatrixElements;
		SINGLE_THREAD{
			aMatrixElements = &sharedPool[sharedInd];
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
						//c->elements[0] = maxVal;
						c[0] = maxVal;
					
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
							//c->elements[numat[0]] = maxVal;
							c[numat[0]] = maxVal;
						}else{
							//c->elements[index] += maxVal;
							c[index] += maxVal;
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

		if((lstep[0] != Max[0]) && it == Maxit2){ printf("Maximum Iterations has been reached"); }


	SINGLE_THREAD{
		delete cc;
		delete multResult;
		delete h_new;
		delete Re;
		//delete aMatrix;
	}
}

__global__ void d_SPMP3DKernel(h_Matrix* f, h_Matrix* Vx, h_Matrix* Vy, h_Matrix* Vz,
						double* tol, double* No, double* toln, int* lstep, int* Max, int* Maxp,
	//					h_Matrix* h, h_Matrix* c, int* Set_ind, int* numat){
					h_Matrix* h, double* c, int* Set_ind, int* numat){
	//if(threadIdx.z == 0 && blockIdx.x <= 3 && blockIdx.y == 0){
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
	

	extern __shared__ double sharedPool[];
	size_t __shared__ sharedPoolIndex;
	
	
	if(threadIdx.x == 0 && threadIdx.y == 0 ){			
		fBlockElements = &sharedPool[0];
		fBlock = new h_Matrix(fBlockElements, Vx->height, Vy->height, Vz->height, -1);
	}
	if(threadIdx.x == 1 && threadIdx.y == 0 ){	
		VxSharedElements = &sharedPool[Vx->height * Vy->height * Vz->height];
		VxShared = new h_Matrix(VxSharedElements, Vx->height, Vx->width, Vx->depth, -1);
	}
	if(threadIdx.x == 2 && threadIdx.y == 0 ){
		VySharedElements = &sharedPool[Vx->height * Vy->height * Vz->height + Vx->numel()];
		VyShared = new h_Matrix(VySharedElements, Vy->height, Vy->width, Vy->depth, -1);
	}
	if(threadIdx.x == 3 && threadIdx.y == 0 ){
		VzSharedElements = &sharedPool[Vx->height * Vy->height * Vz->height + Vx->numel() + Vy->numel()];
		VzShared = new h_Matrix(VzSharedElements, Vz->height, Vz->width, Vz->depth, -1);
	}
	if(threadIdx.x == 0 && threadIdx.y == 1 ){
		hSharedElements = &sharedPool[Vx->height * Vy->height * Vz->height + Vx->numel() + Vy->numel() + Vz->numel()];
		hShared = new h_Matrix(hSharedElements, Vx->height, Vy->height, Vz->height);
	}
	if(threadIdx.x == 0 && threadIdx.y == 2 ){
		cSharedElements = &sharedPool[Vx->height * Vy->height * Vz->height*2 + Vx->numel() + Vy->numel() + Vz->numel()];
		//cShared = new h_Matrix(cSharedElements, Vx->height, Vy->height, Vz->height);
	}
	if(threadIdx.x == 0 && threadIdx.y == 3 ){
		Set_indShared = new int[Vx->height * Vy->height * Vz->height*2]();
	}
	if(threadIdx.x == 0 && threadIdx.y == 4 ){
		tolShared[0] = tol[0];
		NoShared[0] = No[0];
		tolnShared[0] = toln[0];
		lstepShared[0] = lstep[0];
		MaxShared[0] = Max[0];
		MaxpShared[0] = Maxp[0];
		numatShared[0] = 0;
		sharedPoolIndex = Vx->height * Vy->height * Vz->height*4 + Vx->numel() + Vy->numel() + Vz->numel();
	}
	__syncthreads();

	moveBetweenShared(VxShared, Vx, Vx->height);
	moveBetweenShared(VyShared, Vy, Vy->height);
	moveBetweenShared(VzShared, Vz, Vz->height);

	d_clearMatrix(hShared);
	//d_clearMatrix(cShared);
	
	extractBlock(fBlock, f, Vx->height);

	__syncthreads();

	d_SPMP3D(fBlock, VxShared, VyShared, VzShared, tolShared, NoShared, tolnShared, lstepShared, MaxShared, MaxpShared, hShared, cSharedElements, Set_indShared, numatShared, sharedPoolIndex);
//	printf(" %d\n",(blockIdx.x * gridDim.y + blockIdx.y));
	//SINGLE_THREAD{h->elements = &h->elements[(blockIdx.x * gridDim.y + blockIdx.y) * hShared->numel()];}
	__syncthreads();

	//memcpy(Set_ind, Set_indShared, sizeof(int) * fBlock->height * fBlock->width * fBlock->depth);
	
	//SINGLE_THREAD{printf("\nhshared: %f,\tc: %f\t%d,%d - fb: %f\tsmid: %d\t gridDim: %d\n ", hShared->elements[0], cShared->elements[0], blockIdx.x, blockIdx.y, fBlock->elements[0], get_smid(), gridDim.x);}
	

	
	//makeCopyOfMatrixElements(h, hShared);
	SINGLE_THREAD{	
		int blockIndex = (blockIdx.x * gridDim.y + blockIdx.y);
		numat[blockIndex] = numatShared[0];	


		//printf(" %d\n", (blockIdx.x * gridDim.x + blockIdx.y));
		//memcpy(h->elements + (blockIdx.x * gridDim.y + blockIdx.y) * hShared->numel() * sizeof(double), hShared->elements, hShared->numel() * sizeof(double));
		//printf("blockindex: %d, numat: %d\n", blockIndex, numatShared[0]);
		//memcpy(h->elements + blockIndex * hShared->numel() * sizeof(double), hShared->elements, hShared->numel() * sizeof(double));
		//memcpy(h->elements + (blockIdx.x * gridDim.x + blockIdx.y) * hShared->numel() * sizeof(double), hShared->elements, hShared->numel() * sizeof(double));
		//memcpy(h->elements, hShared->elements, hShared->numel() * sizeof(double));
		//memcpy(c->elements + (blockIdx.x * gridDim.x + blockIdx.y) * cShared->numel() * sizeof(double), cShared->elements, cShared->numel() * sizeof(double)); 

		delete fBlock;
		delete VxShared;
		delete VyShared;
		delete VzShared;
		//delete hShared;
		//delete cShared;
		delete [] Set_indShared;
	}

	}
}

#endif
