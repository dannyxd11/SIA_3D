#ifndef CUDA_COMMON_OPS
#define CUDA_COMMON_OPS

#include <iostream>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cstring>
#include <stdio.h>
#include "h_Matrix.h"

#define BLOCK_WIDTH 8
#define SINGLE_THREAD if(threadIdx.x == 0 && threadIdx.y == 0)

#ifndef CUDACHECK
#define CUDACHECK
#define cudaCheck(input){cudaAssert((input), __FILE__, __LINE__); } // http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void cudaAssert(cudaError_t code, const char *file, int line){
	if (code != cudaSuccess){ fprintf(stderr, "CudaAssert %s %s %d\n", cudaGetErrorString(code), file, line); exit(code);}
}
#endif




__host__ __device__ void ind2sub(int height, int width, int depth, int index, int* q){
	    int plane = height * width;
	    q[2] = index / plane;
	    int rem = index % plane;
	    q[1] = rem / height;
	    q[0] = rem % width;
}





__device__ void multiplyCuda(double* a, double* b, double* c, int lda, int ldb, int ldc, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha, double* beta){
	// m is number of rows of op(a)
	// n is number of cols of op(b)
	// k is number of rows of a and width of c)
	int y = threadIdx.y; //col
	int x = threadIdx.x; //row
	int block_size = 8; //todo allow variable block_sizes
/*	
	for(x = threadIdx.x; x < n; x += block_size ){
		for(y = threadIdx.y; y < m; y += block_size){
			if (y < m && x < n){
				double cellSum = 0;
				if (op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_N){
					for(int i = 0; i < k; i++){
						//printf("%d,\n", lda * y + i);
						cellSum += a[lda * i + y] * b[ldb * x + i] * alpha[0];
						//SINGLE_THREAD{printf("\ni: %d, y: %d, x: %d, lda: %d, ldb: %d, alpha: %f, temp: %f, cellSum: %f, aVal: %f, bVal: %f, aind %d, bind: %d, a0 %f, b0 %f", i, y, x, lda, ldb, alpha[0], a[lda * i + y] * b[ldb * x + i] * alpha[0], cellSum, a[lda * i + y], b[ldb * x + i], lda*i+y, ldb*x+i,a[0],b[0]);}
					}
				}else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_N){
					for(int i = 0; i < k; i++){
						//printf("%d - %f,\n", lda * y + i, a[lda * y + i]);
						//printf("%d,%d,%d\n", ldb, x, i);
						//SINGLE_THREAD{printf("\ni: %d, y: %d, x: %d,threadidx: %d, threadidy: %d, lda: %d, ldb: %d, alpha: %f, temp: %f, cellSum: %f, aVal: %f, bVal: %f, aind %d, bind: %d, m: %d, n: %d, k: %d", i, y, x,threadIdx.x, threadIdx.y, lda, ldb, alpha[0], a[lda * y + i] * b[ldb * x + i] * alpha[0], cellSum, a[lda * y + i], b[ldb * x + i], lda * y + i, ldb*x+i, m,n,k);}
						cellSum += a[lda * y + i] * b[ldb * x + i] * alpha[0];

					}
				}else if(op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_T){
					for(int i = 0; i < k; i++){
						cellSum += a[lda * i + y] * b[ldb * i + x] * alpha[0];
					}
				}else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_T){
					for(int i = 0; i < k; i++){
						cellSum += a[lda * y + i] * b[ldb * i + x] * alpha[0];
					}
				}

				c[ldc * x + y] = beta[0] * c[ldc * x + y] + cellSum;
				//SINGLE_THREAD{printf("\nRow: %d,%d, A: %f, / %d    B: %f, / %d    C: %f / %d \t K: %d, M: %d, N: %d\n", x,y,a[lda * x + y],lda * y + x,b[ldb * x + y],ldb * x + y,c[ldc * x + y],ldc * y + x, k, m, n);}
			}
		}
	} */


	if (op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_N){
		for(x = threadIdx.x; x < n; x += block_size ){
			for(y = threadIdx.y; y < m; y += block_size){
				if (y < m && x < n){
					double cellSum = 0;
					for(int i = 0; i < k; i++){
						cellSum += a[lda * i + y] * b[ldb * x + i] * alpha[0];
					}
					c[ldc * x + y] = beta[0] * c[ldc * x + y] + cellSum;
				}
			}
		}
	}else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_N){
		for(x = threadIdx.x; x < n; x += block_size ){
			for(y = threadIdx.y; y < m; y += block_size){
				if (y < m && x < n){
					double cellSum = 0;
					for(int i = 0; i < k; i++){
						cellSum += a[lda * y + i] * b[ldb * x + i] * alpha[0];
					}
					c[ldc * x + y] = beta[0] * c[ldc * x + y] + cellSum;
				}
			}
		}
	}else if(op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_T){
		for(x = threadIdx.x; x < n; x += block_size ){
			for(y = threadIdx.y; y < m; y += block_size){
				if (y < m && x < n){
					double cellSum = 0;
					for(int i = 0; i < k; i++){
						cellSum += a[lda * i + y] * b[ldb * i + x] * alpha[0];
					}
					c[ldc * x + y] = beta[0] * c[ldc * x + y] + cellSum;
				}
			}
		}
	}else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_T){
		for(x = threadIdx.x; x < n; x += block_size ){
			for(y = threadIdx.y; y < m; y += block_size){
				if (y < m && x < n){
					double cellSum = 0;
					for(int i = 0; i < k; i++){
						cellSum += a[lda * y + i] * b[ldb * i + x] * alpha[0];
					}
					c[ldc * x + y] = beta[0] * c[ldc * x + y] + cellSum;
				}
			}
		}
	}


}


__device__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha, double* beta){
	int lda = a->height;
	int ldb = b->height;
	int ldc = m;
	multiplyCuda(a->elements, b->elements, c->elements,lda, ldb, ldc, m, n, k, op1, op2, alpha, beta);
}

__device__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double alpha, double beta){
	int lda = a->height;
	int ldb = b->height;
	int ldc = m;
	multiplyCuda(a->elements, b->elements, c->elements,lda, ldb, ldc, m, n, k, op1, op2, &alpha, &beta);
}

__device__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, double* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha, double* beta){
	int lda = a->height;
	int ldb = b->height;
	int ldc = m;
	multiplyCuda(a->elements, b->elements, c,lda, ldb, ldc, m, n, k, op1, op2, alpha, beta);
}

__device__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
	double* beta = new double[1]();
	matrixMultiplyCuda(a, b, c, m, n, k, op1, op2, alpha, beta);
}

__device__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
	int m;
	int n;

	if(op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_N){
		m = a->height;
		n = b->width;
		if( a->width != b->height ){
			__threadfence();
			asm("trap;");
		}
	}else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_N){
		m = a->width;
		n = b->width;
		if( a->height != b->height ){
			__threadfence();
			asm("trap;");
		}
	}else if(op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_T){
		m = a->height;
		n = b->height;
		if( a->width != b->width ){
			__threadfence();
			asm("trap;");
		}
	}else{
		m = a->width;
		n = b->height;
		if( a->height != b->width ){
			__threadfence();
			asm("trap;");
		}
	}

	int k = a->height;
	if(op1 == CUBLAS_OP_N){
		k = a->width;
	}

	int leadingDimensionA = a->height;
	int leadingDimensionB = b->height;
	int leadingDimensionC = a->height;

	double* beta = new double[1]();
	multiplyCuda(a->elements, b->elements, c->elements, leadingDimensionA, leadingDimensionB, leadingDimensionC, m, n, k, op1, op2, alpha, beta);
}

__global__ void matrixMultiplyCudaKernel(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
	matrixMultiplyCuda(a, b, c, m, n, k, op1, op2, alpha);
}

__global__ void matrixMultiplyCudaKernel(h_Matrix* a, h_Matrix* b, h_Matrix* c, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
	matrixMultiplyCuda(a, b, c, op1 ,op2, alpha);
}


__device__ int transpose(h_Matrix* destMatrix, h_Matrix* srcMatrix){
	if(destMatrix->numel() != srcMatrix->numel() || destMatrix->height != srcMatrix->width || destMatrix->width != srcMatrix->height){
		return -1;
	}
	destMatrix->height = srcMatrix->width;
	destMatrix->width = srcMatrix->height;
	__syncthreads();
	for(int i = 0 ; i < srcMatrix->width / (BLOCK_WIDTH) + ((srcMatrix->width % (BLOCK_WIDTH) == 0) ? 0 : +1); i++){
		for(int j = 0 ; j < srcMatrix->height / (BLOCK_WIDTH) + ((srcMatrix->height % (BLOCK_WIDTH) == 0) ? 0 : +1); j++){
				printf("%d, %d, %d, %d\n", threadIdx.x + i ,threadIdx.y + j, srcMatrix->width / (BLOCK_WIDTH) + ((srcMatrix->width % (BLOCK_WIDTH) == 0) ? 0 : +1), srcMatrix->height / (BLOCK_WIDTH) + ((srcMatrix->height % (BLOCK_WIDTH) == 0) ? 0 : +1));
				if(threadIdx.x + i * BLOCK_WIDTH < srcMatrix->width && threadIdx.y + j * BLOCK_WIDTH < srcMatrix->height){
					destMatrix->setElement(threadIdx.y + j * BLOCK_WIDTH,threadIdx.x + i * BLOCK_WIDTH,srcMatrix->getElement(threadIdx.x + i * BLOCK_WIDTH,threadIdx.y + j * BLOCK_WIDTH)[0]);
				}
		}
	}
	return 0;
}

__global__ void d_transposeKernel(h_Matrix* destMatrix, h_Matrix* srcMatrix ){
	transpose(destMatrix, srcMatrix);
}

__device__ void d_clearMatrix(h_Matrix* matrix){
	for(int i = (matrix->numel() % (BLOCK_WIDTH*BLOCK_WIDTH) == 0)? 0 : -1 ; i < matrix->numel() / (BLOCK_WIDTH*BLOCK_WIDTH); i++){
		if(i * (threadIdx.x * matrix->height + threadIdx.y) < matrix->numel()){
				matrix->elements[i * (threadIdx.x * matrix->height + threadIdx.y)] = 0;
		}
	}
}


__device__ void d_findMaxInd(h_Matrix* matrix, int* maxInd, double* maxVal, int dimension){
	int numberOfElements = matrix->numel();
	int elementsPerThread = (numberOfElements/(BLOCK_WIDTH * BLOCK_WIDTH));

	if(numberOfElements % (BLOCK_WIDTH * BLOCK_WIDTH) != 0){
		elementsPerThread += 1; //Add 1 incase number of elements is not divisible by block dimensions
	}

	int localMaxInd = 0;
	double localMaxVal = 0;

	__shared__ int sharedMaxInd[BLOCK_WIDTH * BLOCK_WIDTH];
	__shared__ double sharedMaxVal[BLOCK_WIDTH * BLOCK_WIDTH];

	__syncthreads();

	for(int i = 0; i < elementsPerThread; i++){
		int index = elementsPerThread * (threadIdx.y * BLOCK_WIDTH + threadIdx.x) + i;
		if(index <= numberOfElements){
			if(index < matrix->numel()){
				if(abs(matrix->elements[index]) > abs(localMaxVal)) {
					localMaxInd = index;
					localMaxVal = matrix->elements[index];				
				}
			}
		}
	}

	// Assign each threads result to a unique index (based on thread) in __shared__ memory space
	sharedMaxInd[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = localMaxInd;
	sharedMaxVal[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = localMaxVal;
	__syncthreads();

	// Use just one thread to process the reduced array. Could possibly be done recursively
	SINGLE_THREAD{
		for(int i = 0; i < BLOCK_WIDTH * BLOCK_WIDTH; i++){
			//if(dimension == 6){printf("\n%d, %f, ", i, sharedMaxVal[i]);}			
			if(abs(sharedMaxVal[i]) > abs(maxVal[0])){
				maxVal[0] = sharedMaxVal[i];
				maxInd[0] = dimension * matrix->height * matrix->width + sharedMaxInd[i];
			}
		}
	}
}

__global__ void d_findMaxIndKernel(h_Matrix* matrix,int* maxInd, double* maxVal, int* dimension){
	d_findMaxInd(matrix, maxInd, maxVal, *dimension);
}


__device__ void d_squaredSum(h_Matrix* matrix, double* sum){
	int numberOfElements = matrix->numel();
	int elementsPerThread = (numberOfElements/(BLOCK_WIDTH * BLOCK_WIDTH));

	if(numberOfElements % (BLOCK_WIDTH * BLOCK_WIDTH) != 0){
		elementsPerThread += 1; //Add 1 incase number of elements is not divisible by block dimensions
	}

	double localSquaredSum = 0;

	__shared__ double sharedSquaredSum[BLOCK_WIDTH * BLOCK_WIDTH];

	if(threadIdx.x ==0 && threadIdx.y == 0){
		for(int i = 0; i < BLOCK_WIDTH * BLOCK_WIDTH; i++){
			sharedSquaredSum[i] = 0;
		}
	}


	// Split array into equal sizes to be processed by eac      sum(sum(sum(abs(Re).^2)))
	for(int i = 0; i < elementsPerThread; i++){
		int index = elementsPerThread * (threadIdx.y * BLOCK_WIDTH + threadIdx.x) + i;
		if(index < numberOfElements){
			localSquaredSum += matrix->elements[index] * matrix->elements[index];
		}
	}

	__syncthreads();

	// Assign each threads result to a unique index (based on thread) in __shared__ memory space
	sharedSquaredSum[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = localSquaredSum;
	__syncthreads();

	// Use just one thread to process the reduced array. Could possibly be done recursively
	if(threadIdx.x == 0 && threadIdx.y == 0){
		sum[0] = 0;
		for(int i = 0; i < BLOCK_WIDTH * BLOCK_WIDTH; i++){
			sum[0] += sharedSquaredSum[i];
		}
	}
	__syncthreads();
}

__global__ void d_squaredSumKernel(h_Matrix* matrix, double* sum){
	d_squaredSum(matrix, sum);
}

__device__ void d_matrixAddition(h_Matrix* matrixA, h_Matrix* matrixB, h_Matrix* matrixC){
	int numberOfElements = matrixA->numel();
	int blockDim = BLOCK_WIDTH * BLOCK_WIDTH;
	int elementsPerThread = (numberOfElements/(blockDim));

	if(numberOfElements % blockDim != 0){
		elementsPerThread += 1; //Add 1 incase number of elements is not divisible by block dimensions
	}

	// Split array into equal sizes to be processed by each thread
	for(int i = 0; i < elementsPerThread; i++){
		int index = elementsPerThread * (threadIdx.y * BLOCK_WIDTH + threadIdx.x) + i;
		if(index < numberOfElements){
			matrixC->elements[index] = matrixA->elements[index]  +  matrixB->elements[index];
		}
	}
	__syncthreads();
}

__global__ void d_matrixAdditionKernel(h_Matrix* matrixA, h_Matrix* matrixB, h_Matrix* matrixC){
	d_matrixAddition(matrixA, matrixB, matrixC);
}

__device__ void d_matrixSubtraction(h_Matrix* matrixA, h_Matrix* matrixB, h_Matrix* matrixC){
	int numberOfElements = matrixA->numel();
	int blockDim = BLOCK_WIDTH * BLOCK_WIDTH;
	int elementsPerThread = (numberOfElements/(blockDim));

	if(numberOfElements % blockDim != 0){
		elementsPerThread += 1; //Add 1 incase number of elements is not divisible by block dimensions
	}

	// Split array into equal sizes to be processed by each thread
	for(int i = 0; i < elementsPerThread; i++){
		int index = elementsPerThread * (threadIdx.y * BLOCK_WIDTH + threadIdx.x) + i;
		if(index < numberOfElements){
			matrixC->elements[index] = matrixA->elements[index]  -  matrixB->elements[index];
		}
	}

	__syncthreads();
}

__global__ void d_matrixSubtractionKernel(h_Matrix* matrixA, h_Matrix* matrixB, h_Matrix* matrixC){
	d_matrixSubtraction(matrixA, matrixB, matrixC);
}

#endif
