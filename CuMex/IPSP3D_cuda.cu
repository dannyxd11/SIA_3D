#ifndef IPSP3D
#define IPSP3D
#define cudaCheck(input){cudaAssert((input), __FILE__, __LINE__); } // http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define BLOCK_WIDTH 8
#define SINGLE_THREAD if(threadIdx.x == 0 && threadIdx.y == 0)

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include <stdio.h>

inline void cudaAssert(cudaError_t code, const char *file, int line){
	if (code != cudaSuccess){ fprintf(stderr, "CudaAssert %s %s %d\n", cudaGetErrorString(code), file, line); exit(code);}
}

void printDeviceDetails(){
	std::cout << "Device Details... \n" << std::endl;
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
				prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
				2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		int driverVersion = 0, runtimeVersion = 0;
		cudaRuntimeGetVersion(&runtimeVersion);
		cudaDriverGetVersion(&driverVersion); // http://rcs.bu.edu/examples/gpu/tutorials/deviceQuery/deviceQuery.cpp
		printf("  Driver Version: %d.%d\n",driverVersion/1000, (driverVersion%100)/10);
		printf("  Runtime Version: %d.%d\n",runtimeVersion/1000, (runtimeVersion%100)/10);
		printf("  CUDA Version: %d.%d\n\n",  prop.major, prop.minor);
	}
}

class h_Matrix{
public:
	double* elements;
	double* devElements;
	int height, width, depth;
	__host__ __device__ h_Matrix() : height(1), width(1), depth(1) {};
	__host__ __device__ h_Matrix(int height, int width, int depth) : height(height), width(width), depth(depth) { elements = new double[height*width*depth](); };
	__host__ __device__ h_Matrix(double* elements, int height, int width, int depth) : height(height), width(width), depth(depth), elements(elements) {};
	__host__ __device__ int numel (){return height * width * depth;};
	__host__ __device__ double* getColDouble(int i){return &elements[height * i];}
	__host__ __device__ double* getElement(int i, int j){return &elements[i * height + j];};
	__host__ __device__ void setElement(int i, int j, double value){elements[i * height + j] = value;};
	__host__ __device__ void setElement(int i, double value){elements[i] = value;};
	__host__ __device__ double* getElement(int i){return &elements[i];};
	__host__ __device__ h_Matrix getCol(int i){h_Matrix newMatrix(getColDouble(i),height, 1, 1); return newMatrix;};
	__host__ __device__ h_Matrix getPlane(int i){h_Matrix newMatrix(&elements[height * width * i], height, width, 1); return newMatrix;};

	__host__ __device__ ~h_Matrix(){if(!elements){delete [] elements;}; if(!devElements){delete [] devElements;};};
};

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
	for(x = threadIdx.x; x < n; x += block_size ){
		for(y = threadIdx.y; y < m; y += block_size){

			//printf("%d,%d,%d,%d, %d, %d, %d\n",y,m, x,n, lda,ldb, k);
			if (y < m && x < n){
				double cellSum = 0;
				if (op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_N){
					for(int i = 0; i < k; i++){
						//printf("%d,\n", lda * y + i);
						cellSum += a[lda * i + y] * b[ldb * x + i] * alpha[0];
						//printf("\ni: %d, y: %d, x: %d, lda: %d, ldb: %d, alpha: %f, temp: %f, cellSum: %f, aVal: %f, bVal: %f, aind %d, bind: %d", i, y, x, lda, ldb, alpha[0], a[lda * i + y] * b[ldb * x + i] * alpha[0], cellSum, a[lda * i + y], b[ldb * x + i], lda*i+y, ldb*x+i);
					}
				}else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_N){
					for(int i = 0; i < k; i++){
						//printf("%d,\n", lda * y + i);
						cellSum += a[lda * y + i] * b[ldb * x + i] * alpha[0];
						//printf("\ni: %d, y: %d, x: %d,threadidx: %d, threadidy: %d, lda: %d, ldb: %d, alpha: %f, temp: %f, cellSum: %f, aVal: %f, bVal: %f, aind %d, bind: %d, m: %d, n: %d, k: %d", i, y, x,threadIdx.x, threadIdx.y, lda, ldb, alpha[0], a[lda * y + i] * b[ldb * x + i] * alpha[0], cellSum, a[lda * y + i], b[ldb * x + i], lda * y + i, ldb*x+i, m,n,k);
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
				//printf("\nRow: %d,%d, A: %f, / %d    B: %f, / %d    C: %f / %d \t K: %d, M: %d, N: %d\n", x,y,a[lda * x + y],lda * x + y,b[ldb * y + x],ldb * y + x,c[ldc * y + x],ldc * y + x, k, m, n);
				c[ldc * x + y] = beta[0] * c[ldc * x + y] + cellSum;
			}
		}
	}
}

__device__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha, double* beta){
	int lda = a->height;
	int ldb = b->height;
	int ldc = m;
	multiplyCuda(a->elements, b->elements, c->elements,lda, ldb, ldc, m, n, k, op1, op2, alpha, beta);
	//printf("%d", c->height);
	//c->height = ldc;
	//c->width = k;
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
//
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




h_Matrix* copyMatrixToDevice(h_Matrix *hostMatrix){
	h_Matrix *deviceMatrix;
	double *deviceElements;

	// Allocate Space on Device for Array
	cudaCheck(cudaMalloc(&deviceElements, hostMatrix->numel() * sizeof(double)));
	hostMatrix->devElements = deviceElements;

	// Allocate Space on Device for Class Container
	cudaCheck(cudaMalloc((void **)&deviceMatrix, sizeof(h_Matrix)));

	// Copy contents of host matrix to the device matrix container
	cudaCheck(cudaMemcpy(deviceMatrix, &hostMatrix, sizeof(h_Matrix), cudaMemcpyHostToDevice));

	// Copy Contents of Array from host to device
	cudaCheck(cudaMemcpy(deviceElements, hostMatrix->elements, hostMatrix->numel() * sizeof(double), cudaMemcpyHostToDevice));

	// Copy address of array to matrix elements
	cudaCheck(cudaMemcpy(&(deviceMatrix->elements), &deviceElements, sizeof(double *), cudaMemcpyHostToDevice));

	// Copy remaining members to device
	cudaCheck(cudaMemcpy(&(deviceMatrix->height), &hostMatrix->height, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(&(deviceMatrix->width), &hostMatrix->width, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(&(deviceMatrix->depth), &hostMatrix->depth, sizeof(int), cudaMemcpyHostToDevice));

	return deviceMatrix;
}

void copyMatrixToHost(h_Matrix *hostMatrix, h_Matrix *deviceMatrix){

	double *hostElements;

	// Copy Matrix Properties
	cudaCheck(cudaMemcpy(&hostMatrix->height, &deviceMatrix->height, sizeof(int),cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(&hostMatrix->width, &deviceMatrix->width, sizeof(int),cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(&hostMatrix->depth, &deviceMatrix->depth, sizeof(int),cudaMemcpyDeviceToHost));


	// Copy Elements from Device to Host)
	if(!hostMatrix->elements){
		delete [] hostMatrix->elements;
	}

	hostElements = new double[hostMatrix->numel()];
	hostMatrix->elements = hostElements;

	// Copy the value (address) oprintf("\n %d,%d, %d, %d, %d, %f" , threadIdx.x, threadIdx.y, hnew->height, hnew->width, hnew->depth, cc[0]);f elements on the device to the host devElements attribute
	cudaCheck(cudaMemcpy(&hostMatrix->devElements, &deviceMatrix->elements, sizeof(double *), cudaMemcpyDeviceToHost));

	// Copy the elements from the device to the elements container on the host
	cudaCheck(cudaMemcpy(hostMatrix->elements, hostMatrix->devElements,sizeof(double) * hostMatrix->numel(),cudaMemcpyDeviceToHost));

}

h_Matrix copyMatrixToHost(h_Matrix *deviceMatrix){
	//Allocate Space on Host for Class Container)
	h_Matrix hostMatrix;
	copyMatrixToHost(&hostMatrix, deviceMatrix);
	return hostMatrix;
}

// destMatrix Elements must be initalized
__device__ int makeCopyOfMatrixElements(h_Matrix* destMatrix, h_Matrix* srcMatrix){
	if(destMatrix->numel() != srcMatrix->numel() || destMatrix->height != srcMatrix->height || destMatrix->width != srcMatrix->width){
		return -1;
	}
	for(int i = (srcMatrix->numel() % (BLOCK_WIDTH*BLOCK_WIDTH) == 0)? 0 : -1 ; i < srcMatrix->numel() / (BLOCK_WIDTH*BLOCK_WIDTH); i++){
		int index = i * (threadIdx.x * destMatrix->height + threadIdx.y);
		destMatrix->elements[index] = srcMatrix->elements[index];
	}
	return 0;
}

__device__ void d_IPSP3d(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){

	double scalar = 1;
	__shared__ int n1, l3;
	__shared__ h_Matrix aMatrix;
	__shared__ double *aMatrixElements;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		n1 = v1->width;
		l3 = v3->height;
		aMatrixElements = new double[re->width]();
		aMatrix.elements = aMatrixElements;__syncthreads();

		aMatrix.width = 8; aMatrix.height = aMatrix.depth = 1;
	}
	__syncthreads();

	for(int i = 0; i < n1; i++){
		cc->setElement(i, 0.0);
		for(int j = 0; j < l3; j++){
			h_Matrix v1Col = v1->getCol(i);
			h_Matrix v2Col = v2->getCol(i);
			matrixMultiplyCuda(&v1Col, &re->getPlane(j), &aMatrix, 1, re->width, v1->height, CUBLAS_OP_T, CUBLAS_OP_N, &scalar);
			__syncthreads();
			matrixMultiplyCuda(&aMatrix, &v2Col, cc->getElement(i), 1, 1, aMatrix.width, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &scalar);
			__syncthreads();
			//if (threadIdx.x == 0){printf("i: %d, j: %d, V3 Val:%f, \n", i, j, v3->getElement(i,j)[0]);}
		}
	}
	return;
}

__global__ void d_IPSP3dKernel(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){
	d_IPSP3d(re, v1, v2, v3, cc);
}


__device__ void d_clearMatrix(h_Matrix* matrix){
	for(int i = (matrix->numel() % (BLOCK_WIDTH*BLOCK_WIDTH) == 0)? 0 : -1 ; i < matrix->numel() / (BLOCK_WIDTH*BLOCK_WIDTH); i++){
		if(i * (threadIdx.x * matrix->height + threadIdx.y) < matrix->numel()){
				matrix->elements[i * (threadIdx.x * matrix->height + threadIdx.y)] = 0;
		}
	}
}

__device__ void d_findMaxInd(h_Matrix* matrix, int* maxInd, double* maxVal){
	int numberOfElements = matrix->numel();
	int elementsPerThread = (numberOfElements/(BLOCK_WIDTH * BLOCK_WIDTH));

	if(numberOfElements % (BLOCK_WIDTH * BLOCK_WIDTH) != 0){
		elementsPerThread += 1; //Add 1 incase number of elements is not divisible by block dimensions
	}

	int localMaxInd = 0;
	double localMaxVal = 0;

	__shared__ int* sharedMaxInd;
	__shared__ double* sharedMaxVal;



	// Split array into equal sizes to be processed by each thread

	for(int i = 0; i < elementsPerThread; i++){
		int index = elementsPerThread * (threadIdx.y * BLOCK_WIDTH + threadIdx.x) + i;
		if(index < numberOfElements){
			if(abs(matrix->elements[index]) > localMaxVal) {
				localMaxInd = index;
				localMaxVal = matrix->elements[index];
			}
		}
	}

	if(threadIdx.x == 0 && threadIdx.y == 0){
			sharedMaxInd = new int[BLOCK_WIDTH * BLOCK_WIDTH]();
			sharedMaxVal = new double[BLOCK_WIDTH * BLOCK_WIDTH]();
	}

	__syncthreads();
	// Assign each threads result to a unique index (based on thread) in __shared__ memory space
	sharedMaxInd[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = localMaxInd;
	sharedMaxVal[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = localMaxVal;
	__syncthreads();

	// Use just one thread to process the reduced array. Could possibly be done recursively
	if(threadIdx.x == 0 && threadIdx.y == 0){
		for(int i = 0; i < BLOCK_WIDTH * BLOCK_WIDTH; i++){
			if(abs(sharedMaxVal[i]) > abs(maxVal[0])){
				maxVal[0] = sharedMaxVal[i];
				maxInd[0] = sharedMaxInd[i];
			}
		}

		delete [] sharedMaxInd;
		delete [] sharedMaxVal;
	}
}

__global__ void d_findMaxIndKernel(h_Matrix* matrix,int* maxInd, double* maxVal){
	d_findMaxInd(matrix, maxInd, maxVal);
}

__device__ void d_IP3d(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){
	double scalar = 1;
	__shared__ h_Matrix aMatrix;
	__shared__ double *aMatrixElements;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		// Initialise shared calculation matrix
		aMatrixElements = new double[v1->width * re->width]();
		aMatrix.elements = aMatrixElements;
		aMatrix.width = re->width; aMatrix.height = v1->width; aMatrix.depth = 1;
	}
	__syncthreads();

	for(int i = 0; i < v3->width; i++){
		for(int j = 0; j < v3->height; j++){
			matrixMultiplyCuda(v1, &re->getPlane(j), &aMatrix, v1->width, v1->height, re->width, CUBLAS_OP_T, CUBLAS_OP_N, &scalar);
			__syncthreads();
			matrixMultiplyCuda(&aMatrix, v2, &cc->getPlane(i), aMatrix.height, v2->width, v2->height, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &scalar);
			__syncthreads();}
	}

	return;
}

__device__ void d_IP3d_max(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc, double* maxVal, int* maxInd){
	double zero = 0;
	double one = 1;
	__shared__ h_Matrix aMatrix;
	__shared__ double *aMatrixElements;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		// Initialise shared calculation matrix
		aMatrixElements = new double[v1->width * re->width]();
		aMatrix.elements = aMatrixElements;
		aMatrix.width = re->width; aMatrix.height = v1->width; aMatrix.depth = 1;
	}
	__syncthreads();

	for(int i = 0; i < v3->width; i++){
		for(int j = 0; j < v3->height; j++){
			matrixMultiplyCuda(v1, &re->getPlane(j), &aMatrix, v1->width, v1->height, re->width, CUBLAS_OP_T, CUBLAS_OP_N, &one);
			__syncthreads();
			if(j == 0){
				matrixMultiplyCuda(&aMatrix, v2, cc, aMatrix.height, v2->width, v2->height, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &zero);
			}else{
				matrixMultiplyCuda(&aMatrix, v2, cc, aMatrix.height, v2->width, v2->height, CUBLAS_OP_N, CUBLAS_OP_N, v3->getElement(i, j), &one);
			}
			__syncthreads();
			d_findMaxInd(cc, maxInd, maxVal);
			SINGLE_THREAD{ printf("\n%f, %f, %f, %d, %d\n", maxVal[0], aMatrix.elements[0], cc->elements[0], cc->height, cc->width); }
		}
	}

	return;
}

__global__ void d_IP3dKernel(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){
	d_IP3d(re, v1, v2, v3, cc);
}

__global__ void d_IP3d_maxKernel(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc, double* max, int* ind){
	d_IP3d_max(re, v1, v2, v3, cc, max, ind);
}

__device__ void d_hnew3d(double cc, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* hnew){
	for(int zk = 0; zk < v3->numel(); zk++){
		matrixMultiplyCuda(v1, v2, &hnew->getPlane(zk), v1->height, v1->height, v2->width, CUBLAS_OP_N, CUBLAS_OP_T, v3->elements[zk] * cc, 0.0);
	}
}

__global__ void d_hnew3dKernel(double* cc, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* hnew){
	d_hnew3d(cc[0], v1, v2, v3, hnew);
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


	// Split array into equal sizes to be processed by each thread
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
		for(int i = 0; i < BLOCK_WIDTH * BLOCK_WIDTH; i++){
			sum[0] += sharedSquaredSum[i];
		}
	}
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

__device__ void d_ProjMP3d(h_Matrix* h, h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* c, double* toln, int* max){


	double delta 	= 	1 / (double)(v1->height * v2->height * v3->height);
	double tol2		=	1e-11;
	__shared__ h_Matrix* hnew;
	__shared__ h_Matrix* cc;
	__shared__ double* hnewElements;
	__shared__ double* ccElements;

	printf("%d,%d,%d     %d,%d,%d     ", re->height, re->width, re->depth, v1->height, v1->width, v1->depth);

	if(threadIdx.x == 0 && threadIdx.y == 0){

		hnew = new h_Matrix(v1->height, v2->height, v3->height);
		hnewElements = new double[hnew->numel()]();
		hnew->elements = hnewElements;
		cc = new h_Matrix(1, v1->width, 1);


	}
	if(threadIdx.x == 0 && threadIdx.y == 1){
	for(int i = 0; i < v1->numel(); i++){
				printf("%f, ", v1->elements[i]);
			}
	}
	__syncthreads();

	for(int it = 0; it < max[0]; it++){
		if(threadIdx.x == 0 && threadIdx.y == 0){
			if(!ccElements){ delete [] ccElements; };
			ccElements = new double[cc->numel()]();
			cc->elements = ccElements;
		}

		__syncthreads();

		d_IPSP3d(re, v1, v2, v3, cc );
		__syncthreads();

		__shared__ int maxInd;
		__shared__ double maxVal;

		if(threadIdx.x == 0 && threadIdx.y == 0){
			maxInd = 0;
			maxVal = 0.0;
			d_findMaxInd(cc, &maxInd, &maxVal);
		}

	__syncthreads();

//	printf("%d, %f\t", maxInd, maxVal);
	if(threadIdx.x == 0 && threadIdx.y == 0){
		printf("\n%g, %g\t\n", maxVal, tol2);
		for(int i = 0; i < cc->numel(); i++){
			printf("%f, ", cc->elements[i]);
		}
		printf("\n\n\n");
	}
	if (maxVal < tol2){ printf("breaking maxVal"); break; }

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
//		for(int i = 0; i < hnew->numel(); i++){
//					printf("%f, ", hnew->elements[i]);
//		}
		printf("\n");
	}



	d_matrixAddition(h, hnew, h);
	d_matrixSubtraction(re, hnew, re);
	d_squaredSum(hnew, &nornu);

	__syncthreads();
//	if(threadIdx.x == 0 && threadIdx.y == 0){
//		for(int i = 0; i < re->numel(); i++){
//			printf("%f, ", re->elements[i]);
//		}http://s.symcb.com/universal-root.crl
//	}

	if(threadIdx.x == 0 && threadIdx.y == 0){
		printf("%f, \n", nornu);

	}
	if (nornu * delta <= toln[0]){ break; }
	}
}

__global__ void d_ProjMP3dKernel(h_Matrix* h, h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* c, double* toln, int* max){
	d_ProjMP3d(h, re, v1, v2, v3, c, toln, max);
}


__global__ void d_SPMP3D(h_Matrix* f, h_Matrix* Vx, h_Matrix* Vy, h_Matrix* Vz,
						double* tol, double* No, double* toln, int* lstep, int* Max, int* Maxp,
						h_Matrix* h, h_Matrix* c, double* Set_ind, int* numat){
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

		double* Di1;
		double* Di2;
		double* Di3;

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
		double* cpElements;
		__shared__ h_Matrix* cp;
		__shared__ double* ccElements;
		__shared__ h_Matrix* cc;
		__shared__ h_Matrix* Re;
		__shared__ double* ReElements;
		__shared__ double tol2;
		__shared__ int imp;
		__shared__ int Maxit2;

		SINGLE_THREAD{
			cpElements = new double[Vx->width * Vy->width * Vz->width];
			cp = new h_Matrix(cpElements, Vx->width, Vy->width, Vz->width);

			ccElements = new double[Vx->width * Vy->width]; // Reduced size so it can be shared
			cc = new h_Matrix(ccElements, Vx->width, Vy->width,1);

			ReElements  = new double[f->numel()];
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

		SINGLE_THREAD{
			h_new_elements = new double[Vx->height * Vy->height * Vz->height]();
			h_new = new h_Matrix(h_new_elements, Vx->height, Vy->height, Vz->height);

			multResultElements = new double[Vx->height];
			multResult = new h_Matrix(multResultElements, 1, Vx->height, 1);
		}

		__shared__ int it;
		__shared__ int s;
		//for((threadIdx.x == 0 && threadIdx.y == 0) ? it = 0 : it; it < Maxit2; (threadIdx.x == 0 && threadIdx.y == 0) ? it++ : it){
		SINGLE_THREAD{ it = s = 0; }

		__syncthreads();
		while(it < Maxit2){
			__syncthreads();
			while(s < lstep[0]){
				// Not including custom indexing processing since it is not priority
				__shared__ double maxVal;
				__shared__ int maxInd;
				SINGLE_THREAD{ maxVal = 0; maxInd = 0; }
				__syncthreads();
				d_IP3d_max(Re, Vx, Vy, Vz, cc, &maxVal, &maxInd); // Returns maxVal as normal form, i.e. not absolute. Hence cscra = maxVal, maxVal = abs(maxVal)
				SINGLE_THREAD{ ind2sub(cc->height, cc->width, cc->depth, maxInd, q); }

				__syncthreads();
				if(abs(maxVal) < tol2){
					SINGLE_THREAD{
						printf("SPMP3D stopped, max(|<f,q|/||q||) <= tol2 %g,%g", maxVal, tol2);
						delete [] cc;
						delete [] cp;
						delete [] multResult;
						delete [] h_new;
						delete [] Re;
					}
					__syncthreads();
					return;
				}


				// Has the indice been stored already, if not add it.
				if (numat[0] == 0){
					SINGLE_THREAD{
						Set_ind[0] = q[0];
						Set_ind[1] = q[1];
						Set_ind[2] = q[2];
						numat[0] += 1;
					}
				}else{
					SINGLE_THREAD{ // Set_ind is small reducing via multiple threads would take longer
						int exists = 0;
						for(int k = 0; k < numat[0]; k++){
							if(Set_ind[k * 3] == q[0] && Set_ind[k * 3 + 1] == q[1] && Set_ind[k * 3 + 2] == q[2]){
								exists = 1;
							}
						}

						if(exists == 0){
						 Set_ind[(numat[0]) * 3] = q[0];
						 Set_ind[(numat[0]) * 3 + 1] = q[1];
						 Set_ind[(numat[0]) * 3 + 2] = q[2];
						 numat[0] += 1;
						}
					}
				}

				//d_hnew3d(maxVal, Vx, Vy, Vz, h_new);

				SINGLE_THREAD{
					cp->elements[(cp->height*cp->width) * q[2] + (cp->height * q[1]) + q[0]] += maxVal;
				}

				__syncthreads();

				d_matrixAddition(h, h_new, h);
				d_matrixSubtraction(Re, h_new, Re);

				__syncthreads();

				__shared__ double nor_new;
				d_squaredSum(Re, &nor_new);
				__syncthreads();
				if(numat[0] >= No[0] || ((nor_new * delta) < tol[0])) break;

				SINGLE_THREAD{ s++; }
				__syncthreads();
			}
			SINGLE_THREAD{ it++; }
			__syncthreads();
		}
		printf("test");








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
//	std::cout << "Copying to Device.." << std::endl;
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
