#ifndef IPSP3D
#define IPSP3D
#define cudaCheck(input){cudaAssert((input), __FILE__, __LINE__); } // http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

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
};



__device__ void multiplyCuda(double* a, double* b, double* c, int lda, int ldb, int ldc, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha, double* beta){
	// m is number of rows of op(a)
	// n is number of cols of op(b)
	// k is number of rows of a and width of c)

	int y = threadIdx.y; //col
	int x = threadIdx.x; //row

	if (y < m && x < n){
	   double cellSum = 0;
			 if (op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_N){
				 for(int i = 0; i < k; i++){
					cellSum += a[lda * i + y] * b[ldb * x + i] * alpha[0];
					//printf("\ni: %d, y: %d, x: %d, lda: %d, ldb: %d, alpha: %f, temp: %f, cellSum: %f, aVal: %f, bVal: %f, aind %d, bind: %d", i, y, x, lda, ldb, alpha[0], a[lda * i + y] * b[ldb * x + i] * alpha[0], cellSum, a[lda * i + y], b[ldb * x + i], lda*i+y, ldb*x+i);
				 }
			 }else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_N){
				 for(int i = 0; i < k; i++){
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
			c[ldc * y + x] = beta[0] * c[ldc * y + x] + cellSum;
			//printf("\nThreadID %d,%d, A: %f, / %d    B: %f, / %d    C: %f / %d \t K: %d, M: %d, N: %d\n", x,y,a[lda * x + y],lda * x + y,b[ldb * y + x],ldb * y + x,c[ldc * y + x],ldc * y + x, k, m, n);
   }
}

__device__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha, double* beta){
	int lda = a->height;
	int ldb = b->height;
	int ldc = m;
	multiplyCuda(a->elements, b->elements, c->elements,lda, ldb, ldc, m, n, k, op1, op2, alpha, beta);
	c->height = ldc;
	c->width = k;
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

    // Copy the value (address) of elements on the device to the host devElements attribute
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




__global__ void d_IPSP3d(h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){

	double scalar = 1;
	__shared__ int n1, l3;)
    __shared__ h_Matrix aMatrix;
	__shared__ double *aMatrixElements;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		n1 = v1->width;
		l3 = v3->height;
		aMatrixElements = new double[re->width]();
		aMatrix.elements = aMatrixElements;
		aMatrix.width =; 8; aMatrix.height = aMatrix.depth = 1;
	}

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


int main() {


    double dxElements[] = {0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.4976, 0.4785, 0.4410, 0.3865, 0.3172, 0.2357, 0.1451, 0.0490, 0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904, 0.4785, 0.3172, 0.0490, -0.2357, -0.4410, -0.4976, -0.3865, -0.1451, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619, 0.4410, 0.0490, -0.3865, -0.4785, -0.1451, 0.3172, 0.4976, 0.2357, 0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157, 0.3865, -0.2357, -0.4785, 0.0490, 0.4976, 0.1451, -0.4410, -0.3172, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3172, -0.4410, -0.1451, 0.4976, -0.0490, -0.4785, 0.2357, 0.3865, 0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778, 0.2357, -0.4976, 0.3172, 0.1451, -0.4785, 0.3865, 0.0490, -0.4410, 0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1451, -0.3865, 0.4976, -0.4410, 0.2357, 0.0490, -0.3172, 0.4785, 0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975, 0.0490, -0.1451, 0.2357, -0.3172, 0.3865, -0.4410, 0.4785, -0.4976, 0.0490, 0.1451, 0.2357, 0.3172, 0.3865, 0.4410, 0.4785, 0.4976, 0.0975, 0.2778, 0.4157, 0.4904, 0.4904, 0.4157, 0.2778, 0.0975, 0.1451, 0.3865, 0.4976, 0.4410, 0.2357, -0.0490, -0.3172, -0.4785, 0.1913, 0.4619, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.2357, 0.4976, 0.3172, -0.1451, -0.4785, -0.3865, 0.0490, 0.4410, 0.2778, 0.4904, 0.0975, -0.4157, -0.4157, 0.0975, 0.4904, 0.2778, 0.3172, 0.4410, -0.1451, -0.4976, -0.0490, 0.4785, 0.2357, -0.3865, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3865, 0.2357, -0.4785, -0.0490, 0.4976, -0.1451, -0.4410, 0.3172, 0.4157, 0.0975, -0.4904, 0.2778, 0.2778, -0.4904, 0.0975, 0.4157, 0.4410, -0.0490, -0.3865, 0.4785, -0.1451, -0.3172, 0.4976, -0.2357, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1913, -0.4619, 0.4785, -0.3172, 0.0490, 0.2357, -0.4410, 0.4976, -0.3865, 0.1451, 0.4904, -0.4157, 0.2778, -0.0975, -0.0975, 0.2778, -0.4157, 0.4904, 0.4976, -0.4785, 0.4410, -0.3865, 0.3172, -0.2357, 0.1451, -0.0490, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000};
    double dyElements[] = {0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.4976, 0.4785, 0.4410, 0.3865, 0.3172, 0.2357, 0.1451, 0.0490, 0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904, 0.4785, 0.3172, 0.0490, -0.2357, -0.4410, -0.4976, -0.3865, -0.1451, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619, 0.4410, 0.0490, -0.3865, -0.4785, -0.1451, 0.3172, 0.4976, 0.2357, 0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157, 0.3865, -0.2357, -0.4785, 0.0490, 0.4976, 0.1451, -0.4410, -0.3172, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3172, -0.4410, -0.1451, 0.4976, -0.0490, -0.4785, 0.2357, 0.3865, 0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778, 0.2357, -0.4976, 0.3172, 0.1451, -0.4785, 0.3865, 0.0490, -0.4410, 0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1451, -0.3865, 0.4976, -0.4410, 0.2357, 0.0490, -0.3172, 0.4785, 0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975, 0.0490, -0.1451, 0.2357, -0.3172, 0.3865, -0.4410, 0.4785, -0.4976, 0.0490, 0.1451, 0.2357, 0.3172, 0.3865, 0.4410, 0.4785, 0.4976, 0.0975, 0.2778, 0.4157, 0.4904, 0.4904, 0.4157, 0.2778, 0.0975, 0.1451, 0.3865, 0.4976, 0.4410, 0.2357, -0.0490, -0.3172, -0.4785, 0.1913, 0.4619, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.2357, 0.4976, 0.3172, -0.1451, -0.4785, -0.3865, 0.0490, 0.4410, 0.2778, 0.4904, 0.0975, -0.4157, -0.4157, 0.0975, 0.4904, 0.2778, 0.3172, 0.4410, -0.1451, -0.4976, -0.0490, 0.4785, 0.2357, -0.3865, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3865, 0.2357, -0.4785, -0.0490, 0.4976, -0.1451, -0.4410, 0.3172, 0.4157, 0.0975, -0.4904, 0.2778, 0.2778, -0.4904, 0.0975, 0.4157, 0.4410, -0.0490, -0.3865, 0.4785, -0.1451, -0.3172, 0.4976, -0.2357, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1913, -0.4619, 0.4785, -0.3172, 0.0490, 0.2357, -0.4410, 0.4976, -0.3865, 0.1451, 0.4904, -0.4157, 0.2778, -0.0975, -0.0975, 0.2778, -0.4157, 0.4904, 0.4976, -0.4785, 0.4410, -0.3865, 0.3172, -0.2357, 0.1451, -0.0490, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000};
    double dzElements[] = {0.5774,0.5774,0.5774,0.7887,0.5774,0.2113,0.7071,0.0000,-0.7071,0.5774,-0.5774,-0.5774,0.4082,-0.8165,0.4082,0.2113,-0.5774,0.7887,0.2113,0.5774,0.7887,0.4082,0.8165,0.4082,0.5774,0.5774,-0.5774,0.7071,0.0000,-0.7071,0.7887,-0.5774,0.2113,0.5774,-0.5774,0.5774,1.0000,0,0,0,1.0000,0,0,0,1.0000};
    double reElements[] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6};
    double* ccElements = new double[40]();

    printDeviceDetails();

	//Initialise Matrix containers
    std::cout << "Initializing.." << std::endl;
    h_Matrix h_dx(dxElements, 8, 1, 1);
    h_Matrix h_dy(dyElements, 8, 1, 1);
    h_Matrix h_dz(dzElements, 3, 1, 1);
    h_Matrix h_re(reElements, 8, 8, 8);
    h_Matrix h_cc(ccElements, 1, 40, 1);

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*32);

    //Copy Matrix's to    std::cout << "Device Details... \n" << std::endl;
    std::cout << "Copying to Device.." << std::endl;
    h_Matrix* d_dx = copyMatrixToDevice(&h_dx);
    h_Matrix* d_dy = copyMatrixToDevice(&h_dy);
    h_Matrix* d_dz = copyMatrixToDevice(&h_dz);
    h_Matrix* d_re = copyMatrixToDevice(&h_re);
    h_Matrix* d_cc = copyMatrixToDevice(&h_cc);

    std::cout << "Starting.." <<std::endl;

    double* cElements = new double[h_re.width]();
    h_Matrix h_c(cElements, 1, 8, 1);
    h_Matrix* d_c = copyMatrixToDevice(&h_c);

    double *devScalar;
    double scalar = 1;

    cudaCheck( cudaMalloc(&devScalar, sizeof(double)));
    cudaCheck( cudaMemcpy(devScalar, &scalar, sizeof(double), cudaMemcpyHostToDevice));


    //status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, h_dx.height, h_re.height, &scalar, &d_dx->elements[0], h_dx.height, &d_re->elements[0], h_re.height, 0, &aMatrix[0], 1);
    //status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, h_re.height, h_re.height, 1, &scalar, &d_dx->elements[0], 1, &d_re->elements[0], 1, 0, &aMatrix[0], h_re.height);
    //MatrixMultiplyBLAS(handle, d_dx->elements, d_re->elements, aMatrix, h_dx.height, 1, h_re.width, CUBLAS_OP_T, CUBLAS_OP_N);
    //matrixMultiplyCuda<<<1,threadsPerBlock>>>(d_dx, d_re, d_c, CUBLAS_OP_T, CUBLAS_OP_N, devScalar);

    dim3 threadsPerBlock(8, 8);
    for(int i = 0; i < 1000; i++){
    d_IPSP3d<<< 1, threadsPerBlock>>>(d_re, d_dx, d_dy, d_dz, d_cc);
    }


    //matrixMultiplyCudaKernel<<<1,threadsPerBlock>>>(d_dx, d_re, d_c, 1, h_re.width, h_dx.height, CUBLAS_OP_T, CUBLAS_OP_N, devScalar);
   // double* testOutput = new double[h_dx.height * h_re.width]();

    // Either works :)
    //h_Matrix results = 	copyMatrixToHost(d_cc);
    copyMatrixToHost(&h_cc, d_cc);

    printf("%d,%d,%d, %f\n", h_cc.height, h_cc.width, h_cc.depth, h_cc.elements[0]);

    for(int i = 0; i < h_cc.height * h_cc.width * h_cc.depth; i++){
    	std::cout << h_cc.elements[i] << ", " << std::endl;
    }

    cudaError_t err = cudaGetLastError();
    
    printf("\nError: %s\n", cudaGetErrorString(err));

    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_dz);
    cudaFree(d_re);
    cudaFree(d_cc);
    return 0;
}

#endif

