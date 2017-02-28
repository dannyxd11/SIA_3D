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



class h_Matrix{
public:
    double* elements;
    double* devElements;
    int height, width, depth;
    h_Matrix(int height, int width, int depth) : height(height), width(width), depth(depth) { elements = new double[height*width*depth](); };
    h_Matrix(double* elements, int height, int width, int depth) : height(height), width(width), depth(depth), elements(elements) {};
    int numel (){return height * width * depth;};
};

__device__ void multiplyCuda(double* a, double* b, double* c, int lda, int ldb, int ldc, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
	    int y = threadIdx.y;
	    int x = threadIdx.x;

		if (y < m && x < n){
	       double cellSum = 0;
				 if (op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_N){
					 for(int i = 0; i < k; i++){
						cellSum += a[lda * i + y] * b[ldb * x + i] * alpha[0];
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
				c[ldc * x + y] = cellSum;
				printf("ThreadID %d,%d, A: %f, / %d    B: %f, / %d    C: %f / %d\n", x,y,a[lda * x + y],lda * x + y,b[ldb * y + x],ldb * y + x,c[ldc * y + x],ldc * y + x);
	   }
}

__global__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
	int lda = a->height;
	int ldb = b->height;
	int ldc = m;
	multiplyCuda(a->elements, b->elements, c->elements,lda, ldb, ldc, m, n, k, op1, op2, alpha);
}

__global__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
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

//
//   if (y < m && x < n){
//       double cellSum = 0;
//			 if (op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_N){
//				 for(int i = 0; i < k; i++){
//					cellSum += a->elements[leadingDimensionA * i + y] * b->elements[leadingDimensionB * x + i];
//				 }
//			 }else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_N){
//				 for(int i = 0; i < k; i++){
//				 	cellSum += a->elements[leadingDimensionA * y + i] * b->elements[leadingDimensionB * x + i];
//				 }
//			 }else if(op1 == CUBLAS_OP_N && op2 == CUBLAS_OP_T){
//				 for(int i = 0; i < k; i++){
//				 	cellSum += a->elements[leadingDimensionA * i + y] * b->elements[leadingDimensionB * i + x];
//				 }
//			 }else if(op1 == CUBLAS_OP_T && op2 == CUBLAS_OP_T){
//				 for(int i = 0; i < k; i++){
//				 	cellSum += a->elements[leadingDimensionA * y + i] * b->elements[leadingDimensionB * i + x];
//				 }
//			 }
//			c->elements[leadingDimensionC * x + y] = cellSum;
//			printf("ThreadID %d,%d, A: %f, / %d    B: %f, / %d    C: %f / %d\n", x,y,a->elements[leadingDimensionA * x + y],leadingDimensionA * x + y,b->elements[leadingDimensionB * y + x],leadingDimensionB * y + x,c->elements[leadingDimensionC * y + x],leadingDimensionC * y + x);
//   }

	 multiplyCuda(a->elements, b->elements, c->elements, leadingDimensionA, leadingDimensionB, leadingDimensionC, m, n, k, op1, op2, alpha);

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


__global__
void d_IPSP3d(cublasHandle_t handle, h_Matrix* re, h_Matrix* v1, h_Matrix* v2, h_Matrix* v3, h_Matrix* cc){

	__shared__ int n1, l3;

	if(threadIdx.x == 0){
		n1 = v1->width;
		l3 = v3->height;
	}


 //   cublasStatus_t status;

 //   __shared__ double* c;
 //   __shared__ double scalar;
 //   __syncthreads();

//    if(threadIdx.x == 0){
//    	 c = new double[v2->height * re->width]();
//    	 scalar = 1;
//    	 printf("%f\n", cc->elements[0]);
//__global__ void matrixMultiplyCuda(h_Matrix* a, h_Matrix* b, h_Matrix* c, int m, int n, int k, cublasOperation_t op1, cublasOperation_t op2, double* alpha){
//    	 //status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, v2->height, re->height, &scalar, &v2->elements[0], v2->height, &re->elements[0], re->height, 0, c, v2->height);
//    	 printf("%f", c[0]);
//    	 if (status != CUBLAS_STATUS_SUCCESS){printf("Cublas DGEMM failure (v2xre), dan you fool\n"); return;}
//    	 if (status == CUBLAS_STATUS_SUCCESS){printf("Dan you genius\n"); return;}
//    	 printf("\n%f, %d", c[0], status);
//    	 cc->elements[0] = c[0];
//    }



/*
    //    for(int i = 0; i < n1; i ++){
//
//        for(int j = 0; j < l3; j++){
//            // V2(:,n)'*Re(:,:,zk)'  // height of V2 (1 since its a row) width of Re (Transposed so height)
//            double* c = new double(v2->height * re->width);
//            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, re->width, re->height, 1, v2->elements.data(), v2->height, re->elements.data(), re->height, 0, c,v2->height);
//
//        }
//    }
*/
    return;
}


int main() {


    double dxElements[] = {0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.4976, 0.4785, 0.4410, 0.3865, 0.3172, 0.2357, 0.1451, 0.0490, 0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904, 0.4785, 0.3172, 0.0490, -0.2357, -0.4410, -0.4976, -0.3865, -0.1451, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619, 0.4410, 0.0490, -0.3865, -0.4785, -0.1451, 0.3172, 0.4976, 0.2357, 0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157, 0.3865, -0.2357, -0.4785, 0.0490, 0.4976, 0.1451, -0.4410, -0.3172, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3172, -0.4410, -0.1451, 0.4976, -0.0490, -0.4785, 0.2357, 0.3865, 0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778, 0.2357, -0.4976, 0.3172, 0.1451, -0.4785, 0.3865, 0.0490, -0.4410, 0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1451, -0.3865, 0.4976, -0.4410, 0.2357, 0.0490, -0.3172, 0.4785, 0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975, 0.0490, -0.1451, 0.2357, -0.3172, 0.3865, -0.4410, 0.4785, -0.4976, 0.0490, 0.1451, 0.2357, 0.3172, 0.3865, 0.4410, 0.4785, 0.4976, 0.0975, 0.2778, 0.4157, 0.4904, 0.4904, 0.4157, 0.2778, 0.0975, 0.1451, 0.3865, 0.4976, 0.4410, 0.2357, -0.0490, -0.3172, -0.4785, 0.1913, 0.4619, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.2357, 0.4976, 0.3172, -0.1451, -0.4785, -0.3865, 0.0490, 0.4410, 0.2778, 0.4904, 0.0975, -0.4157, -0.4157, 0.0975, 0.4904, 0.2778, 0.3172, 0.4410, -0.1451, -0.4976, -0.0490, 0.4785, 0.2357, -0.3865, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3865, 0.2357, -0.4785, -0.0490, 0.4976, -0.1451, -0.4410, 0.3172, 0.4157, 0.0975, -0.4904, 0.2778, 0.2778, -0.4904, 0.0975, 0.4157, 0.4410, -0.0490, -0.3865, 0.4785, -0.1451, -0.3172, 0.4976, -0.2357, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1913, -0.4619, 0.4785, -0.3172, 0.0490, 0.2357, -0.4410, 0.4976, -0.3865, 0.1451, 0.4904, -0.4157, 0.2778, -0.0975, -0.0975, 0.2778, -0.4157, 0.4904, 0.4976, -0.4785, 0.4410, -0.3865, 0.3172, -0.2357, 0.1451, -0.0490, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000};
    double dyElements[] = {0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.4976, 0.4785, 0.4410, 0.3865, 0.3172, 0.2357, 0.1451, 0.0490, 0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904, 0.4785, 0.3172, 0.0490, -0.2357, -0.4410, -0.4976, -0.3865, -0.1451, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619, 0.4410, 0.0490, -0.3865, -0.4785, -0.1451, 0.3172, 0.4976, 0.2357, 0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157, 0.3865, -0.2357, -0.4785, 0.0490, 0.4976, 0.1451, -0.4410, -0.3172, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3172, -0.4410, -0.1451, 0.4976, -0.0490, -0.4785, 0.2357, 0.3865, 0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778, 0.2357, -0.4976, 0.3172, 0.1451, -0.4785, 0.3865, 0.0490, -0.4410, 0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1451, -0.3865, 0.4976, -0.4410, 0.2357, 0.0490, -0.3172, 0.4785, 0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975, 0.0490, -0.1451, 0.2357, -0.3172, 0.3865, -0.4410, 0.4785, -0.4976, 0.0490, 0.1451, 0.2357, 0.3172, 0.3865, 0.4410, 0.4785, 0.4976, 0.0975, 0.2778, 0.4157, 0.4904, 0.4904, 0.4157, 0.2778, 0.0975, 0.1451, 0.3865, 0.4976, 0.4410, 0.2357, -0.0490, -0.3172, -0.4785, 0.1913, 0.4619, 0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.2357, 0.4976, 0.3172, -0.1451, -0.4785, -0.3865, 0.0490, 0.4410, 0.2778, 0.4904, 0.0975, -0.4157, -0.4157, 0.0975, 0.4904, 0.2778, 0.3172, 0.4410, -0.1451, -0.4976, -0.0490, 0.4785, 0.2357, -0.3865, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3865, 0.2357, -0.4785, -0.0490, 0.4976, -0.1451, -0.4410, 0.3172, 0.4157, 0.0975, -0.4904, 0.2778, 0.2778, -0.4904, 0.0975, 0.4157, 0.4410, -0.0490, -0.3865, 0.4785, -0.1451, -0.3172, 0.4976, -0.2357, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913, 0.1913, -0.4619, 0.4785, -0.3172, 0.0490, 0.2357, -0.4410, 0.4976, -0.3865, 0.1451, 0.4904, -0.4157, 0.2778, -0.0975, -0.0975, 0.2778, -0.4157, 0.4904, 0.4976, -0.4785, 0.4410, -0.3865, 0.3172, -0.2357, 0.1451, -0.0490, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 0.3536, -0.3536, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000};
    double dzElements[] = {0.5774,0.5774,0.5774,0.7887,0.5774,0.2113,0.7071,0.0000,-0.7071,0.5774,-0.5774,-0.5774,0.4082,-0.8165,0.4082,0.2113,-0.5774,0.7887,0.2113,0.5774,0.7887,0.4082,0.8165,0.4082,0.5774,0.5774,-0.5774,0.7071,0.0000,-0.7071,0.7887,-0.5774,0.2113,0.5774,-0.5774,0.5774,1.0000,0,0,0,1.0000,0,0,0,1.0000};
    double reElements[] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6};
    double* ccElements = new double[40]();



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



	//Initialise Matrix containers
    std::cout << "Initializing.." << std::endl;
    h_Matrix h_dx(dxElements, 8, 40, 1);
    h_Matrix h_dy(dyElements, 8, 40, 1);
    h_Matrix h_dz(dzElements, 3, 40, 1);
    h_Matrix h_re(reElements, 8, 8, 8);
    h_Matrix h_cc(ccElements, 1, 40, 1);

    //Copy Matrix's to Device
    std::cout << "Copying to Device.." << std::endl;
    h_Matrix* d_dx = copyMatrixToDevice(&h_dx);
    h_Matrix* d_dy = copyMatrixToDevice(&h_dy);
    h_Matrix* d_dz = copyMatrixToDevice(&h_dz);
    h_Matrix* d_re = copyMatrixToDevice(&h_re);
    h_Matrix* d_cc = copyMatrixToDevice(&h_cc);

    cublasStatus_t stat;
    cublasHandle_t handle;
    std::cout << "Starting.." <<std::endl;
    cublasCreate(&handle);

    double* cElements = new double[h_re.width]();
    h_Matrix h_c(cElements, 1, 8, 1);


    h_Matrix* d_c = copyMatrixToDevice(&h_c);

    double *aMatrix;
    cudaCheck( cudaMalloc( &aMatrix, sizeof(double) * h_re.width));
    double scalar = 1;
    double* devScalar;
    cudaCheck( cudaMalloc(&devScalar, sizeof(double)));
    cudaCheck( cudaMemcpy(devScalar, &scalar, sizeof(double), cudaMemcpyHostToDevice));



    printf("%d,%d,%d,%d,%d", h_re.width, h_dx.width,h_dx.height,h_re.height, h_dx.height);

    //status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, h_dx.height, h_re.height, &scalar, &d_dx->elements[0], h_dx.height, &d_re->elements[0], h_re.height, 0, &aMatrix[0], 1);
    //status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, h_re.height, h_re.height, 1, &scalar, &d_dx->elements[0], 1, &d_re->elements[0], 1, 0, &aMatrix[0], h_re.height);
    //MatrixMultiplyBLAS(handle, d_dx->elements, d_re->elements, aMatrix, h_dx.height, 1, h_re.width, CUBLAS_OP_T, CUBLAS_OP_N);

    dim3 threadsPerBlock(8, 8);
    //matrixMultiplyCuda<<<1,threadsPerBlock>>>(d_dx, d_re, d_c, CUBLAS_OP_T, CUBLAS_OP_N, devScalar);
    matrixMultiplyCuda<<<1,threadsPerBlock>>>(d_dx, d_re, d_c, 1, h_re.width, h_dx.height, CUBLAS_OP_T, CUBLAS_OP_N, devScalar);
    double* testOutput = new double[h_dx.height * h_re.width]();
    cudaCheck(cudaMemcpy(testOutput, h_c.devElements, sizeof(double)  * h_re.width, cudaMemcpyDeviceToHost));


    for(int i = 0; i < h_c.height * h_c.width ; i++){
    	std::cout << testOutput[i] << ", " << std::endl;
    }
    //d_IPSP3d<<< 1, 64>>>(handle, d_re, d_dx, d_dy, d_dz, d_cc);

    cudaError_t err = cudaGetLastError();
    
    printf("Error: %s\n", cudaGetErrorString(err));

    //double *c = new double[1];
    //cudaMemcpy(c, d_cc->elements, sizeof(double *), cudaMemcpyDeviceToHost);
    //std::cout << c[0] << std::endl;
    cublasDestroy(handle);

    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_dz);
    cudaFree(d_re);
    cudaFree(d_cc);
    return 0;
}

#endif
