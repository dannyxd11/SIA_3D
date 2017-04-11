#ifndef CUDA_HELPERS
#define CUDA_HELPERS

#include <iostream>
#include <stdlib.h>
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

int getNumberOfSMPs(){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		return prop.multiProcessorCount;
	}
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
		printf("  CUDA Version: %d.%d\n",  prop.major, prop.minor);
		printf("  Number of MultiProcessors: %d\n\n", prop.multiProcessorCount);
	}
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
	for(int i = 0 ; i < (srcMatrix->numel() / (BLOCK_WIDTH*BLOCK_WIDTH)) + ((srcMatrix->numel() % (BLOCK_WIDTH*BLOCK_WIDTH) == 0) ? 0 : +1); i++){
		int index = i * (BLOCK_WIDTH * BLOCK_WIDTH) + (threadIdx.x * srcMatrix->height + threadIdx.y);
		destMatrix->elements[index] = srcMatrix->elements[index];
	}
	return 0;
}

__device__ int extractBlock(h_Matrix* destMatrix, h_Matrix* srcMatrix, int blockSize){

	for(int dim = 0 ; dim < srcMatrix->depth; dim++){
		int dimModifier = dim * srcMatrix->height * srcMatrix->width;
		int index = dimModifier + (blockIdx.x * srcMatrix->height * blockSize) + (threadIdx.x * srcMatrix->height) + (blockIdx.y * blockSize) + threadIdx.y;
		int destIndex = (dim * blockSize * blockSize) + (threadIdx.x * blockSize) + threadIdx.y;
		if(index < srcMatrix->numel() && destIndex < destMatrix->numel()){
			destMatrix->elements[destIndex] = srcMatrix->elements[index];
		}
	}
	return 0;
}

__device__ int moveBetweenShared(h_Matrix* destMatrix, h_Matrix* srcMatrix, int blockSize){
	for(int i = 0; i <= srcMatrix->numel() / (blockSize * blockSize); i++){
		int index = i * blockSize * blockSize + threadIdx.x * srcMatrix->height + threadIdx.y;
		
		if(index < srcMatrix->numel()){
			destMatrix->elements[index] = srcMatrix->elements[index];
		}
	}
	__syncthreads();
}

__device__ uint get_smid(void) { /* https://devtalk.nvidia.com/default/topic/481465/any-way-to-know-on-which-sm-a-thread-is-running-/ */ 

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}


#endif
