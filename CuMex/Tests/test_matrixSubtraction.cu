#include "../IPSP3D_cuda.cu"

int main() {
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*32);
	
	double reElements[] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6};

	printDeviceDetails();

	//Initialise matrix containers on host
	std::cout << "Initializing.." << std::endl;
	h_Matrix h_re(reElements, 8, 8, 3);
	h_Matrix h_calc(8,8,3);
	
	
	//Copy Matrix's global to device
	std::cout << "Copying to Device.." << std::endl;
	h_Matrix* d_re = copyMatrixToDevice(&h_re);
	h_Matrix* d_calc = copyMatrixToDevice(&h_calc);

	std::cout << "Starting Subtraction Test.." <<std::endl;

	dim3 threadsPerBlock(8, 8);

	d_matrixSubtractionKernel<<< 1, threadsPerBlock>>>(d_re, d_re, d_calc);

	cudaCheck(cudaDeviceSynchronize());

	copyMatrixToHost(&h_calc, d_calc);	

	for(int i = 0; i < h_calc.numel(); i++){
		if(h_calc.elements[i] != 0){
			std::cout << "Actual does not match expected" << std::endl;
			return -1;
		}
	}

	cudaError_t err = cudaGetLastError();

	printf("\nError: %s\n", cudaGetErrorString(err));

	cudaFree(d_re);
	return 0;
}
