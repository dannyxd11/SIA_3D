#ifndef SPMP3D
#define SPMP3D

#include "h_Matrix.h"
#include "loadImage.cpp"
#include "CreateDict.cpp"
#include "Routines.cu"

#ifndef CUDACHECK
#define CUDACHECK
#define cudaCheck(input){cudaAssert((input), __FILE__, __LINE__); } // http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void cudaAssert(cudaError_t code, const char *file, int line){
	if (code != cudaSuccess){ fprintf(stderr, "CudaAssert %s %s %d\n", cudaGetErrorString(code), file, line); exit(code);}
}
#endif

void startSPMP3DRoutine(int blockSize, h_Matrix* inputImage, h_Matrix* Dx, h_Matrix* Dy, h_Matrix* Dz){
	int numberOfConcurrentBlocks = getNumberOfSMPs();

	dim3 threadsPerBlock(blockSize, blockSize,3);
	int nBlocks = (inputImage->height/8) * (inputImage->width/8);
	dim3 numberOfBlocks(inputImage->height/8, inputImage->width/8);
	//dim3 numberOfBlocks(3, 1);
	std::cout << "Initializing.." << std::endl;
	h_Matrix h_h((inputImage->height/8)*8, (inputImage->width/8)*8, (inputImage->depth/3)*3);

	double* h_c = new double[(inputImage->height/8)*8 * (inputImage->width/8)*8 *  (inputImage->depth/3)*3 * 2]();
//	h_Matrix h_c((inputImage->height/8)*8, (inputImage->width/8)*8, (inputImage->depth/3)*3);

	cudaCheck( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*256) );

	//Copy Matrix'sglobal to device
	std::cout << "Copying to Device.." << std::endl;
	//printf("%d, %d, %d\n", inputImage->height, inputImage->width, inputImage->depth);
	h_Matrix* d_f = copyMatrixToDevice(inputImage);
	h_Matrix* d_dx = copyMatrixToDevice(Dx);
	h_Matrix* d_dy = copyMatrixToDevice(Dy);
	h_Matrix* d_dz = copyMatrixToDevice(Dz);
//	h_Matrix* d_c = copyMatrixToDevice(&h_c);
	h_Matrix* d_h = copyMatrixToDevice(&h_h);

	//h_h.height = 8; h_h.width = 8; h_h.depth = 3;
	//h_c.height = 8; h_c.width = 8; h_c.depth = 3;
	double pss = 40.5;
	// Declare host variables
	double h_tol = pow(255,2)/(pow(10,pss/10.0));//5.7954;
	double h_No = inputImage->numel()/15.0;//1.3107e+4;
	double h_toln = 1e-8;
	int h_lstep = -1;
	int h_Max = 50000;
	//int h_Max = 5;
	int h_Maxp = 50000;
	//int h_Maxp = 5;
	int* h_Set_ind = new int[8*8*3]();
	int* h_numat = new int[nBlocks]();
	double* d_c;

	// Declare device variables
	double* d_tol;
	double* d_No;
	double* d_toln;
	int* d_lstep;
	int* d_Max;
	int* d_Maxp ;
	int* d_Set_ind;
	int* d_numat;

	// Allocate Space on device
	cudaCheck( cudaMalloc( &d_tol, sizeof(double) ) );
	cudaCheck( cudaMalloc( &d_No, sizeof(double) ) );
	cudaCheck( cudaMalloc( &d_toln, sizeof(double) ) );
	cudaCheck( cudaMalloc( &d_lstep, sizeof(int) ) );
	cudaCheck( cudaMalloc( &d_Max, sizeof(int) ) );
	cudaCheck( cudaMalloc( &d_Maxp, sizeof(int) ) );
	cudaCheck( cudaMalloc( &d_Set_ind, sizeof(int) * Dx->height * Dy->height * Dz->height ) );
	cudaCheck( cudaMalloc( &d_numat, sizeof(int) * nBlocks) );
	cudaCheck( cudaMalloc( &d_c, sizeof(double) * (inputImage->height/8)*8 * (inputImage->width/8)*8 *  (inputImage->depth/3)*3 * 2) );

	// Initalise values on device
	cudaCheck( cudaMemcpy( d_tol, &h_tol, sizeof(double), cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_No, &h_No, sizeof(double), cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_toln, &h_toln, sizeof(double), cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_lstep, &h_lstep, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_Max, &h_Max, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_Maxp, &h_Maxp, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_Maxp, &h_Maxp, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_Set_ind, h_Set_ind, sizeof(int) * 192, cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_numat, h_numat, sizeof(int) * nBlocks, cudaMemcpyHostToDevice));
	cudaCheck( cudaMemcpy( d_c, h_c, sizeof(double) * (inputImage->height/8)*8 * (inputImage->width/8)*8 *  (inputImage->depth/3)*3 * 2, cudaMemcpyHostToDevice));

	int blockDim = Dx->height * Dy->height * Dz->height;
	printf("Tolerance: %f, Max Iterations %d, PSS: %f, LStep: %d, h_No: %f\nXBlocks: %d, YBlocks:%d\n", h_tol, h_Max, pss, h_lstep, h_No,(inputImage->width/8),(inputImage->height/8));
	std::cout << "Calculating..";
	// *4 for extra c
	d_SPMP3DKernel<<< numberOfBlocks, threadsPerBlock,(blockDim * 4 + Dx->numel() + Dy->numel() + Dz->numel() + Dx->numel()) * sizeof(double) >>>(d_f, d_dx, d_dy, d_dz, d_tol, d_No, d_toln, d_lstep, d_Max, d_Maxp, d_h, d_c, d_Set_ind, d_numat);
	//d_SPMP3DKernel<<< 14, threadsPerBlock, (blockDim + Dx->numel() + Dy->numel() + Dz->numel() + Dx->numel() + h_h.numel() + h_c.numel()) * sizeof(double)>>>(d_f, d_dx, d_dy, d_dz, d_tol, d_No, d_toln, d_lstep, d_Max, d_Maxp, d_h, d_c, d_Set_ind, d_numat);
	cudaCheck( cudaDeviceSynchronize() );
	std::cout << "Done." << std::endl;
	std::cout << "Outcome.." << std::endl;
	cudaError_t err = cudaGetLastError();	
	std::cout << err << std::endl;
	
	cudaCheck( cudaMemcpy( h_numat, d_numat, sizeof(int) * nBlocks, cudaMemcpyDeviceToHost))
	int numberOfCoeffs = 0;
	for(int i = 0; i < nBlocks; i++){
		numberOfCoeffs += h_numat[i];
	}
	std::cout << "numberOfCoeffs.. " << numberOfCoeffs << std::endl;
	std::cout << "Sparsity Ratio.. " << inputImage->numel() / (double)numberOfCoeffs << std::endl;
//	std::cout << numberOfCoeffs << std::endl;
	delete [] h_Set_ind;

	cudaCheck( cudaFree(d_tol) );
	cudaCheck( cudaFree(d_No) );
	cudaCheck( cudaFree(d_toln) );
	cudaCheck( cudaFree(d_lstep) );
	cudaCheck( cudaFree(d_Max) );
	cudaCheck( cudaFree(d_Maxp) );
	cudaCheck( cudaFree(d_Set_ind) );
	cudaCheck( cudaFree(d_numat) );

	cudaCheck( cudaFree(d_f) );
	cudaCheck( cudaFree(d_dx) );
	cudaCheck( cudaFree(d_dy) );
	cudaCheck( cudaFree(d_dz) );
	cudaCheck( cudaFree(d_c) );
	cudaCheck( cudaFree(d_h) );


}

int main(int argc, char** argv )
{
    if ( argc != 2 ) { printf("usage: SPMP3D.out <Image_Path>\n"); return -1; }

    cudaCheck( cudaDeviceReset() );

    printDeviceDetails();

    //cudaCheck( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*32) );
    cudaCheck( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
    //cudaFuncSetCacheConfig(d_SPMP3D, inputImagecudaFuncCachePreferShared);
    cudaCheck( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );

    h_Matrix* inputImage = loadImageToMatrix(argv[1]);
 //   h_Matrix* testImage = new h_Matrix(inputImage->elements[8*8], inputImage->height, inputImage->width, inputImage->depth);

    h_Matrix Dx = createStandardDict();
    h_Matrix Dy = createStandardDict();
    h_Matrix Dz = createDzDict();

    startSPMP3DRoutine(8, inputImage, &Dx, &Dy, &Dz);

    //for(int i = 0; i < Dx.numel(); i ++){ printf("%f, ", Dx.elements[i]); }
    //std::cout << "\n\n";
    //for(int i = 0; i < Dz.numel(); i ++){ printf("%f, ", Dz.elements[i]); }
    //waitKey(0);
    delete inputImage;
    return 0;
}




#endif

