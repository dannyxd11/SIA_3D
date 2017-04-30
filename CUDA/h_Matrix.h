#ifndef h_Matrix_guard
#define h_Matrix_guard
#ifdef __CUDACC__
#define CUDA_HOST_DEV __host__ __device__
#else
#define CUDA_HOST_DEV
#endif


class h_Matrix{
public:
    double* elements;
    double* devElements;
    int preventFree;
    int height, width, depth;
    
    CUDA_HOST_DEV h_Matrix();
    CUDA_HOST_DEV h_Matrix(int height, int width, int depth);
    CUDA_HOST_DEV h_Matrix(int height, int width, int depth, int preventFree);
    CUDA_HOST_DEV h_Matrix(double* elements, int height, int width, int depth);
    CUDA_HOST_DEV h_Matrix(double* elements, int height, int width, int depth, int preventFree);
    CUDA_HOST_DEV int numel ();
    CUDA_HOST_DEV double* getColDouble(int i);
    CUDA_HOST_DEV double* getElement(int i, int j);
    CUDA_HOST_DEV void setElement(int i, int j, double value);
    CUDA_HOST_DEV void setElement(int i, double value);
    CUDA_HOST_DEV double* getElement(int i);
    CUDA_HOST_DEV h_Matrix getCol(int i);
    CUDA_HOST_DEV h_Matrix getPlane(int i);
    CUDA_HOST_DEV void addDoubleElementwise(double val);
    CUDA_HOST_DEV void multDoubleElementwise(double val);
    CUDA_HOST_DEV static void multiply(h_Matrix* a, h_Matrix* b, h_Matrix* result);
    
    CUDA_HOST_DEV ~h_Matrix();
};

#endif
