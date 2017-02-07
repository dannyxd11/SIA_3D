#ifndef IPSP3D
#define IPSP3D

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



class h_Matrix{
public:
    thrust::host_vector<double> elements;
    int height, width, depth;
    h_Matrix(int height, int width, int depth) : height(height), width(width), depth(depth) { elements = thrust::host_vector<double>(height*width*depth, 0); };//elements.reserve(height*width*depth);};
    h_Matrix(double* ele, int height, int width, int depth) : height(height), width(width), depth(depth) { elements = thrust::host_vector<double>(ele, ele + height*width*depth*sizeof(double));}
};


class d_Matrix{
public:
    thrust::device_vector<double> elements;
    double* elements_ptr;
    int height, width, depth;
    d_Matrix(int height, int width, int depth) : height(height), width(width), depth(depth) { elements = thrust::device_vector<double>(height*width*depth, 0); };//elements.reserve(height*width*depth);};
    d_Matrix(double* ele, int height, int width, int depth) : height(height), width(width), depth(depth) { elements = thrust::device_vector<double>(ele, ele + height*width*depth*sizeof(double)); elements_ptr = elements.data().get();}
};

/*void IPSP3d(Matrix re, Matrix v1, Matrix v2, Matrix v3, Matrix* cc){
    int n1 = v1.getWidth();
    int l3 = v3.getHeight();


    Matrix v1Trans (v1.getHeight(), v1.getWidth(), v1.getDepth());
    std::vector<double> v(v1.getElements(), v1.getElements() + (v1.getSize() * sizeof(double)));
    v1Trans.setElements(v);
    v1Trans.transpose();

    Matrix row(1, v1Trans.getWidth(), 1);
    Matrix plane(re.getHeight(), re.getWidth(), 1);
    //Matrix aMatrix(row.getHeight(), plane.getWidth(), 1);
    Matrix b2(v2.getHeight(), 1, 1);
    //Matrix calcMatrix(aMatrix.getHeight(), b2.getWidth(), 1);


    for(int i = 0; i < n1; i++){
		cc->setElement(i, 0);
        for(int j = 0; j < l3; j++){
            row.setElements(v1Trans.getRow(i));
            plane.setElements(re.getPlane(j));
            Matrix aMatrix = Matrix::blasMultiply(row, plane);
            b2.setElements(v2.getCol(i), b2.getSize());
            Matrix calcMatrix = Matrix::blasMultiply(aMatrix, b2);
            cc->setElement(i, cc->getElement(i) + calcMatrix.getElement(0) * v3.getElement(i, j));
        }
    }
}*/


__global__
void d_IPSP3d(cublasHandle_t handle, d_Matrix* re, d_Matrix* v1, d_Matrix* v2, d_Matrix* v3, d_Matrix* cc){
    int n1 = v1->width;
    int l3 = v3->height;

    cublasStatus_t status;
    double* c = new double(v2->height * re->width);
    double scalar = 1;
    status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, re->width, re->height, &scalar, v2->elements_ptr, v2->height, re->elements_ptr, re->height, 0, c,v2->height);

    if (status != CUBLAS_STATUS_SUCCESS){printf("Cublas DGEMM failure (v2xre)\n"); return;}
    if (status == CUBLAS_STATUS_SUCCESS){printf("Dan you genius\n"); return;}
//    for(int i = 0; i < n1; i ++){
//
//        for(int j = 0; j < l3; j++){
//            // V2(:,n)'*Re(:,:,zk)'  // height of V2 (1 since its a row) width of Re (Transposed so height)
//            double* c = new double(v2->height * re->width);
//            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, re->width, re->height, 1, v2->elements.data(), v2->height, re->elements.data(), re->height, 0, c,v2->height);
//
//        }
//    }

    return;
}


int main() {

    double dxElements[]  = {0.35355,0.49759,0.49039,0.47847,0.46194,0.44096,0.41573,0.38651,0.35355,0.3172,0.27779,0.2357,0.19134,0.14514,0.097545,0.049009,0,0.049009,0.097545,0.14514,0.19134,0.2357,0.27779,0.3172,0.35355,0.38651,0.41573,0.44096,0.46194,0.47847,0.49039,0.49759,1,0,0,0,0,0,0,0,
                            0.35355,0.47847,0.41573,0.3172,0.19134,0.049009,-0.097545,-0.2357,-0.35355,-0.44096,-0.49039,-0.49759,-0.46194,-0.38651,-0.27779,-0.14514,0,0.14514,0.27779,0.38651,0.46194,0.49759,0.49039,0.44096,0.35355,0.2357,0.097545,-0.049009,-0.19134,-0.3172,-0.41573,-0.47847,0,1,0,0,0,0,0,0,
                            0.35355,0.44096,0.27779,0.049009,-0.19134,-0.38651,-0.49039,-0.47847,-0.35355,-0.14514,0.097545,0.3172,0.46194,0.49759,0.41573,0.2357,0,0.2357,0.41573,0.49759,0.46194,0.3172,0.097545,-0.14514,-0.35355,-0.47847,-0.49039,-0.38651,-0.19134,0.049009,0.27779,0.44096,0,0,1,0,0,0,0,0,
                            0.35355,0.38651,0.097545,-0.2357,-0.46194,-0.47847,-0.27779,0.049009,0.35355,0.49759,0.41573,0.14514,-0.19134,-0.44096,-0.49039,-0.3172,0,0.3172,0.49039,0.44096,0.19134,-0.14514,-0.41573,-0.49759,-0.35355,-0.049009,0.27779,0.47847,0.46194,0.2357,-0.097545,-0.38651,0,0,0,1,0,0,0,0,
                            0.35355,0.3172,-0.097545,-0.44096,-0.46194,-0.14514,0.27779,0.49759,0.35355,-0.049009,-0.41573,-0.47847,-0.19134,0.2357,0.49039,0.38651,0,0.38651,0.49039,0.2357,-0.19134,-0.47847,-0.41573,-0.049009,0.35355,0.49759,0.27779,-0.14514,-0.46194,-0.44096,-0.097545,0.3172,0,0,0,0,1,0,0,0,
                            0.35355,0.2357,-0.27779,-0.49759,-0.19134,0.3172,0.49039,0.14514,-0.35355,-0.47847,-0.097545,0.38651,0.46194,0.049009,-0.41573,-0.44096,0,0.44096,0.41573,-0.049009,-0.46194,-0.38651,0.097545,0.47847,0.35355,-0.14514,-0.49039,-0.3172,0.19134,0.49759,0.27779,-0.2357,0,0,0,0,0,1,0,0,
                            0.35355,0.14514,-0.41573,-0.38651,0.19134,0.49759,0.097545,-0.44096,-0.35355,0.2357,0.49039,0.049009,-0.46194,-0.3172,0.27779,0.47847,0,0.47847,0.27779,-0.3172,-0.46194,0.049009,0.49039,0.2357,-0.35355,-0.44096,0.097545,0.49759,0.19134,-0.38651,-0.41573,0.14514,0,0,0,0,0,0,1,0,
                            0.35355,0.049009,-0.49039,-0.14514,0.46194,0.2357,-0.41573,-0.3172,0.35355,0.38651,-0.27779,-0.44096,0.19134,0.47847,-0.097545,-0.49759,0,0.49759,0.097545,-0.47847,-0.19134,0.44096,0.27779,-0.38651,-0.35355,0.3172,0.41573,-0.2357,-0.46194,0.14514,0.49039,-0.049009,0,0,0,0,0,0,0,1
    };

    double dyElements[]  = {0.35355,0.49759,0.49039,0.47847,0.46194,0.44096,0.41573,0.38651,0.35355,0.3172,0.27779,0.2357,0.19134,0.14514,0.097545,0.049009,0,0.049009,0.097545,0.14514,0.19134,0.2357,0.27779,0.3172,0.35355,0.38651,0.41573,0.44096,0.46194,0.47847,0.49039,0.49759,1,0,0,0,0,0,0,0,
                            0.35355,0.47847,0.41573,0.3172,0.19134,0.049009,-0.097545,-0.2357,-0.35355,-0.44096,-0.49039,-0.49759,-0.46194,-0.38651,-0.27779,-0.14514,0,0.14514,0.27779,0.38651,0.46194,0.49759,0.49039,0.44096,0.35355,0.2357,0.097545,-0.049009,-0.19134,-0.3172,-0.41573,-0.47847,0,1,0,0,0,0,0,0,
                            0.35355,0.44096,0.27779,0.049009,-0.19134,-0.38651,-0.49039,-0.47847,-0.35355,-0.14514,0.097545,0.3172,0.46194,0.49759,0.41573,0.2357,0,0.2357,0.41573,0.49759,0.46194,0.3172,0.097545,-0.14514,-0.35355,-0.47847,-0.49039,-0.38651,-0.19134,0.049009,0.27779,0.44096,0,0,1,0,0,0,0,0,
                            0.35355,0.38651,0.097545,-0.2357,-0.46194,-0.47847,-0.27779,0.049009,0.35355,0.49759,0.41573,0.14514,-0.19134,-0.44096,-0.49039,-0.3172,0,0.3172,0.49039,0.44096,0.19134,-0.14514,-0.41573,-0.49759,-0.35355,-0.049009,0.27779,0.47847,0.46194,0.2357,-0.097545,-0.38651,0,0,0,1,0,0,0,0,
                            0.35355,0.3172,-0.097545,-0.44096,-0.46194,-0.14514,0.27779,0.49759,0.35355,-0.049009,-0.41573,-0.47847,-0.19134,0.2357,0.49039,0.38651,0,0.38651,0.49039,0.2357,-0.19134,-0.47847,-0.41573,-0.049009,0.35355,0.49759,0.27779,-0.14514,-0.46194,-0.44096,-0.097545,0.3172,0,0,0,0,1,0,0,0,
                            0.35355,0.2357,-0.27779,-0.49759,-0.19134,0.3172,0.49039,0.14514,-0.35355,-0.47847,-0.097545,0.38651,0.46194,0.049009,-0.41573,-0.44096,0,0.44096,0.41573,-0.049009,-0.46194,-0.38651,0.097545,0.47847,0.35355,-0.14514,-0.49039,-0.3172,0.19134,0.49759,0.27779,-0.2357,0,0,0,0,0,1,0,0,
                            0.35355,0.14514,-0.41573,-0.38651,0.19134,0.49759,0.097545,-0.44096,-0.35355,0.2357,0.49039,0.049009,-0.46194,-0.3172,0.27779,0.47847,0,0.47847,0.27779,-0.3172,-0.46194,0.049009,0.49039,0.2357,-0.35355,-0.44096,0.097545,0.49759,0.19134,-0.38651,-0.41573,0.14514,0,0,0,0,0,0,1,0,
                            0.35355,0.049009,-0.49039,-0.14514,0.46194,0.2357,-0.41573,-0.3172,0.35355,0.38651,-0.27779,-0.44096,0.19134,0.47847,-0.097545,-0.49759,0,0.49759,0.097545,-0.47847,-0.19134,0.44096,0.27779,-0.38651,-0.35355,0.3172,0.41573,-0.2357,-0.46194,0.14514,0.49039,-0.049009,0,0,0,0,0,0,0,1
    };

    double dzElements[] = {0.57735,0.78868,0.70711,0.57735,0.40825,0.21132,0,0.21132,0.40825,0.57735,0.70711,0.78868,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0.57735,0.57735,4.9996e-17,-0.57735,-0.8165,-0.57735,0,0.57735,0.8165,0.57735,9.9992e-17,-0.57735,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0.57735,0.21132,-0.70711,-0.57735,0.40825,0.78868,0,0.78868,0.40825,-0.57735,-0.70711,0.21132,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    double reElements[] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6};


    d_Matrix dx(dxElements, 8, 40, 1);
    d_Matrix dy(dyElements, 8, 40, 1);
    d_Matrix dz(dzElements, 3, 40, 1);
    d_Matrix re(reElements, 8, 8, 3);
    d_Matrix cc(1, 40, 1);

    cublasStatus_t stat;
    cublasHandle_t handle;

    cublasCreate(&handle);
    d_IPSP3d<<< 1, 32>>>(handle, &re, &dx, &dy, &dz, &cc);
    cublasDestroy(handle);
    return 0;
}

#endif