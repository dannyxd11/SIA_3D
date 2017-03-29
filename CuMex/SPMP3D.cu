#ifndef SPMP3D
#define SPMP3D

#include <algorithm> // copy
#include "h_Matrix.h"
#include "loadImage.cpp"
#include "CreateDict.cpp"

int main(int argc, char** argv )
{
    if ( argc != 2 ) { printf("usage: SPMP3D.out <Image_Path>\n"); return -1; }



    h_Matrix* inputImage = loadImageToMatrix(argv[1]);


    h_Matrix Dx = createStandardDict();
    h_Matrix Dy = createStandardDict();
    h_Matrix Dz = createDzDict();

    for(int i = 0; i < Dx.numel(); i ++){ printf("%f, ", Dx.elements[i]); }
    std::cout << "\n\n";
    for(int i = 0; i < Dz.numel(); i ++){ printf("%f, ", Dz.elements[i]); }
    //waitKey(0);
    delete inputImage;
    return 0;
}




#endif
