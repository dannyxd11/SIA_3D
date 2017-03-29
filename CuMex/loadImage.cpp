#ifndef loadImage
#define loadImage
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "h_Matrix.h"

using namespace cv;

h_Matrix* loadImageToMatrix(char* path){
    Mat image = imread( path, 1 );
    
    if ( !image.data ) { printf("No image data \n"); return new h_Matrix(); }

//    namedWindow("Original Image", CV_WINDOW_NORMAL );
//    imshow("Original Image", image);
//    moveWindow("Original Image",0,0);
    //setWindowProperty("Original Image", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    
    h_Matrix* inputImage = new h_Matrix(image.rows, image.cols, 3);
    
    for(int i = 0; i < image.cols; i++){
        for(int j = 0; j < image.rows; j++){
            Vec3b intensity = image.at<Vec3b>(j,i);
            inputImage->elements[i * inputImage->height + j] = (double)intensity.val[2];
            inputImage->elements[i * inputImage->height + j + inputImage->height*inputImage->width] = (double)intensity.val[1];
            inputImage->elements[i * inputImage->height + j + inputImage->height*inputImage->width * 2] = (double)intensity.val[0];
        }
    }

    return inputImage;
}


#endif
