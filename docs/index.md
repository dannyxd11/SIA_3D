## Sparse Image Approximation in 3D

This site hosts implementations of two Matching Pursuit strategies for approximation of colour images. 
The strategies, termed 'Orthogonal Matching Pursuit' and 'Self-Projected Matching Pursuit' have been prototyped in MATLAB.
The novelty with this project is that the algorithms attempt to take a sparse representation of every dimension similtaneously through the use of three separable dictionaries, which achieve a high redundancy whilst using low storage. Past projects have tackled the problem by vectorising iamges into 1D arrays or processing each colour channel individually. Ofcourse, the approach taken in this project is far more computationally intensive which is why an attempt of using the GPU for accelaration has been made. 

Further C++ MEX files have been created to provide substancially increased performance, up to 90% when using the smallest realistic block size. 
An attempt of accelarating the process further by using NVidia's CUDA technology was also made. 

The provided C++ and CUDA files were developed as part of a Final Year Project at Aston Nniversity whilst under the supervision of Dr Laura Rebollo-Neira and Dr George Vogiatzis. The MATLAB files, and the Self-Projected Matching Pursuit routine are part of a larger research programme led by Dr Rebollo-Neira, further details can be found at the projects site: http://www.nonlinear-approx.info/.

Although the software here is currently designed for Images, the same software could be used for approximation of 3D images given a suitable sized dictionary for the third / z dimension.






