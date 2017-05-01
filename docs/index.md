## Sparse Image Approximation in 3D

This site hosts implementations of two Matching Pursuit strategies for approximation of colour images. 
The strategies, termed 'Orthogonal Matching Pursuit' and 'Self-Projected Matching Pursuit' have been prototyped in MATLAB.
The novelty with this project is that the algorithms attempt to take a sparse representation of every dimension similtaneously through the use of three separable dictionaries, which achieve a high redundancy whilst using low storage. Past projects have tackled the problem by vectorising images into 1D arrays or processing each colour channel individually. Ofcourse, the approach taken in this project is far more computationally intensive which is why an attempt of using the GPU for accelaration has been made. 

Further C++ MEX files have been created to provide substancially increased performance, up to 90% when using the smallest realistic block size. 
An attempt of accelarating the process further by using NVidia's CUDA technology was also made resulting in a implementation of MP3D. 

The provided C++ and CUDA files were developed as part of a Final Year Project at Aston University whilst under the supervision of Dr Laura Rebollo-Neira and Dr George Vogiatzis. The MATLAB files, and 3D Sparse Image Approximation routines are part of a larger research programme led by Dr Rebollo-Neira, further details can be found at the projects site: <http://www.nonlinear-approx.info/> 

Although the software here is currently designed for Images, the same software could be used for approximation of 3D objects given a suitably sized dictionary for the third / z dimension.

Below is an example of an approximation made with SPMP3D when projection is taking place every iteration. This is equivilent to Orthogonal Matching Pursuit.
The target PSNR is 47dB with a square block size of 8 x 8. 
The achieved Sparsity Ratio was 19.5924.

### Original Image
[![Original Image](https://dannyxd11.github.io/SIA_3D/docs/Images/original.png "Original Image")](https://dannyxd11.github.io/SIA_3D/docs/Images/original.png)
heic1209a.png courtesy of the ESA - <https://www.spacetelescope.org/>

### Approximated Image
[![Approximated Image](https://dannyxd11.github.io/SIA_3D/docs/Images/approximation-47dB-19.5924.png "Approximated Image")](https://dannyxd11.github.io/SIA_3D/docs/Images/approximation-47dB-19.5924.png)

#### Other Approximations:
[PNSR : 40dB - Sparsity Ratio : 64.61166](https://dannyxd11.github.io/SIA_3D/docs/Images/approximation-40dB-64.6166.png)

[PNSR : 45dB - Sparsity Ratio : 32.5186](https://dannyxd11.github.io/SIA_3D/docs/Images/approximation-45dB-32.5186.png)

[PNSR : 55dB - Sparsity Ratio : 6.1855](https://dannyxd11.github.io/SIA_3D/docs/Images/approximation-55dB-6.1855.png)

[PNSR : 75dB - Sparsity Ratio : 2.239](https://dannyxd11.github.io/SIA_3D/docs/Images/approximation-75dB-2.239.png)

Licence:

MEX Files / CUDA Routine Copyright (C) 2017  Daniel Whitehouse, <dannyxd@me.com>, Aston University
MATLAB Routine Copyright (C) 2017, Laura Rebollo-Neira, <l.rebollo-neira@aston.ac.uk>, Aston University,  
  Full Details at <http://www.nonlinear-approx.info/>
Part of a Final Year Project for the Degree of Computer Science and Mathematics at Aston University


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.




