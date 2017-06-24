## Sparse Image Approximation in 3D

This site hosts two different implementations of  the Orthogonal Matching Pursuit (OMP) strategy, for  the approximation of  3D images. In addition to a standard implementation, here dedicated to operate with separable dictionaries in 3D (OMP3D) the low memory  version of the same approach, termed  [Self-Projected Matching Pursuit](http://www.nonlinear-approx.info/examples/node1.html)  is also extended to 3D (SPMP3D).  Both implementations have been shown appropriate for the simultaneous approximation of colour images. By approximating a classic [data set](http://r0k.us/graphics/kodak/),  this project demonstrated that the sparsity achieved by the simultaneous approximation of three colour channels is significantly higher than that achieved  by approximating each channel independently.  The gain in sparsity comes at expenses of computational complexity. Hence, since the approximation is carried out on a partition of the image, the possibility of massive palletization with [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) was investigated.

The project was developed  by [Daniel Whitehouse](https://dan-whitehouse.me), as dissertation for his the [BSc (Hons) Computer Science and Mathematics](http://www.aston.ac.uk/study/undergraduate/courses/eas/bsc-computer-science-and-mathematics/) degree, at  [Aston University](http://www.aston.ac.uk). It was supervised by [Laura Rebollo-Neira](http://www.aston.ac.uk/eas/staff/a-z/dr-laura-rebollo-neira/)  ([Mathematics Department](http://www.aston.ac.uk/eas/about-eas/academic-groups/mathematics/)) and [George Vogiatzis](http://www.aston.ac.uk/eas/staff/a-z/gv/) ([Computer Science Department](http://www.aston.ac.uk/eas/about-eas/academic-groups/computer-science/)).  The main deliveries are: 

• Production of C++ MATLAB executables for the SPMP3D and OMP3D approaches,  which significantly increased performance compared to the stand alone MATLAB implementation.

• An original investigation into the benefit of approximating 3 channel colour images simultaneously, instead of approximating  each colour  independently.

• A study into the feasibility of using the GPU and fast-access memory for the SPMP3D routine.

• Development of a CUDA enabled command line application for calculating the sparsity ratio of a given image using the plain Matching Pursuit  algorithm dedicated to 3D (MP3D). In addition, an extension to fully support the SPMP3D algorithm has been outlined.

The full report can be [downloaded here](https://github.com/dannyxd11/SIA_3D/raw/gh-pages/docs/Sparse%20Image%20Approximation%20in%203-Dimensions%20with%20GPU%20Utilisation%20-%20Daniel%20Whitehouse%20.pdf). 

Below is an example of an approximation made with SPMP3D when  the projection is taking place at every iteration. This is equivalent to the OMP3D  method.  The target PSNR is 47dB with a square block size of 8 x 8. The achieved Sparsity Ratio (SR) was 19.6 when processing the 3 colours simultaneously with a dictionary of redundancy 125.

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

MEX Files / CUDA Routine Copyright (C) 2017  [Daniel Whitehouse](https://dan-whitehouse.me), <dannyxd@me.com>, Aston University
MATLAB Routine Copyright (C) 2017, [Laura Rebollo-Neira](http://www.aston.ac.uk/eas/staff/a-z/dr-laura-rebollo-neira/), [Mathematics Department](http://www.aston.ac.uk/eas/about-eas/academic-groups/mathematics/), Aston University


This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
See <http://www.gnu.org/licenses/>.

