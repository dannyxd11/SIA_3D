addpath('\Mex');

mex Mex\hnew3D_mex.cpp -lmwblas;
mex Mex\IP3D_mex.cpp -lmwblas;
mex Mex\IPSP3d_mex.cpp -lmwblas;
mex Mex\ProjMP3D_mex.cpp -lmwblas;

printf('Mex Compiled');