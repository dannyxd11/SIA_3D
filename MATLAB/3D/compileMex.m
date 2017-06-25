workingDir = pwd

addpath(['..' filesep '..' filesep 'Mex'])
cd(['..' filesep '..' filesep 'Mex'])

mex hnew3D_mex.cpp -lmwblas;
mex IP3D_mex.cpp -lmwblas;
mex IPSP3D_mex.cpp -lmwblas;
mex ProjMP3D_mex.cpp -lmwblas;
mex SPMP3D_mex.cpp -lmwblas;
mex OMP3D_mex.cpp -lmwblas;
mex kronecker.cpp -lmwblas;
mex o_reorthogonalize.cpp -lmwblas;
mex reorthogonalize.cpp -lmwblas;
mex orthogonalize.cpp -lmwblas;
mex biorthogonalize.cpp -lmwblas;

cd(workingDir)
disp 'Mex Files Compiled';