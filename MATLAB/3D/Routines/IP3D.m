function [cc]=IP3D(Re,Vx,Vy,Vz);
%InPro3D calculate the inner product bettwen Re (a 3D array) %and all the atoms of 3 separable dictiories Vx, Vy and Vz, 
%each dictionary is a 2 Array
%The output is the 3D array cc
%
[Lx,Nx]=size(Vx);
[Lz,Ny]=size(Vy);
[Lz,Nz]=size(Vz);
%
cc=zeros(Nx,Ny,Nz);
%
       for m3=1:Nz
       cc(:,:,m3)=0;
       for zk=1:Lz       
            cc(:,:,m3)=cc(:,:,m3)+(Vx'*Re(:,:,zk)*Vy).*Vz(zk,m3);
       end
       
       end
%

