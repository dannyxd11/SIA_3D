function [cc]=IPSP3D(Re,V1,V2,V3);
%InPro3D calculate the inner product bettwen Re (a 3D array) %and all the atoms of 3 separable dictiories Vx, Vy and Vz, 
%each dictionary is a 2 Array
%The output is the 3D array cc
%
[L1,N1]=size(V1);
[L3,N3]=size(V3);
%
      for n=1:N1
      cc(n)=0;
      for zk=1:L3
      cc(n)=cc(n)+(V1(:,n)'*Re(:,:,zk)*V2(:,n)).*V3(zk,n);
      end
      end
%
%

