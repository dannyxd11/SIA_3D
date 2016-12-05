function [h_new]=hnew3D(ccn1,V1n1,V2n1,V3n1);
%
[L3]=size(V3n1);
%
     for zk=1:L3
     h_new(:,:,zk)=V1n1*ccn1*V2n1'*V3n1(zk);
     end
%
%

