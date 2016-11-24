function [h,Re,c]=ProjMP3D(h,Re,V1,V2,V3,c,toln,Max);
%
% 2DProjMP Projected Matching Pursuit 
%
% It uses the MP method for projecting the MP solution onto the spann of 
% selected %2D atoms
%
% Usage:    [h, c] = ProjMP2D(Re,h V1,V2,c,toln,Max);
%           [h, c] = ProjMP2D(Re,h V1,V2,c);
%
%
% Inputs:
%   Re     Residual of MP
%   h      Approximation by MP
%   V1     1D Dictionary selected by MP (horizontal coordinate)
%   V2     1D Dictionary selected by MP (vertical coordinate)
%   c      coefficientes in the MP approximation
%   toln   numerical tolerance for the projection  (default tre=1e-3)
%   Max    maximum number of iterations (default Max=3000)
%
% Outputs:
%   h     projected approximation  
%   c     coefficients in the decomposition
%
%   Reference: see SfProjMP2D
%
%

%  More information at http://www.nonlinear-approx.info/
%
%setting defauls
%
if (nargin<7) | (isempty(Max)==1)  Max=3000;end
if (nargin<6) | (isempty(toln)==1) toln =1e-3;end
%
name='ProjMP3D';
[L1,N1]=size(V1);
[L2,N2]=size(V2);
[L3,N3]=size(V3);
delta=1/(L1*L2*L3);
tol2=1e-11;
%Iniciate the  iterations
for it=1:Max;
%========== this takes long in matlab==============
%     for n=1:N1
%     cc(n)=0;
%     for zk=1:L3
%     cc(n)=cc(n)+(V1(:,n)'*Re(:,:,zk)*V2(:,n)).*V3(zk,n);
%     end
%     end
    [cc]=IPSP3D_mex(Re,V1,V2,V3);
%=============================================
       [c1,n1]=max(abs(cc));
     if c1 < tol2 
     fprintf('%s stopped, max(|<f,D>|)<= tol2=%g.\n',name,tol2); break;
     end
     for zk=1:L3
     h_new(:,:,zk)=V1(:,n1)*cc(n1)*V2(:,n1)'*V3(zk,n1);
     end
     c(n1)=c(n1)+cc(n1);
     h=h+h_new;
     Re=Re-h_new;%computes the new residual 
     nornu=sum(sum(sum(abs(h_new).^2)))*delta;
%     nor=sum(sum(sum(abs(Re).^2)))*delta
     if  (nornu <= toln) break;end;
end
if it==Max fprintf('%s Maximum number of iterations has been reached.\n',name);
end

%
%============================================
%      Laura Rebollo-Neira 2016
%  Mathematics Department, Aston Uni, UK
%=============================================
%
