function [H,Di1,Di2, beta, c, Q, nore1] = OMP2D_m(f,Dx,Dy,tol, No, indx, indy);
% OMP2D Orthogonal Matching Pursuit in 2D
%
% It creates an atomic decomposition of a 2D signal using OMP criterion and assuming
% separable dictionaries. You can choose a tolerance, the number of atoms to take in 
% or an initial subspace to influence the OMP algorithm. 
% 
%  
% Usage:    [H, Dx2,Dy2,Di1,Di2, beta, c,Q ] = OMP2D( f, Dx,Dy, tol, No, indx,indy);
%           [H] = OMP2D(f,Dx,Dy); variables tol, No, can also be []
%                 
%
% Inputs:
%   f       analyzing 2D signal (Image)
%   Dx      dictionary of normalized atoms (w.r.t. Image rows)
%   Dy      dictionary of normalized atoms (w.r.t. Image columns)
%   tol     desired distance between f and its approximation H  (mse) default=6.5
%   No      (optional) maximal number of atoms to choose, if the number of chosen atoms
%           equals to No, routine will stop (default numel(f))
%   indx    (optional) indices for an initial subspace; They operate as indx(k),indy(k) 
%   indy    (optional) indices determining  the initial subspace (as above)
%
% Outputs:
%   H       Approximation of f
%   Di1     indices of selected atoms w.r.t the original Dx
%   Di2     indices of selected atoms w.r.t the original Dy
%   beta    biorthogonal atoms corresponding to the 2D selected atoms 
%   c       coefficients of the atomic decomposition
%   Q       orthogonal 2D basis for the selected subspace
%  
%   
% The implementation of OMP method is based on  Gram-Schmidt orthonormalization and 
% adaptive birothogonalization, as proposed in Ref 2 below.
%
%  References:
%
%  1-Y.C. Pati, R. Rezaiifar, and P.S. Krishnaprasad, "Orthogonal matching pursuits:
%  recursive function approximation with applications to wavelet decomposition", in Proceedings of
%  the 27th Asilomar Conference on Signals, Systems and Computers, 1993.
%   
%  2-L. Rebollo-Neira and D. Lowe, "Optimized Orthogonal Matching Pursuit Approach", IEEE
%  Signal Processing Letters, Vol(9,4), 137-140, (2002). 
%
% See also MP2D, ProjMP2D, SfProjMP2D
%
% More  information at  http://www.nonlinear-approx.info/

name='OMP2D_m';  %name of routine
% get inputs and setup parameters
%!!!! modification of OMP 1D
[Lx,Nx]=size(Dx);
[Ly,Ny]=size(Dy);
delta=1/(Lx*Ly);
N=Nx*Ny;
Dix=1:Nx;
Diy=1:Ny;
if sum(sum(f).^2)*delta<1e-9
H=zeros(size(f));
Di1=[];
Di2=[];
beta=[];
Q=[];
c=0;
return
end
%!!!!end 
zmax=2;%number of reorthogonalizations
beta=[];
Re=f;
if (nargin<7) | (isempty(indx)==1)  indx=[];end
if (nargin<6) | (isempty(indy)==1)  indy=[];end
if (nargin<5) | (isempty(No)==1)    No=Lx*Ly;end
if (nargin<4) | (isempty(tol)==1)   tol=6.5;end; 

numind=numel(indx);

%atoms having smaller norm than tol1 are supposed be zero ones
tol1=1e-7; %1e-5
%threshold for coefficients
tol2=1e-10;   %0.0001  %1e-5
%===============================
% Main algorithm: at kth iteration
%===============================
H=min(No,N); %maximal number of function in sub-dictionary
for k=1:H    
  %finding of maximal coefficient
  cc=zeros(Nx,Ny);
  %modification of 1D
  if k<=numind 
    [testx,qx]=ismember(indx(k),Dix);
    [testy,qy]=ismember(indy(k),Diy);
    if testx ~=1  error('Demanded index (x) %d is out of dictionary',indx(k));end
    if testy ~=1  error('Demanded index (y) %d is out of dictionary',indy(k));end
    q=[indx(k),indy(k)];
  else
  %%%modification of 1D OMP 
   cc=abs(Dx'*Re*Dy);
  %cc=abs(InPrRDCDB(Re,2,2));
    [c1,c2]=max(cc);
    [max_c,b2]=max(c1);
    q=[c2(b2) b2];
  %%%end modification
  %stopping criterion (coefficient)
    if max_c<tol2 
      k=k-1;
      fprintf('%s stopped, max(|<f,q>|/||q||)<= tol2=%g.\n',name,tol2);
      break;
    end
  end  
%!!! modification of 1D OMP
    Di1(k)=q(1);
    Di2(k)=q(2);
%!!!end  
  if k>1
%!!!modification of 1D OMP
  new_atom=kron(Dy(:,q(2)),Dx(:,q(1)));
   %Q(:,k) is the orthogonalization of D(:,k)  w.r.t Q(:,1),..., Q(:,k-1) 
    for p=1:k-1
     Q(:,k)=new_atom-(Q(:,p)'*new_atom)*Q(:,p); %orthogonalization
    end 
%!!!end  
   %re-orthogonalization of Q(:,k)  w.r.t Q(:,1),..., Q(:,k-1) 
   for zi=1:zmax
    for p=1:k-1
     Q(:,k)=Q(:,k)-(Q(:,p)'*Q(:,k))*Q(:,p); %re-orthogonalization
    end	   
   end
  end
%!!!modification of 1D OMP
   if k==1 Q(:,k)=kron(Dy(:,q(2)),Dx(:,q(1))); end
%!!!end  
  nork=norm(Q(:,k)); 
  Q(:,k)=Q(:,k)/nork; %normalization
  % compute biorthogonal functions beta from 1 to k-1
%!!!modification of OMP1D
  if k>1
    beta=beta-Q(:,k)*(new_atom'*beta)/nork; 
  end	
%!!!end
  beta(:,k)=Q(:,k)/nork; % kth biorthogonal function
%
%!!!modification  of 1D OMP compute the residuum and reshape;
  h=f(:)'*Q(:,k)*Q(:,k)';
  Re=Re(:)-h';
  nore1(k)=(norm(Re))^2*(delta);
  Re=reshape(Re,Lx,Ly);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%
  %stopping criterion (distance)
  if (tol~= 0) & (nore1(k) < tol) break;end;
end
%!!!modification  of 1D OMP to obtain the approximation and selected atoms
c=f(:)'*beta;
%Dx2=Dx(:,Di1);
%Dy2=Dy(:,Di2);
H=0;
id=numel(Di1);
%for n=1:id;
%H=H+Dx2(:,n)*c(n)*Dy2(:,n)';
%end
for n=1:id;
H=H+Dx(:,Di1(n))*c(n)*Dy(:,Di2(n))';
end
%
%!!!end

%Laura REBOLLO-NEIRA 2010, based on the 1D implementation copyright below
%
%Copyright (C) 2006 Miroslav ANDRLE and Laura REBOLLO-NEIRA
%
%This program is free software; you can redistribute it and/or modify it under the terms 
%of the GNU General Public License as published by the Free Software Foundation; either 
%version 2 of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
%without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%See the GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License along with this program;
%if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
%Boston, MA  02110-1301, USA.
