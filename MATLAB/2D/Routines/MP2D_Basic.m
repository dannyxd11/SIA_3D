function[h,Di1,Di2,c]=MP2D_Basic(f,Vx,Vy,tol,No,Maxit);
%
% Matching Pursuit 2D (Basic version)
%
% Usage: [h,Di1,Di2,c]=MP2D_Basic(f,Vx,Vy,tol,No,Max);
%
% Inputs:
%   f     2D signal (Image)
%   Vx    dictionary of normalized 1D atoms (image's rows)
%   Vy    dictionary of normalized 1D atoms (image's columns)
%   tol   tolerance for the approximation error (ems = (error norm)^2/numel(f))  
%   No    maximum number of atoms in the approximation
%   Max   maximum number of interations.
%
%  Outputs:
%
%   h     approximation of f ( Approximated Image)
%   Di1   indexes of selected (distinct) atoms (w.r.t. the original Vx)
%   Di2   indexes of selected (distinct) atoms (w.r.t. the original Vy)
%   c     coefficeints of the atomic decomposition: sum_n(c(n)*Vx(:,Di1(n))*Vy(:,Di2(n))
%
[Lx,Nx]=size(Vx);%Lx is the number if rows in the image, Nx the number of vectors in Vx
[Ly,Ny]=size(Vy);%Ly is the number if columns in the image, Ny the number of vectors in Vy
delta=1/(Lx*Ly);
%
name='MP2D_Basic';
cp=zeros(Nx,Ny);
cc=zeros(Nx,Ny);
Di1=[];
Di2=[];
numat=0;
h=zeros(Lx,Ly);
%Check of the block has intensity zero and returns the zero block
if sum(sum(f).^2)*delta<1e-9
c=[];
return
end
%============================================================
%
Re=f;
tol2=1e-9;%to stop when there is no solution for some problem
%
%========= Algorithm starts===================================
%
for it=1:Maxit;
       cc=(Vx'*Re*Vy); %inner product of Residue and 1D dictionaries (from left and right)
       [c1,c2]=max(abs(cc)); 
       [max_c,b2]=max(c1); %in combination with the previous line selects the indices 
       q=[c2(b2) b2]; %indices of the 1D selected atoms in each direction
%
         if max_c < tol2 fprintf('%s stopped, max(|<f,D>|)<= tol2=%g.\n',name,tol2); break; end 
%
%=This is to collect selected atoms with the same indices, it coud be done at the end outside
      indq1q2=0;
      [testq1,indq1]=ismember(q(1),Di1);  
      if (testq1 ==1 & q(2)==Di2(indq1)) indq1q2=1; end
      if indq1q2==0
         Di1=[Di1 q(1)]; %Stores indices of distinct atoms (rows)
         Di2=[Di2 q(2)]; %Stores indices of distinct atoms (collums) 
         numat=numat+1; %counts the number of distinct atoms
      end
%=========================================================================
      cscra=cc(q(1),q(2)); 
      h_new=Vx(:,q(1))*cscra*Vy(:,(q(2)))'; %to be added to the previous approximation
      cp(q(1),q(2))=cp(q(1),q(2))+cscra;  %add coefficients of identical atoms
      h=h+h_new; %Approximated Image 
      Re=Re-h_new;%New residual
      nor_new=sqrt(sum(sum(abs(Re).^2))); 
      if (numat>=No | (nor_new < tol)) break;end;
end 
     l=numel(Di1); %number of different atoms
%
%======stores coefficients as a vector is not actualy needed=========
    for n=1:l;
     c(n)=cp(Di1(n),Di2(n));
    end
%================================================================
if it==Maxit fprintf('%s Maximum number of iterations has been reached\n',name); end
%
%==============================================
%      Laura Rebollo-Neira 2010
%  Mathematics Department, Aston Uni, UK
%==============================================
%
