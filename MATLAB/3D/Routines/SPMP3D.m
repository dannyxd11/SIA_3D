function[h,Set_ind,c]=SPMP3D(f,Vx,Vy,Vz,tol,No,toln,lstep,Max,Maxp,indx,indy,indz);

% SPMP3D Self Projected Matching Pursuit 3D
%
% It projects MP3D approximation at every "lstep" iterations using 
% Projected  Matching  Pursuit 2D (ProjMP3D)
% For lstep=1 it converges to Orthogonal Matching Pursuit 3D (OMP3D)
% For lstep=0 it projects MP3D only once to improve upon the final MP3D approximation
% For lstep=-1 it gives plain MP3D with no projection at all 
% Other values of lstep projects every lstep steps 
%
% Inputs:
%   f     3D array
%   Vx    dictionary of normalized 1D atoms (image's raws)
%   Vy    dictionary of normalized 1D atoms (image's columns)
%   Vz    dictionary of normalized 1D atoms (dim 3)
%   tol   tolerance for the approximation (ems = (error norm)^2/numel(f))  
%   No    maximum number of atoms in the approximation
%   toln  numerical tolerance for the projection  (default tre=1e-3)
%   lstep length of the steps for projectting (default lstep=0)
%   Max   maximum number of iterations  (default Max=7000)
%   Maxm  maximum number of iterations for projection (default Max=7000)
%   indx  (optional) indices for an initial subspace; They operate as indx(k),indy(k), indz(k)

%
%  Outputs:
%
%   h               approximation of f 
%   Set_ind(:,1)    indexes of selected (distinct) atoms (w.r.t. the original Vx)
%   Set_ind(:,2)    indexes of selected (distinct) atoms (w.r.t. the original Vy)
%   Set_ind(:,3)    indexes of selected (distinct) atoms (w.r.t. the original Vz)
%   c     coefficients of the atomic decomposition
%
%
%setting defauls
%
[Lx,Nx]=size(Vx);%Lx is the number if rows in the image, Nx the number of vectors in Vx
[Ly,Ny]=size(Vy);%Ly is the number if columns in the image, Ny the number of vectors in Vy
[Lz,Nz]=size(Vz);%Lz is the number if columns in the image, Nz the number of vectors in Vz
delta=1/(Lx*Ly*Lz);
Nxyz=Lx*Ly*Lz;
%
if (nargin<12) | (isempty(indy)==1)  indz=[];end
if (nargin<11) | (isempty(indy)==1)  indy=[];end
if (nargin<10) | (isempty(indx)==1)  indx=[];end
if (nargin<9) | (isempty(Maxp)==1)  Maxp=7000;end
if (nargin<8)  | (isempty(Max)==1)   Max=7000;end
if (nargin<7)  | (isempty(lstep)==1) lstep=0;end
if (nargin<6)  | (isempty(toln)==1)  tolnu=1e-3;end
if (nargin<5)  | (isempty(No)==1)    No=Nxyz;end
if (nargin<4)  | (isempty(tol)==1)   tol=5e-4*sum(sum(sum(abs(f).^2)))*delta;end;
%
name='SPMP3D';
cp=zeros(Nx,Ny,Nz);
cc=zeros(Nx,Ny,Nz);
%MaxInt=max(max(max(f)));
Di1=[];
Di2=[];
Di3=[];
numat=0;
Set_ind=[];
Dix=1:Nx;
Diy=1:Ny;
Diz=1:Nz;
numind=numel(indx);
h=zeros(Lx,Ly,Lz);
%Check of the block has intensity zero and returns the zero block
if sum(sum(sum(f).^2))*delta<1e-9;
c=[];
return
end
%
Re=f;
tol2=1e-9; %to stop when there is no solution for some problem
%
imp=0;
if (lstep == -1) imp=1; end
if (imp==1) lstep=0;end
if (lstep == 0) Maxit2=1; lstep=Max;
   else Maxit2=Max/lstep; end
for it=1:Maxit2;
  for s=1:lstep;
      if (numat+1)<=numind
          [testx,qx]=ismember(indx(numat+1),Dix);
	  [testy,qy]=ismember(indy(numat+1),Diy);
          [testz,qz]=ismember(indz(numat+1),Diz);
	           if testx ~=1  error('Demanded index (x) %d is out of dictionary',indx(numat+1));end
		   if testy ~=1  error('Demanded index (y) %d is out of dictionary',indy(numat+1));end
                   if testz ~=1   error('Demanded index (z) %d is out of dictionary',indz(numat+1));end
        q(1)=indx(numat+1);
        q(2)=indy(numat+1);
        q(3)=indz(numat+1);
        cc(q(1),q(2),q(3))=0;
        for zk=1:Lz;
	cc(q(1),q(2),q(3))= cc(q(1),q(2),q(3))+Vx(:,q(1))'*Re(:,:,zk)*Vy(:,q(2))*Vz(zk,q(3));
        end
      else 
%===================This takes long in Matlab========
%       for m3=1:Nz
%       cc(:,:,m3)=0; 
%       for zk=1:Lz
%       cc(:,:,m3)= cc(:,:,m3)+(Vx'*Re(:,:,zk)*Vy).*Vz(zk,m3);
%       end
%       end
% %inner product of Residue and all the dictionary atoms
       [cc]=IP3D_mex(Re,Vx,Vy,Vz);       
%=====================================================
       [max_c,maxind]=max(abs(cc(:))); %chose max index of long vector       
       [q(1),q(2),q(3)]=ind2sub(size(cc),maxind);%reshape to long vector to get the 3D index    
%
         if max_c < tol2 %fprintf('%s stopped, max(|<f,D>|)<= tol2=%g.\n',name,tol2);              		
		 return; end 
    end 
%This is to collect selected atoms with the same index
      vq=[q(1),q(2),q(3)];
      if(isempty(Set_ind)==1)
        Set_ind=vq;
        numat=1;
      else
      [testq1,indq1]=ismember(vq,Set_ind,'rows');
         if testq1==0
        Set_ind =[Set_ind;vq]; %Stores indices of distinct atoms (rows)
         numat=numat+1; %counts the number of distinct atoms
         end
      end
       
%=========================================================================
      cscra=cc(q(1),q(2),q(3)); 
      
      h_new = hnew3D_mex(cscra, Vx(:,q(1)), Vy(:,q(2)), Vz(:,q(3)));
      %for zk=1:Lz
       %h_new(:,:,zk)=Vx(:,q(1))*cscra*Vy(:,(q(2)))'.*Vz(zk,q(3));
      %end %to be added to the previous approximation
%=========================================================
      cp(q(1),q(2),q(3))=cp(q(1),q(2),q(3))+cscra;  %add coefficients of identical atoms
      h=h+h_new; %Approximated Image
      Re=Re-h_new;%
      nor_new=sum(sum(sum(abs(Re).^2)))*delta;       
      if (numat>=No | (nor_new < tol)) 
	      break;end;
  end 
     l=size(Set_ind,1); %number of different atoms
%======stores coefficients as a vector=========
    for n=1:l;
     c(n)=cp(Set_ind(n,1),Set_ind(n,2),Set_ind(n,3));
    end
%================================================================
if imp ~= 1  %if imp=-1 skips projection 
Di1=Set_ind(:,1);
Di2=Set_ind(:,2);
Di3=Set_ind(:,3);
[h,Re,c]=ProjMP3D_mex(h,Re,Vx(:,Di1),Vy(:,Di2),Vz(:,Di3),c,toln,Maxp);
l=numel(c);
%======================================================
 for n=1:l;
  cp(Set_ind(n,1),Set_ind(n,2),Set_ind(n,3))=c(n);
 end
%======================================================
end 
 nore=sum(sum(sum(abs(Re).^2)))*delta;
  if (numat>=No | (nore < tol)) break;end;
end
if (lstep ~=Max) & (it==Maxit2) fprintf('%s Maximum number of iterations has been reached\n',name);
end
%
%============================================
%      Laura Rebollo-Neira 2016
%  Mathematics Department, Aston Uni, UK
%=============================================
%
