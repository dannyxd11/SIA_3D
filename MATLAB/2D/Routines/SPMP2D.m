function [h,Di1,Di2,c] = SPMP2D(f,Vx,Vy,tol,No,toln,lstep,Max,Maxp,indx,indy)
%
% SPMP2D Self Projected Matching Pursuit 2D
%
% It projects MP2D approximation at every "lstep" iterations using 
% Projected  Matching  Pursuit 2D (ProjMP2D)
% For lstep=1 it converges to Orthogonal Matching Pursuit (OMP) 
% For lstep=0 it projects MP2D only once to improve upon the final MP2D approximation
% For lstep=-1 it gives plain MP with no projection at all 
% 
% Usage: [h,Di1,Di2,c] = SPMP2D(f,Vx,Vy,tol,No,toln,lstep,Max,Maxp,indx,indy);
%        [h, Di1,Di2,c] = SPMP2D(f,Vx,Vy);
%
% Inputs:
%   f     2D signal (Image)
%   Vx    dictionary of normalized 1D atoms (image's raws)
%   Vy    dictionary of normalized 1D atoms (image's columns)
%   tol   tolerance for the approximation (ems = (error norm)^2/numel(f))  
%   No    maximum number of atoms in the approximation
%   toln  numerical tolerance for the projection  (default tre=1e-3)
%   lstep length of the steps for projectting (default lstep=0)
%   Max   maximum number of iterations  (default Max=7000)
%   Maxm  maximum number of iterations for projection (default Max=7000)
%   indx  (optional) indices for an initial subspace; They operate as indx(k),indy(k)

%
%  Outputs:
%
%   h     approximation of f (Image)
%   Di1   indexes of selected (distinct) atoms (w.r.t. the original Vx)
%   Di2   indexes of selected (distinct) atoms (w.r.t. the original Vy)
%   c     coefficeints of the atomic decomposition: sum_n(c(n)*Vx(:,Di1(n))*Vy(:,Di2(n))
%
%   Reference:
%   Self Projected Matching Pursuit Method in: 
%   to be completed
%
%
%  See also   SPMP OMP2D OMP and all Pursuit routines available at
%  http://www.nonlinear-approx.info/
%

% 
%setting defauls
%
[Lx,Nx]=size(Vx);
[Ly,Ny]=size(Vy);
%delta=1/(Lx*Ly);
delta = 1;
Nxy=Lx*Ly;
%
if (nargin<11) | (isempty(indy)==1)  
    indy=[];
end
if (nargin<10) | (isempty(indx)==1)  indx=[];end
if (nargin<9) | (isempty(Maxp)==1)  Maxp=7000;end
if (nargin<8)  | (isempty(Max)==1)   Max=7000;end
if (nargin<7)  | (isempty(lstep)==1) lstep=0;end
if (nargin<6)  | (isempty(toln)==1)  tolnu=1e-3;end
if (nargin<5)  | (isempty(No)==1)    No=Nxy;end
if (nargin<4)  | (isempty(tol)==1)   tol=5e-4*sum(sum(abs(f).^2))*delta;end;
%
name='SPMP2D';
cp=zeros(Nx,Ny);
cc=zeros(Nx,Ny);
MaxInt=max(max(f));
Di1=[];
Di2=[];
Dix=1:Nx;
Diy=1:Ny;
numat=0;
numind=numel(indx);
h=0;
Re=f;
tol2=1e-9;
%tol2 = 1;
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
	           if testx ~=1  error('Demanded index (x) %d is out of dictionary',indx(numat+1));end
		   if testy ~=1  error('Demanded index (y) %d is out of dictionary',indy(numat+1));end
        q(1)=indx(numat+1);
        q(2)=indy(numat+1);
	cc(q(1),q(2))=Vx(:,q(1))'*Re*Vy(:,q(2));
      else 
       cc=(Vx'*Re*Vy);
       [c1,c2]=max(abs(cc)); 
       [max_c,b2]=max(c1); %selects the atom
       q=[c2(b2) b2]; %indices of the 1D selected atoms in each direction
         if max_c < tol2 fprintf('%s stopped, max(|<f,D>|)<= tol2=%g.\n',name,tol2); 
             break; end 
      end 
      indq1q2=0;
      [testq1,indq1]=ismember(q(1),Di1);  
      if (testq1 ==1 & q(2)==Di2(indq1)) indq1q2=1; end
      if indq1q2==0
%%%%%%%%%%%%%%%%%new!!!!!%%%%%%%%%%%%%%%%%%%%%%    
      Di1=[Di1 q(1)];
      Di2=[Di2 q(2)];
%%%%%%%%%%%%%%%%%new!!!!!%%%%%%%%%%%%%%%%%%%%%%    
      numat=numat+1; %counts the number of distinct atoms
      end
      cscra=cc(q(1),q(2)); 
      h_new=Vx(:,q(1))*cscra*Vy(:,(q(2)))';
      cp(q(1),q(2))=cp(q(1),q(2))+cscra;
      h=h+h_new; %needs to be recalcualted below
      Re=Re-h_new;%computes the new residual
      nor_new=sum(Re(:).^2); 
      if (numat>=No | (nor_new < tol)) break;end;
  end 
%======option to disregard small (or zero) coefficients=====================
%    tre=max(max(abs(cp)))*trer;
%    clear Di1 Di2
%    [Di1, Di2]=find(abs(cp)>tre); %disregard the atoms with |coefficient|<tre  
    l=numel(Di1);
%     clear c D1 D2
    for n=1:l;
     c(n)=cp(Di1(n),Di2(n));
    end
%    h=zeros(Lx,Ly);
%    for n=1:l
%       h=h+c(n)*Vx(:,Di1(n))*Vy(:,Di2(n))';
%    end
%    Re=f-h;
%=================================================================  
if imp ~= 1  %if imp=1 pure MP
%======================main modifiction to MP======================
% call ProjMP2D to project Re via MP using dictionaries D1 and D2
%==================================================================c1
c1 = c;

[h,Re,c]=ProjMP2D(h,Re,Vx(:,Di1),Vy(:,Di2),c,toln,Maxp);
% [h1,Re1,c1]=ProjMP2D_Mex(h,Re,Vx(:,Di1),Vy(:,Di2),c,toln,Maxp);
% [h2,Re2,c2]=ProjMP2D(h,Re,Vx(:,Di1),Vy(:,Di2),c,toln,Maxp);

l=numel(c);
 for n=1:l;
  cp(Di1(n),Di2(n))=c(n);
 end
 %h2=f-Re;
%Re=f-h;
 %norm(h-h2) 
end 
 %nore=sum(sum(abs(Re).^2))*delta;
 nore = sum(Re(:).^2);
  if (numat>=No | (nore < tol)) break;end;
end
if (lstep ~=Max) & (it==Maxit2) fprintf('%s Maximum number of iterations has been reached\n',name);
end
%
%============================================
%      Laura Rebollo-Neira 2010
%  Mathematics Department, Aston Uni, UK
%=============================================
%
