%test rgb images to use the same indexes apprximated by OMPMl (1D)

%load the image 
ImagePath='Images/027_opo0613a_256.jpg' 
% ImagePath='Images/027_opo0613a.jpg' 
%
%%%%%%%%%%%%%%%%prnr for tol%%%%
pss=40.5;
%%%%%%%%%%%%%%%%%%
SRT=15;
clear AA fs mIAp
bw=8
blockWidth=bw;
Lz=3;

%lstep=1

A=double(imread(ImagePath));
mI1=A(:,:,1);
mI2=A(:,:,2);
mI3=A(:,:,3);
MaxInt=255;


%resize(mI)

s=floor(size(mI1)/bw)
%
mI1=mI1(1:s(1)*bw,1:s(2)*bw);
mI2=mI2(1:s(1)*bw,1:s(2)*bw);
mI3=mI3(1:s(1)*bw,1:s(2)*bw);
%
AA(:,:,3)=mI3;
AA(:,:,2)=mI2;
AA(:,:,1)=mI1;
%
dz=zeros(bw,1);
%
[cmI1, nYBlocks, nXBlocks] = BlockImage(mI1, blockWidth);
[cmI2, nYBlocks, nXBlocks] = BlockImage(mI2, blockWidth);
[cmI3, nYBlocks, nXBlocks] = BlockImage(mI3, blockWidth);
%
[Lx,Ly]=size(cmI1{1,1});
delta=1/(Lx*Ly*Lz);
a=2;
Ex(1:Lx,1:Lx)=0;
Ey(1:Ly,1:Ly)=0;
%
for i=1:Lx; Ex(i,i)=1;end
    tic;
for i=1:Ly; Ey(i,i)=1;end
%
a2=a/2
Dcz=DCos(Lz,a*Lz,a);
Dsz=DSin(Lz,a*Lz,a);
Dcz=NormDict(Dcz);
Dsz=NormDict(Dsz);
%
DDx=DCos(Lx,a*Lx,a);
DSx=DSin(Lx,a*Lx,a);
DDx=NormDict(DDx);
DSx=NormDict(DSx);
%
%======= simple dictionary========================
Dx=[DDx DSx eye(bw,bw)] ;
Dy=Dx;
Dz=[Dcz Dsz eye(3,3)];
%=================================================
%
cmIa1= cell(nYBlocks,nXBlocks);
cmIa2= cell(nYBlocks,nXBlocks);
cmIa3= cell(nYBlocks,nXBlocks);
%
No=numel(A)/SRT;
%
tol=MaxInt^2/(10^(pss/10))
tol2=sqrt(tol*(Lx*Ly*Lz));
%
Max=50000;
Maxp=50000;
toln=1e-8;
toln2=sqrt(toln*Lx*Ly*Lz);

indx=[];
indy=[];
indz=[];

clear cDi1 cDi2 cDi3

nCoe=0;
k=0;
darkb=0;
Ml=3;
pp(1:Ml)=1/Ml;
c1=fix(clock)
tic;
timeTakenPerBlock = zeros(size(nYBlocks));
for i = 1:nYBlocks
    tic;
for j = 1:nXBlocks
k=k+1;
B1=cmI1{i,j};
B2=cmI2{i,j};
B3=cmI3{i,j};
fs(:,:,1)=B1;
fs(:,:,2)=B2;
fs(:,:,3)=B3;
 if (norm(B1,'fro')+ norm(B1,'fro')+norm(B1,'fro'))>1e-9
[h,cDi1,cDi2,cDi3,beta,c,Q,nore1]=OMP3D_mex(fs,Dx,Dy,Dz,tol,No,indx,indy,indz);
  cc{k}=c;
  cmIa1{i,j}=h(:,:,1);
  cmIa2{i,j}=h(:,:,2);
  cmIa3{i,j}=h(:,:,3);
  di=numel(c);
  nCoe = nCoe + di;
  else
  cmIa1{i,j}=dz*dz';
  cmIa2{i,j}=dz*dz';
  cmIa3{i,j}=dz*dz';
  nCoe = nCoe + 1;
  darkb=darkb+1;
  end
end
timeTakenPerBlock(i) = toc;
fprintf('Block Number: %d/%d - %f\n', i,nYBlocks,timeTakenPerBlock(i));
 
end 
c2=fix(clock)
mImage1=cell2mat(cmIa1);
mImage2=cell2mat(cmIa2);
mImage3=cell2mat(cmIa3);
sxyz=numel(AA);
mse1=sum(sum((mI1-mImage1).^2));
mse2=sum(sum((mI2-mImage2).^2));
mse3=sum(sum((mI3-mImage3).^2));
MSE=(mse1+mse2+mse3)/(sxyz);
PSNR=10*log10(MaxInt^2/MSE)
SR=numel(AA)/nCoe
mIAp(:,:,1)=mImage1;
mIAp(:,:,2)=mImage2;
mIAp(:,:,3)=mImage3;
fprintf('number of dark blocks.\n')
darkb
figure, plot(timeTakenPerBlock);
timeTaken = sum(timeTakenPerBlock)
toc
return
