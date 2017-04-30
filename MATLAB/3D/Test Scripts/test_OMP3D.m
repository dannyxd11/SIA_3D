function [PSNR, SR, SSIM] = test_OMP3D(ImagePath, bw, usemex);

if nargin<3 ; usemex = 1; end;
if nargin<2 ; blockWidth = 8; end;
if nargin<1 ; ImagePath = '../../../Images/027_opo0613a_256.jpg'; end;

%% PSNR for breaking tolerance

pss=40.5;

%% Set Target SR

SRT=15;

%% No. Dimensions

Lz=3;

%% Add path to required functions

addpath('../Routines')
addpath('../Functions')
addpath('../../../Mex')

%% Load image 

A=double(imread(ImagePath));
mI1=A(:,:,1);
mI2=A(:,:,2);
mI3=A(:,:,3);

%% Max Pixel Intesnsity

MaxInt=255;

%% Block Image

s=floor(size(mI1)/blockWidth);
mI1=mI1(1:s(1)*blockWidth,1:s(2)*blockWidth);
mI2=mI2(1:s(1)*blockWidth,1:s(2)*blockWidth);
mI3=mI3(1:s(1)*blockWidth,1:s(2)*blockWidth);

AA(:,:,3)=mI3;
AA(:,:,2)=mI2;
AA(:,:,1)=mI1;

%% Delta / Number of Images.
Lx = blockWidth;
Ly = Lx;
delta=1/(Lx*Ly*Lz);

%% Generation of Standard Dictionaries

dz=zeros(blockWidth,1);
[cmI1, nYBlocks, nXBlocks] = BlockImage(mI1, blockWidth);
[cmI2, nYBlocks, nXBlocks] = BlockImage(mI2, blockWidth);
[cmI3, nYBlocks, nXBlocks] = BlockImage(mI3, blockWidth);

% Set Redundancy
a=2; 
a2=a/2;

% Create Sine and Cosine Dictionary for X and Y components
DDx=DCos(Lx,a*Lx,a);
DSx=DSin(Lx,a*Lx,a);
DDx=NormDict(DDx);
DSx=NormDict(DSx);

% Create Sine and Cosine Dictionary for Z component
Dcz=DCos(Lz,a*Lz,a);
Dsz=DSin(Lz,a*Lz,a);
Dcz=NormDict(Dcz);
Dsz=NormDict(Dsz);

% Building Complete Dictionary
Dx=[DDx DSx eye(blockWidth,blockWidth)] ;
Dy=Dx;
Dz=[Dcz Dsz eye(Lz,Lz)];

%% Preallocated a Cell for Approximation Blocks

cmIa1= cell(nYBlocks,nXBlocks); cmIa2= cell(nYBlocks,nXBlocks); cmIa3= cell(nYBlocks,nXBlocks);
cDi1= cell(nYBlocks,nXBlocks); cDi2= cell(nYBlocks,nXBlocks); cDi3= cell(nYBlocks,nXBlocks);
cc= cell(nYBlocks,nXBlocks);


%% Max number of Coefficients per Block
No=numel(A)/SRT

%% Setting Tolerance levels using the PSS and Image Size
tol=MaxInt^2/(10^(pss/10))
tol2=sqrt(tol*(Lx*Ly*Lz));
toln=1e-8;
toln2=sqrt(toln*Lx*Ly*Lz);

%% Max number of Iterations
Max=50000;
Maxp=50000;


%% Set of custom indices if desired
indx=[];
indy=[];
indz=[];

%% Preallocating Variables
nCoe=0;
k=0;
darkb=0;

%% Timers
timeTakenPerBlock = zeros(1,nXBlocks*nYBlocks);

%% Starting Routine
for i = 1:nYBlocks
    tic;
    for j = 1:nXBlocks
        k=k+1;
        B1=cmI1{i,j}; B2=cmI2{i,j}; B3=cmI3{i,j};
        fs(:,:,1)=B1; fs(:,:,2)=B2; fs(:,:,3)=B3;
        
        if ( norm(B1,'fro') + norm(B1,'fro') + norm(B1,'fro') ) > 1e-9

            if usemex == 1;
                [h,cDi1,cDi2,cDi3,beta,c,Q,nore1]=OMP3D_mex(fs,Dx,Dy,Dz,tol,No,indx,indy,indz);
            else;
                [h,cDi1,cDi2,cDi3,beta,c,Q,nore1]=OMP3D(fs,Dx,Dy,Dz,tol,No,indx,indy,indz);
            end;    
              
            cc{k}=c;
  
            cmIa1{i,j}=h(:,:,1);
            cmIa2{i,j}=h(:,:,2);
            cmIa3{i,j}=h(:,:,3);
            nCoe = nCoe + numel(c);
  
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

%% Constructing the approximation
mImage1=cell2mat(cmIa1); mImage2=cell2mat(cmIa2); mImage3=cell2mat(cmIa3);
mIAp(:,:,1)=mImage1; mIAp(:,:,2)=mImage2; mIAp(:,:,3)=mImage3;

%% Caclulate PSNR
sxyz=numel(AA);
mse1=sum(sum((mI1-mImage1).^2)); mse2=sum(sum((mI2-mImage2).^2)); mse3=sum(sum((mI3-mImage3).^2));
MSE=(mse1+mse2+mse3)/(sxyz);

%% Sparsity Ratio
SR=numel(AA)/nCoe

%% Show Original and Approximation
if usejava('jvm');
    figure, imshow(uint8(mIAp)), title('Approximation');
    figure, imshow(imread(ImagePath)), title('Original');
end;

%% Resulting Values
darkb
PSNR=10*log10(MaxInt^2/MSE)
timeTaken = sum(timeTakenPerBlock)
SSIM = ssim(AA, mIAp)
return
