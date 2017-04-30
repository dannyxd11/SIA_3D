function [PSNR, SR, SSIM] = test_MP2D(ImagePath, blockWidth);

if nargin<2 ; blockWidth = 8; end;
if nargin<1 ; ImagePath = '../../../Images/027_opo0613a_256.jpg'; end;


%% PSNR for breaking tolerance

pss=40.5;

%% Set Target SR

SRT=15;

addpath('../Routines/')
addpath('../Functions/')

%% Load image 

[mImage, sInfo, map] = LoadImage(ImagePath); 
[mI, bitDepth] = MImage2Grey(mImage, sInfo, map);
s=floor(size(mI)/blockWidth)
mI=mI(1:s(1)*blockWidth,1:s(2)*blockWidth); 


%% Max Pixel Intesnsity

MaxIntb=255; 


%% Block Image

[cmI, nYBlocks, nXBlocks] = BlockImage(mI, blockWidth); 
[Lx,Ly]=size(cmI{1,1});

%% Delta / Number of Images.
delta=1/(Lx*Ly);
cmIa= cell(nYBlocks,nXBlocks);

No=Lx*Ly; %Max Number of atoms (Lx*Ly is to stop by error in the approximation)

%% Max number of Iterations
Maxit=50000; %Max number of iteration

%% Setting Tolerance levels using the PSS and Image Size
tol=MaxIntb^2/(10^(pss/10))%set the approximaiton quality according to the desired PSNR (pss)
tol2=sqrt(tol*(Lx*Ly))


%% Generation of Standard Dictionaries

get_dict_TrigPlus 

%% Preallocating Variables
q=0;
nCoe=0;

%% Starting Routine
tic;
for i = 1:nYBlocks
    for j = 1:nXBlocks
        q=q+1;
        B=cmI{i,j};
        [cmIa{i,j},Di1{q},Di2{q},c{q}]=MP2D_Basic(B,Dx,Dy,tol2,No,Maxit);%(Approximate each block)

        nCoe = nCoe + numel(c{q}); 
    end
end 

%% Constructing the approximation
mImage=cell2mat(cmIa);

%% Caclulate PSNR
PSNR=CalcPSNR2(mI,mImage,MaxIntb) 

%% Sparsity Ratio
SR= numel(mI)/nCoe % Sparsity ratio
toc

SSIM = ssim(mI, mImage)

if usejava('jvm');
    figure(1), imshow(uint8(mI)), title('Original') 
    figure(2), imshow(uint8(mImage)), title('Approximation')
end;
