%Example to test the routine MP2D_Basic for Malaga GPU implementation
%To change the Image change ImagePath
%To change the tolerance for the approximation change pss 
%To change the block's size change BlockWidth

function [PSNR, SR, SSIM] = Test_OMP2D(ImagePath, BlockWidth);
bw = BlockWidth;
if nargin<2 ; bw = 8; BlockWidth = 8; end;
if nargin<1 ; ImagePath = 'Images/kodak7.png'; end;
%ImagePath='Images/027_opo0613a.jpg'
pss=40.5;
clear c
%
[mImage, sInfo, map] = LoadImage(ImagePath); %load the image 
[mI, bitDepth] = MImage2Grey(mImage, sInfo, map);%convert it to Grey
%
%
s=floor(size(mI)/bw)
%
mI=mI(1:s(1)*bw,1:s(2)*bw); %rezise the images to be divisible by BlockWidth;
%
MaxIntb=255; %assume 8-bit image
%
tic;
[cmI, nYBlocks, nXBlocks] = BlockImage(mI, BlockWidth); %for blocking the image in cells
%
[Lx,Ly]=size(cmI{1,1}); %get block size
delta=1/(Lx*Ly);
cmIa= cell(nYBlocks,nXBlocks);
No=Lx*Ly; %Max Number of atoms (Lx*Ly is to stop by error in the approximation)
Maxit=10000; %Max number of iteration
tol=MaxIntb^2/(10^(pss/10))%set the approximaiton quality according to the desired PSNR (pss)
tol2=sqrt(tol*(Lx*Ly))
%
%====== input dictionaries===================
get_dict_for_2D %get dictionaries Dx and Dy
%============================================
%
%======= loop for approximating each block (sequencially with sigle processor)
q=0;
indx=[];
indy=[];
nCoe=0;
for i = 1:nYBlocks
for j = 1:nXBlocks
 q=q+1;
 B=cmI{i,j};
 [cmIa{i,j},Di1{q},Di2{q},beta,c{q},Q]=OMP2D_m(B,Dx,Dy,tol,No,indx,indy);%(Approximate each block)
% [cmIa{i,j},Di1{q},Di2{q},cbeta,c{q},Q,vin]=OMP2D_Mex(B,Dx,Dy,tol2^2,No,indx,indy);
%save the outputs in cells
 nCoe = nCoe + numel(c{q}); %count the total nonzero coefficients
end
end 
%===================================================================================
%
clear mImage
mImage=cell2mat(cmIa);%assemble the images from the blocks
fprintf('PSNR of the approximation\n');
PSNR=CalcPSNR2(mI,mImage,MaxIntb) % PSNR of the approximation
fprintf('Sparsity Ratio');
SR= numel(mI)/nCoe % Sparsity ratio
toc;
SSIM = ssim(mI, mImage)
%figure(1)
%imshow(uint8(mI)) %original image
%figure(2)
%imshow(uint8(mImage))% approximated image
