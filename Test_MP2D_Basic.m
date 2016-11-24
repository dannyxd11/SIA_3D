%Example to test the routine MP2D_Basic for Dan
%To change the Image change ImagePath
%To change the tolerance for the approximation change pss 
%To change the block's size change BlockWidth
ImagePath='027_opo0613a.jpg'
pss=40
BlockWidth=16%blocksize 
bw=BlockWidth;
%
%===============================
%Step 1) load the image
%
[mImage, sInfo, map] = LoadImage(ImagePath); %e 
%==============================
%Step 2) convert it to Grey
%
[mI, bitDepth] = MImage2Grey(mImage, sInfo, map);%
%===============================
%
s=floor(size(mI)/bw)
%
mI=mI(1:s(1)*bw,1:s(2)*bw); %rezise the images to be divisible by BlockWidth;
%
MaxIntb=255; %assume 8-bit image
%
tic;
%===============================
%Step 3) makes a partition for the image into cells
%
[cmI, nYBlocks, nXBlocks] = BlockImage(mI, BlockWidth); %for blocking the image in cells
%===============================
[Lx,Ly]=size(cmI{1,1}); %get block size
delta=1/(Lx*Ly);
cmIa= cell(nYBlocks,nXBlocks);
No=Lx*Ly; %Max Number of atoms (Lx*Ly is to stop by error in the approximation)
Maxit=50000; %Max number of iteration
tol=MaxIntb^2/(10^(pss/10))%set the approximaiton quality according to the desired PSNR (pss)
tol2=sqrt(tol*(Lx*Ly))
%
%====== Input Dictionary===================
%Step 4)
get_dict_TrigPlus %get dictionaries Dx and Dy
%
%============================================
%Setp 5)  loop for approximating each block (sequencially with sigle processor)
q=0;
nCoe=0;
for i = 1:nYBlocks
for j = 1:nXBlocks
 q=q+1;
 B=cmI{i,j};
 [cmIa{i,j},Di1{q},Di2{q},c{q}]=MP2D_Basic(B,Dx,Dy,tol2,No,Maxit);%(Approximate each block)
%save the outputs in cells
 nCoe = nCoe + numel(c{q}); %count the total nonzero coefficients
end
end 
%===================================================================================
%Setp 6)
clear mImage
mImage=cell2mat(cmIa);%assemble the images from the blocks
fprintf('PSNR of the approximation\n');
PSNR=CalcPSNR2(mI,mImage,MaxIntb) % PSNR of the approximation
fprintf('Sparsity Ratio');
SR= numel(mI)/nCoe % Sparsity ratio
toc;
figure(1)
imshow(uint8(mI)) %original image
figure(2)
imshow(uint8(mImage))% approximated image
