function [PSNR, SR, SSIM] = test_OMP2D(ImagePath, blockWidth, usemex);

if nargin<3 ; usemex = 0; end;
if nargin<2 ; blockWidth = 8; end;
if nargin<1 ; ImagePath = '../../../Images/027_opo0613a_256.jpg'; end;

addpath('../Routines')
addpath('../Functions')

%% Target PSNR
pss=40.5;


%% Load Image
[mImage, sInfo, map] = LoadImage(ImagePath); %load the image 
[mI, bitDepth] = MImage2Grey(mImage, sInfo, map);%convert it to Grey
tempmI = double(imread(ImagePath));
s=floor(size(mI)/blockWidth)
mI=mI(1:s(1)*blockWidth,1:s(2)*blockWidth); 


%% Max Pixel Intensity
MaxIntb=255; %assume 8-bit image


%% Partition Image
[cmI, nYBlocks, nXBlocks] = BlockImage(mI, blockWidth); %for blocking the image in cells
[Lx,Ly]=size(cmI{1,1}); %get block size
cmIa= cell(nYBlocks,nXBlocks);
No=Lx*Ly; 


%% Setting of Tolerance based on PSNR
tol=MaxIntb^2/(10^(pss/10))
tol2=sqrt(tol*(Lx*Ly))


%% Creation of Dictionaries
get_dict_for_2D 

%% Preallocation of Variables
q=0;
indx=[];
indy=[];
nCoe=0;

%% Timers
timeTakenPerBlock = zeros(size(nYBlocks));

%% Starting Routine
for i = 1:nYBlocks
    tic;
    for j = 1:nXBlocks
        q=q+1;
        B=cmI{i,j};
       if usemex == 1
           [cmIa{i,j},Di1{q},Di2{q},cbeta,c{q},Q,vin]=OMP2D_Mex(B,Dx,Dy,tol2^2,No,indx,indy);
       else;
           [cmIa{i,j},Di1{q},Di2{q},beta,c{q},Q]=OMP2D(B,Dx,Dy,tol,No,indx,indy);
       end;

        nCoe = nCoe + numel(c{q}); 
    end
    timeTakenPerBlock(i) = toc;
    fprintf('Block Number: %d/%d - %f\n', i,nYBlocks,timeTakenPerBlock(i));
end 

%% Reassmeble Approximation
mImage=cell2mat(cmIa);%assemble the images from the blocks

%% Results
PSNR=CalcPSNR2(mI,mImage,MaxIntb) 
SR= numel(mI)/nCoe
TotalTimeTaken = sum(timeTakenPerBlock)
SSIM = ssim(mI, mImage)

if usejava('jvm');
    figure(1), imshow(uint8(mI)), title('Original') 
    figure(2), imshow(uint8(mImage)), title('Approximation')
end;
