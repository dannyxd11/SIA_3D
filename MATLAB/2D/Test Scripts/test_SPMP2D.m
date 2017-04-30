function [PSNR, SR, SSIM] = test_SPMP2D(ImagePath, blockWidth);
if nargin<3 ; usemex = 0; end;
if nargin<2 ; blockWidth = 8; end;
if nargin<1 ; ImagePath = '../../../Images/027_opo0613a_256.jpg'; end;


addpath('../Routines')
addpath('../Functions')

pss=40.5;

%% Load Image
[mImage, sInfo, map] = LoadImage(ImagePath); 
[mI, bitDepth] = MImage2Grey(mImage, sInfo, map);
tempmI = double(imread(ImagePath));
s=floor(size(mI)/blockWidth)
mI=mI(1:s(1)*blockWidth,1:s(2)*blockWidth);

%% Max Pixel Intensity
MaxIntb=255;

%% Partition Image
[cmI, nYBlocks, nXBlocks] = BlockImage(mI, blockWidth); 
[Lx,Ly]=size(cmI{1,1}); 
cmIa= cell(nYBlocks,nXBlocks);
No=Lx*Ly;



%% Generate Dictionary
get_dict_for_2D 

%% Preallocate Variables
q=0;
indx=[];
indy=[];
nCoe=0;

%% Projection Stage. -1 - No Projection, Otherwise, project every n steps.
lstep = -1;

%% Max Iteration
Max=50000;
Maxp=50000;

%% Setting Tolerance based on PSNR
MSE = (255^2)/(10^(pss/10))
SSE = MSE*blockWidth*blockWidth;
tol=MaxIntb^2/(10^(pss/10))
toln=1e-8;
tol = SSE
tol2=sqrt(tol*(Lx*Ly))

%% Timers
timeTakenPerBlock = zeros(size(nYBlocks));

%% Starting Routine
for i = 1:nYBlocks
    tic;
    for j = 1:nXBlocks
        q=q+1;
        B=cmI{i,j};
        if usemex == 1;
            [cmIa{i,j},Di1{q},Di2{q},c{q}] = SPMP2D_Mex(B,Dx,Dy,tol2,No,toln,lstep,Max,Maxp,indx,indy);%(Approximate each block)
        else;
            [cmIa{i,j},Di1{q},Di2{q},c{q}] = SPMP2D(B,Dx,Dy,tol,No,toln,lstep,Max,Maxp,indx,indy);
        end;
        
         nCoe = nCoe + numel(c{q}); %count the total nonzero coefficients
    end
    timeTakenPerBlock(i) = toc;
    fprintf('Block Number: %d/%d - %f\n', i,nYBlocks,timeTakenPerBlock(i));
end 


%% Reassemble Approximation
mImage=cell2mat(cmIa);

%% Results
PSNR=CalcPSNR2(mI,mImage,MaxIntb) 
SR= numel(mI)/nCoe 
TotalTimeTaken = sum(timeTakenPerBlock)
SSIM = ssim(mI, mImage)

if usejava('jvm');
    figure(1), imshow(uint8(mI)), title('Original') 
    figure(2), imshow(uint8(mImage)), title('Approximation')
end;
