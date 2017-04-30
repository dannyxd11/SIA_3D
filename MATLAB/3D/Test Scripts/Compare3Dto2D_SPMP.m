ImagesToTest = [string('Images/kodak1.png');
                'Images/kodak2.png';
                'Images/kodak3.png';
                'Images/kodak4.png';
                'Images/kodak5.png';
                'Images/kodak6.png';
                'Images/kodak7.png';                                
                'Images/kodak8.png';
                'Images/kodak9.png';
                'Images/kodak10.png';
                'Images/kodak11.png';
                'Images/kodak12.png';
                'Images/kodak13.png';
                'Images/kodak14.png';                                                
                'Images/kodak15.png';
                'Images/kodak16.png';
                ]
BlockSize = 48;
Results = cell(1,size(ImagesToTest,1));
n = 1;
parfor n = 1:size(ImagesToTest,1)
    Results{n}.name=char(ImagesToTest(n));
    Results{n}.blockSize = BlockSize;
    
    [PSNR3D, SR3D, SSIM3D] = test_OMP3D_Dan(char(ImagesToTest(n)), BlockSize);
    
    Results{n}.PSNR3D = PSNR3D;
    Results{n}.SR3D = SR3D;
    Results{n}.SSIM3D =SSIM3D;
    
    [PSNR2D, SR2D, SSIM2D] = Test_OMP2D(char(ImagesToTest(n)), BlockSize);
    
    Results{n}.PSNR2D = PSNR2D;
    Results{n}.SR2D = SR2D;
    Results{n}.SSIM2D = SSIM2D;
end;

Results
a = cell2mat(Results);