function psnr = CalcPSNR2( mImage1, mImage2, maxIntensity )
% CalcPSNR returns the PSNR between the original image and its approximation
%
% Calculates the Peak Signal to Noise Ratio (PSNR) between 2 matrices containing
% pixel intensity values.
%
% Usage:            psnr = CalcPSNR( mImage1, mImage2 );
%
% Inputs:
%   mImage1         matrix of pixel intensity values representing the original image
%   mImage2         matrix of pixel intensity values representing the approximated image
%   maxIntensity    maximum allowed pixel intensity, defualt is 256 (8 bit image) 
%
% Outputs:
%    psnr       the PSNR resulting from the  approximation
%

error(nargchk(2,3,nargin));

if nargin < 3
    
    maxIntensity = 255;
    
end

%e = double(mImage1) - double(mImage2);
%[ y, x ] = size(e);
%mse = sum( e(:).^2 )/( x*y );
[ y, x ] = size(mImage1);
mse=sum(sum((mImage1-mImage2).^2));
mse=mse/( x*y );
psnr = 10*log10(((maxIntensity)^2)/mse);
