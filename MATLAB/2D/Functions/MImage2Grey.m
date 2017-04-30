function [mImageGrey, bitDepth] = MImage2Grey(mImage,sInfo,map)
% Converts indexed and truecolor image matrices to greyScale.
% Retutns  a double precision greyscale image matrix and the bitdepth of each pixel.
% If the bitdepth is not in the sInfo structure the function approximates
% it using the max and min pixel values.

if nargin < 2 || nargin > 3
    
    error('Wrong number of input arguments');
    
elseif nargin == 2
    
    map = [];
    
end
        
% Check ColorType field exists
if isfield(sInfo,'ColorType')
    
    colorType = sInfo.ColorType;
    
else
    
    error('Cannot convert to greyscale as oringinal color type is not availiable!')
    
end

bitDepth = 0;
if isfield(sInfo,'BitDepth')
        
        bitDepth = sInfo.BitDepth;
        
else
        
        fprintf('BitDepth not avaliable using an approximation!\n');        
        
end   

if strcmp(colorType, 'grayscale')
    
    mImageGrey = double(mImage);
    
    if isempty(bitDepth) || bitDepth == 0
        
        bitDepth = ApproxBitdepth(mImageGrey);
        
    end
    
elseif strcmp(colorType, 'truecolor')
        
    mImageGrey = double(rgb2gray(mImage));
    
    if isempty(bitDepth) || bitDepth == 0
        
        bitDepth = ApproxBitdepth(mImageGrey);
        
    else
   
        % Assume 3 colorplanes
        bitDepth = bitDepth/3;
        
    end
    
    fprintf('Supplied image is truecolor, converted to greyscale %i bit image\n', bitDepth);
    
    
elseif strcmp(colorType, 'indexed')
    
    if isempty(map)
        
        error('Cannot convert indexed image without color map to grey scale!')
        
    else
        
        mImageGrey = double(ind2gray(mImage,map));
        
        if isempty(bitDepth) || bitDepth == 0
        
            bitDepth = ApproxBitdepth(mImage);
        
        end
        
        fprintf('Supplied image is indexed, converted to greyscale %i bit image\n', bitDepth);
        
    end

end
