function [mImage, sInfo, map] = LoadImage(imagePath)
% Takes the path to an image file as input and retutns a matrix
% representing the image and a structure containing meta information

% Check if file exists
if exist(imagePath,'file') ~= 2
       
    error('%s not found in current directory or on MATLAB search path!',imagePath);
    
end

% Assume its an image if imfinfo doesn't throw an exception
try
    
    sInfo = imfinfo(imagePath);
    
catch ME
    
    error('%s does not appear to be an image',imagePath);
    
end

% Read the image into a matrix    
try
    
    [mImage, map] = imread(imagePath);
    
catch ME
    
    error('%s cannot be read!',imagePath);
    
end