function [cImage, nYBlocks, nXBlocks] = BlockImage(mImage, blockWidth)
% Check the image can be split exactly into square blocks of width
% blockWidth
[y x] = size(mImage);

if mod(y,blockWidth) || mod(x,blockWidth);
    
    error('\n\nA %ix%i image cannot be split exactly into square blocks of width %i',y,x,blockWidth);
    
    % Present option of possible block sizes of resize
%     fprintf('A %ix%i image cannot be split exactly into square blocks of width %i',y,x,blockWidth);
%     option = input('Enter 1 to exit or 2 to reduce the size of the input image\n');
%     
%     if option == 1

end

% Split the image into a cell array of square blocks
cImage = mat2cell(mImage, blockWidth*ones(1, (y/blockWidth)), ...
    blockWidth*ones(1, (x/blockWidth)));
[nYBlocks, nXBlocks]  = size(cImage);