function mVectors = TranslatePrototype( vPrototype, szSpace, dictionary )
% TranslatePrototype translates a vector to construct either a dictionary or a basis
%
% Constructs a matrix whos columns are vectors forming either a redundant dictionary
% or a basis for the Euclidean space of dimension szSpace. Each vector is generated
% by translating one point at a time the discrete values contained in vPrototype,
% i.e.
% szSpace = 3;
% vPrototype = [ 1 ];
% mVectors = [ 1 0 0;
%              0 1 0; 
%              0 0 1 ];
% If dictionary is not specified or set to 1 we apply the 'cut off' approach to
% create a dictionary for the space i.e.
% dictionary = 1;
% szSpace = 3;
% vPrototype = [ 1 1];
% mVectors = [ 1 0 0;
%              1 1 0;
%              0 1 1;
%              0 0 1 ]
% If dictionary is set to 0 we adopt cyclic boundry conditions to create a basis for
% the space, i.e.
% dictionary = 0;
% szSPace = 3;
% vPrototype = [ 1 1];
% mVectors = [ 1 1 0;
%              0 1 1;
%              1 0 1 ];
% 
% Usage:        mVectors = TranslatePrototype( vPrototype, szSpace, dictionary );
%               mVectors = TranslatePrototype( vPrototype, szSpace);
%
% Inputs:
%   vPrototype  vector representing the shape to be translated.
%   szSpace     size of the Euclidean space we want to span.
%   dictionary  0 to generate a basis and 1 to generate a redundant dictionary for 
%               the space, the default is to generate a dictionary.
%
% Outputs:
%   mVectors    matrix whos columns span the space of dimension szSpace
%
% See also GenerateTrapezium

% See   http://www.ncrg.aston.ac.uk/Projects/HNLApprox/ or
%       http://www.nonlinear-approx.info

if nargin < 2
    
    error('Not enough input arguments!')
    
elseif nargin < 3
    
    dictionary = 1;
    
end

if size(vPrototype,2) ~= 1
    
    vPrototype = vPrototype';
    
end

if size(vPrototype,2) ~= 1
    
    error('The prototype atom must be a 1d column vector');
    
end

lSupport = numel(vPrototype);

% For a redundant dictionary we generate vectors in a larger space,
% each one containing a translation of the whole prototype vector vPrototype.
% We then cut at the borders to get the dictionary in the smaller space.
% For the basis we only generate vectors in the existing sized space, where
% each one contains a translation of the whole prototype vector.
% We then construct the vectors at the boundries seperatly.
if dictionary
    
    nVectors = szSpace + lSupport - 1;
    szEnlargedSpace = szSpace + 2*(lSupport - 1);
    
else
    
    nVectors = szSpace;
    szEnlargedSpace = szSpace;
    
end

% Initialize the dictionary or basis
mVectors = zeros(szEnlargedSpace,nVectors);

% Index's for the start and end of the prototype vector within each vector
iStartOfAtom = 1;
iEndOfAtom = lSupport;

% Generate all the vectors that will contain all the points from vPrototype 
% starting from the left and translating to the right by one point at a
% time. 
iAtom = 1;
while (szEnlargedSpace - iStartOfAtom + 1) >= lSupport
    
    mVectors(iStartOfAtom:iEndOfAtom, iAtom) = vPrototype;
    iStartOfAtom = iStartOfAtom + 1;
    iEndOfAtom = iEndOfAtom + 1;
    iAtom = iAtom + 1;
    
end

% Apply either the cyclic boundry conditions or the 'cut off' approach.
if dictionary
    
    % Cut the dictionary at the borders.
    mVectors = mVectors(lSupport:end-lSupport +1,:);
    
else
    
    % To get a basis generate the aditional vectors where the points of the
    % prototype atom which extend past the length of the support are 
    % repeated at the beginning of the vector.
    while szSpace >= iStartOfAtom

        lLeftOfAtom = szSpace - iStartOfAtom + 1;
        lRightOfAtom = lSupport - lLeftOfAtom;

        mVectors(iStartOfAtom:szSpace, iAtom) = vPrototype(1:lLeftOfAtom);
        mVectors(1:lRightOfAtom, iAtom) = vPrototype(lLeftOfAtom + 1:end);

        iStartOfAtom = iStartOfAtom + 1;
        iAtom = iAtom + 1;

    end   
            
end