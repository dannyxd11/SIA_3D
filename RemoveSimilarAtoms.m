function [ mUniqueDictionary iRemovedAtoms ] = RemoveSimilarAtoms( mDictionary, tol )
% RemoveSimilarAtoms removes any similar atoms from a given dictionary
%
% Normalises the dictioanry and then removes any atoms that have an inner product of 
% within tol of 1.
%
% Useage:       [ mUniqueDictionary iRemovedAtoms ] = RemoveSimilarAtoms( mDictionary,...
%                   tol );
%
% Inputs:
%   mDictionary     dictionary of atoms
%   tol             tolerance of how similar atoms can be
%
% Outputs:
%   mUniqueDictionary   dictionary with similar atoms removed
%   iRemovedAtoms       index of the similar atoms

% See   http://www.ncrg.aston.ac.uk/Projects/HNLApprox/
%       http://www.nonlinear-approx.info/
if nargin < 2
    tol = 1e-13;
end

% The dictonary needs to be normalized, similar atoms will have an inner
% product equal to 1;

mInnerProducts = NormDict(mDictionary)'*NormDict(mDictionary);
mUniqueInnerProducts = triu(mInnerProducts,1);

% If  the innerproduct between atoms is within tol of 1 then consider them
% to be the same.
[ i, j ] = find(mUniqueInnerProducts > (1-tol));
[ i, j ] = find(abs(mUniqueInnerProducts) > (1-tol));

% The i j pairs correspond to atoms that are considered to be the same,
% i.e. i(1) is the same as j(1).
% Therefore we remove the unique elemnts of either i or j.

% First we check that both i and j have the same number of unique elements,
% if not then then alter the tol chosen
iRemovedAtoms = unique(j);
if numel(iRemovedAtoms) ~= numel(unique(i))
    
    error('Tolerance for atom removal may be incorrect!');
    
end

mUniqueDictionary = mDictionary;
mUniqueDictionary(:,iRemovedAtoms) = [];
