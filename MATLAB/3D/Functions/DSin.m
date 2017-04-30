function mSines = DSin( szSpace, nFrequencies, redundancy )
% DCos generates a matrix whos columns are discrete cosine vectors.
%
% Returns discrete cosine vectors that belong to the Euclidean space of size szSpace.  The 
% deafult is to return a basis for the space.
%
% Usage             mCosines = DCos( szSpace, nFrequencies, redundancy );
%                   mCosines = DCos( szSpace, nFrequencies );
%                   mCosines = DCos( szSpace );
%
% Inputs:
%   szSpace         the size of the Euclidean space the vectors should belong to
%   nFrequencies    number of frequencies to use starting from 0.  If not specified will
%                   be the same as the size of the space
%   redundancy      redundancy of the dictionary, the default is 1 (basis)
%
% Outputs:
%   mCosines        matrix whos columns are discrete cosine vectors.

% See   http://www.ncrg.aston.ac.uk/Projects/HNLApprox/
%       http://www.nonlinear-approx.info/

%error(nargchk(1,3,nargin));

if nargin <2

    nFrequencies = szSpace;
    redundancy = 1;
    
elseif nargin < 3
    
    redundancy = 1;
      
end

n = 1:nFrequencies;
k = 1:szSpace;
N = redundancy*szSpace;
mSines = sin( pi*(2*k-1)'*(n/(2*N)) );
