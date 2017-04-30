function vTrapezium = GenerateTrapezium( lBase, lTop )
% GenerateTrapezium generates a vector representing a trapezium
%
% Generates a discrete vector of points representing the vertical distance between
% the base of an isosceles trapezium and the other three sides. You choose the length
% of the trapeziums base, this will 2 less than the size of vTrapezium as the base
% values are zero. You also choose the length of the trapeziums top.
% 
% Usage:        vTrapezium = GenerateTrapezium( lBase, lTop );
%               vTrapezium = GenerateTrapezium( lBase );
%
% Inputs:
%   lBase       number of discrete points for the trapeziums base
%   lTop        number of discrete points for the trapeziums top, the default is 1
%
% Outputs:
%   vTrapezium  column vector of points representing the vertical distance between 
%               the base of an isosceles trapezium and the other three sides.
%
% See also TranslatePrototype

% See   http://www.ncrg.aston.ac.uk/Projects/HNLApprox/ or
%       http://www.nonlinear-approx.info


% The default trapezium will be a triangle so we only require the length of
% the trapziums base and not length of its top.
if nargin < 1

    error('Length of trapeziums base is required!');

elseif nargin < 2

    lTop = 1;     

end

% Number of discrete points used to represent both the non parallel sides
lSlopes = lBase - lTop;

% We are generating an isoscelies trapezium so the length of the non
% parallel sides need to be equal.
if mod((lSlopes),2)
    
    error('The trapezium cannot be skewed!')
    
end

% The length of the slopes can't be negative, to allow the generation of
% haar's we allow the trapezium to have four parallel sides.
if  lSlopes < 0
    
    error('The length of the top is too great!');
    
end

lSlope = (lSlopes)/2;

vLeftSlope = CalculateLineVector( lSlope );
vRightSlope = Reverse( vLeftSlope );
    
% Construct the trapezium
vTop = ones(lTop,1);
vTrapezium = [ vLeftSlope; vTop; vRightSlope ];

end


function vReversed = Reverse(v)

% Puts the components of a vector in reverse order

    vReversed = v(numel(v):-1:1);

end


function vLine = CalculateLineVector( lLine )

% Given a number of discrete points calculates a vector vLine containing 
% the y coordinates of a line assuming that there are lLine equidistant x 
% coordinates.  The line starts at (0,0) and ends at (1,1).

    vLine = (1:lLine)' * (1/(lLine + 1));

end