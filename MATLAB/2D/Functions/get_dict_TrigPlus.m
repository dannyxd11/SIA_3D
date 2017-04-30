Lx=blockWidth;
Ly=blockWidth;
a=2; %redundacy for trigonometric dictionaries
DDxc=DCos(Lx,a*Lx,a);
DDxc=NormDict(DDxc); %normalize redundant Cosine Dictionary
%
DDxs=DSin(Ly,a*Ly,a);
DDxs=NormDict(DDxs); %normalize redundant Sine Dictionary
%
Dx=[DDxc DDxs eye(blockWidth,blockWidth)]; %Cosie-Sine dictionary + the Standard Euclidean basis: eye(blockWidth,blockWidth)
Dy=Dx;
