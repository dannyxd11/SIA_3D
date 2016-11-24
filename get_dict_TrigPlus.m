Lx=bw;
Ly=bw;
a=2; %redundacy for trigonometric dictionaries
DDxc=DCos(Lx,a*Lx,a);
DDxc=NormDict(DDxc); %normalize redundant Cosine Dictionary
%
DDxs=DSin(Ly,a*Ly,a);
DDxs=NormDict(DDxs); %normalize redundant Sine Dictionary
%
Dx=[DDxc DDxs eye(bw,bw)]; %Cosie-Sine dictionary + the Standard Euclidean basis: eye(bw,bw)
Dy=Dx;
