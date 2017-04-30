Lx=blockWidth;
Ly=blockWidth;
a=2; %redundacy for trigonometric dictionaries
DDxc=DCos(Lx,a*Lx,a);
DDxc=NormDict(DDxc); %normalize redundant Cosine Dictionary
%
DDxs=DSin(Ly,a*Ly,a);
DDxs=NormDict(DDxs); %normalize redundant Sine Dictionary
%
atom1 = GenerateTrapezium(3, 1); %genetare prototype (hat with Support 3)
d1 = TranslatePrototype(atom1, blockWidth, 1); %genetare dictionary by translation
d1n=NormDict(d1);
%
atom2 = GenerateTrapezium(5, 1); %genetare prototype (hat with Support 5)
d2 = TranslatePrototype(atom2, blockWidth, 1); %genetare dictionary by translation
d2n=NormDict(d2);
%
atom3= GenerateTrapezium(7, 1); %genetare prototype (hat with Support 7)
d3 = TranslatePrototype(atom3, blockWidth, 1);
d3n=NormDict(d3);
%
haar2=[-0.5 0.5];
d4 = TranslatePrototype(haar2, blockWidth, 1);
d4n=NormDict(d4);
%
haar4=[-0.25 -0.25  0.25 0.25];
d5=TranslatePrototype(haar4, blockWidth, 1);
d5n=NormDict(d5);
%
haar6=[-0.25 -0.25 -0.25  0.25 0.25 0.25];
d6=TranslatePrototype(haar6, blockWidth, 1);
d6n=NormDict(d6);
%
Ds1=[d1n d2n d3n d4n d5n d6n];
Dx=[DDxc DDxs eye(blockWidth,blockWidth) Ds1];
[Dx,irm]=RemoveSimilarAtoms(Dx);
Dy=Dx;
