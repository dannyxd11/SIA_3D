%
function [beta]=Biorthogonalize(beta,Qk,new_atom,nork);
%

beta = beta - Qk * (new_atom'*beta) / nork;

%

