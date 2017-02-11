function [Q]=Orthogonalize(Q,new_atom);   
%
k=size(Q,2) +1;
   for p=1:k-1
     Q(:,k)=new_atom-(Q(:,p)'*new_atom)*Q(:,p); %orthogonalization
    end

