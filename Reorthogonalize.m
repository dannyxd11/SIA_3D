function [Q]=Reorthogonalize(Q,zmax);
%
k=size(Q,2);
    for zi=1:zmax
    for p=1:k-1
      Q(:,k)=Q(:,k)-(Q(:,p)'*Q(:,k))*Q(:,p); %re-orthogonalization
    end
    end
