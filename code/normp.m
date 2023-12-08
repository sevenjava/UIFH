function nw = normp(S)
    z=sum(S.^2,'all');
    z=sqrt(z);
    nw = (S/z).^(1/2); %%%%%power-scaling
end