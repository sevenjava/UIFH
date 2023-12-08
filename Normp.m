function nS = Normp(S)

    z = sqrt(sum(S.^2,'all'));
    
    nS = sqrt(S./z);
end