function b = BlockIIE(feat, sim)
% Block-level evaluation (Block Information Importance Evalutation)

    n = length(feat);
    b = zeros(1,n);
    
    sim = sim ./ max(sum(sim),1e-5);
    feat = feat ./ max(sum(feat),1e-5);
    for i = 1:n
        if feat(i) > 0
            b(i) = log(1+sim(i)/feat(i).^2);
        else
            b(i) = 0;
        end
    end
    
end