function [IFMap, similar] = GetIFMap(deepf, lowf)
% Get Importance Feature Map of deepf, including IFMap and similar

    n = size(deepf, 3);
    deepf2 = permute(sum(deepf, [1,2]), [3,2,1]);
    deepf2 = deepf2(:);    
    similar = single(zeros(1,n));
    for i = 1:n
        similar(i) = 1 ./ (sum(abs(deepf2(i)-lowf))+1);
    end
    t = mean(similar);
    idx = similar > t;
    IFMap = sum(deepf(:,:,idx), 3);
end