function feat = uifh(deepf, lowf)
% The information importance evaluation of UIFH

    % deep feature maps evaluation
    [IFMap, sim] = GetIFMap(deepf, lowf);
    
    % Pixel-level evaluation
    S = SpaIIE(IFMap);
    deepf = deepf .* S;
    feat = permute(sum(deepf,[1,2]), [1,3,2]);
    
    % Block-level evaluation
    b = BlockIIE(feat, sim);
    feat = feat .* b;
end