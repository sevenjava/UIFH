function S = SpaIIE(IFMap)
% Pixel-level evaluation (Spatial Information Importance Evalutation)

    [u0, v0] = Center(IFMap);

    Cau = Cauchy(IFMap, u0, v0);

    S = IFMap .* Cau;

    S = Normp(S);
end