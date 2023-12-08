function [u0, v0] = Center(IFMap)
% get the centre point of locations
    Z = IFMap;
    [h,w] = size(Z);
    rate = 0.2;     % 0.1~0.25
    m = round(rate*h*w);
    idx = uint8(zeros(m,2));
    for i = 1:m
        maxv = max(Z,[],'all');
        [x,y] = find(Z == maxv);
        Z(x,y) = -inf;
        idx(i,1) = mean(x);
        idx(i,2) = mean(y);
    end
    u0 = floor(mean(idx(:,1)));
    v0 = floor(mean(idx(:,2)));
end