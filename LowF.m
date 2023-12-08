function gaborVector = LowF(img)
% utilize the opponent color space and Gabor filters to extract low-level features

    % convert rgb color space to opponent color space
    ocsimg = rgb2ocs(img);
    % Downsampling each component 4 times to obtain 12 maps
    xcyimg = cell(3,4);
    for i = 1:3
        timg = ocsimg(:,:,i);
        for j = 1:4            
            timg = timg(1:2:end,1:2:end);
            xcyimg{i,j} = timg;
        end
    end
    ximg = xcyimg(:); 
    % Filter each map using Gabor filter bank in 12 orientation,resulting in
    % 144 maps.
    ang = [0, pi/12, 2*pi/12, 3*pi/12, 4*pi/12, 5*pi/12, 6*pi/12, 7*pi/12, 8*pi/12, 9*pi/12, 10*pi/12, 11*pi/12];
    gaborArray = gabor(8.0, ang);
    h = length(ximg);
    w = length(gaborArray);
    gaborVector = single(zeros(h,w));
    for i = 1:h
        gabortmp = imgaborfilt(ximg{i}, gaborArray);
        for j = 1:w
            gaborVector(i,j) = sum(gabortmp(:,:,j),[1 2]);
        end
    end
    gaborVector = gaborVector(:);
end

