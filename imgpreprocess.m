function img = imgpreprocess(imdata)
    if size(imdata,3) == 1
        imdata = cat(3, imdata, imdata, imdata);
    end
    img = single(imdata);
    [h, w, ~] = size(img);
    
    minsize = 227;
    if h < minsize
        img = imresize(img, [minsize, w]);
    end
    [h, w, ~] = size(img);
    if w < minsize
        img = imresize(img, [h, minsize]);
    end
end
