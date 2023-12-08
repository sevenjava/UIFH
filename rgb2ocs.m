function [Oc] = rgb2ocs(I)
%%%convert a RGB image to Opponent color image
    oimg=single(I)./255;
    
    % 抽取RGBY分量值
    rimg=oimg(:,:,1);
    gimg=oimg(:,:,2);
    bimg=oimg(:,:,3);
% % % 2010TPAMI,Evaluating color descriptors for object and scene
% % % recognition
    Oc(:,:,1)=(rimg-gimg)./sqrt(2);
    Oc(:,:,2)=(rimg+gimg-2*bimg)./sqrt(6);
    Oc(:,:,3)=(rimg+gimg+bimg)./sqrt(3);

end