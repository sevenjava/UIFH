function [test_pca,query_pca] = he_pca(XTrain,XTest,Query,dim)
% % 
% % %refer to:http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
    epsilon=1*10^(-5); %%%%%%%%% a small constant 
    XTrain(isnan(XTrain)) = 0;
    x=XTrain';
    avg=mean(x,1);
    x=x-avg;
    sigma=x*x'/size(x,2);
    [u,s,~]=svd(sigma);

    y=XTest';
    avg=mean(y,1);
    y=y-avg;
    yRot=u'*y;
%     yPCA=yRot(1:dim,:)'; %%%%pca
    
    yPCAWhite=diag(1 ./ sqrt(diag(s) + epsilon)) * yRot;
    test_feat = yPCAWhite(1:dim, :)';
% % % % % % % %query feature pca-whiting % % % % % % % % % % % % % % %     
    if ~isempty(Query)
        q=Query';
        qy=q-mean(q,1);
        
        q_xRot=u'*qy;
%         qyPCA = q_xRot(1:dim, :)'; %%%%%query pca 
        q_xPCAWhite=diag(1./sqrt(diag(s)+epsilon))*q_xRot;
        query_feat=q_xPCAWhite(1:dim,:)';
    end
% % % % % % normalize%%%%%%%%%%%%%%%
    test_pca=normalize(test_feat,2,"norm");
    query_pca=normalize(query_feat,2,"norm");

end