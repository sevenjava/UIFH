function resultfiles = rank_list(features_pca, query_feature_pca, files, query_files, rank_list_path, qe)
    % if ~exist(rank_list_path,'dir')
    %     mkdir(rank_list_path);
    % else
    %     delete(strcat(rank_list_path, '/*.mat'));
    % end
    
    num_return = size(files,1);
    num_query = size(query_feature_pca,1);
    resultfile = cell(num_return,1);
    resultfiles = cell(num_return,num_query);
    
    % L2
    for i=1:num_query
        dist=pdist2(query_feature_pca(i,:),features_pca,'euclidean');
        [L1_sorted, index] = sort(dist);
        
        %QE（）
        if qe~=0
            Q=zeros(1,size(features_pca,2));
            for  s=1:qe
                Q=Q+features_pca(index(s),:);
            end
            Q=normalize(Q);
            diff = repmat(Q,[num_return,1]) - features_pca;
            diff = diff.*diff;
            dist = sum(diff,2);
            [L2_sorted, index] = sort(dist);
        end
 
        for k=1:num_return
            result_image_name = split(files(index(k)).name,'.');
            resultfile{k,1} = result_image_name{1};
            resultfiles{k,i} = result_image_name{1};
        end

        % save(strcat(rank_list_path, query_files{i},'.mat'), 'resultfile');
    end
end
