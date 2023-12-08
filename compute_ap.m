function ap = compute_ap(query, gt_path, resultfile, rank_list_path)
    % load ranked list and Groundtruth
    % ranked_list = importdata([rank_list_path,query,'.mat']);
    ranked_list = resultfile;
    good_set = importdata([gt_path,query,'_good.txt']);
    ok_set = importdata([gt_path,query,'_ok.txt']);
    junk_set = importdata([gt_path,query,'_junk.txt']);
    % combine good_set and ok_set as relative images
    pos_set = [good_set;ok_set];
    old_recall = 0.0;
    old_precision = 1.0;
    ap = 0.0;
    intersect_size = 0;
    j = 0;
    for i=1:size(ranked_list,1)
        if ismember(ranked_list(i),junk_set)
            continue;
        end
        if intersect_size == size(pos_set,1)
            break;
        end
        if ismember(ranked_list(i),pos_set)
            intersect_size = intersect_size + 1;
        end
        
        recall = intersect_size / size(pos_set,1);
        precision = intersect_size / (j + 1.0);
        ap = ap + (recall - old_recall)*((old_precision + precision) / 2.0);
    
        old_recall = recall;
        old_precision = precision;
        j = j + 1;    
    end
end

