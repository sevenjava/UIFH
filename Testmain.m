% % for an image, if the single size is < 227, resize it to 227.

%% 1 setting params for test

reset(gpuDevice(1));

datasetpath = strcat('H:/data/datasets/');
modelpath = strcat('H:/data/networks/');
rank_list_path = './results/';

netmodels = ["vgg16", "repvgg", "resnet50", "alexnet"];
netmodelfilenames = ["vgg16", "repvgglarge", "resnet50", "alexnet"];
layers = ["pool5", "mul4_24", "activation_49_relu", "pool5"];
dims = [512, 640, 2048, 256];

idx = 1;
netmodel = netmodels(idx);
dim = dims(idx);
net = importdata(strcat(modelpath, netmodelfilenames(idx), '.mat'));
layer = layers(idx);

% % loading the file
% Oxford5k
ofiles = dir(strcat(datasetpath, '/oxford5k/oxbuild_images/*.jpg'));
ofile_num = size(ofiles,1);
oqfiles = dir(strcat(datasetpath, '/oxford5k/query_images_crop_crow/*.jpg'));
oqfile_num = size(oqfiles,1);
oqueryPic = importdata(strcat(datasetpath, '/oxford5k/Oqueryimg.mat'));
ogt_files = '../../data/datasets/oxford5k/gt_files_170407/';

% Paris6k
pfiles = dir(strcat(datasetpath, '/paris6k/paris_images/*/*.jpg'));
pfile_num = size(pfiles,1);
pqfiles = dir(strcat(datasetpath, '/paris6k/query_images_crop_crow/*.jpg'));
pqfile_num = size(oqfiles,1);
pqueryPic = importdata(strcat(datasetpath, '/paris6k/Pqueryimg.mat'));
pgt_files = '../../data/datasets/paris6k/gt_files_120310/';

% Holidays
hfiles = dir(strcat(datasetpath, '/Holidays_upright/jpg/*.jpg'));
hfile_num = size(hfiles,1);
gnd_holidays = importdata(strcat(datasetpath, '/Holidays_upright/','gnd_holidays.mat'));
[gnd_h,imlist,qidx] = deal(gnd_holidays.gnd,gnd_holidays.imlist,gnd_holidays.qidx);

% Flickr100k
ffiles = dir(strcat(strcat(datasetpath, '/Flickr100K/oxc1_100k/*/*.jpg')));
ffile_num = size(ffiles,1);

% rOxford5k
rofiles = dir(strcat(datasetpath, '/roxford5k/jpg/*.jpg'));
roqfiles = dir(strcat(datasetpath, '/roxford5k/query_images_crop/*.jpg'));

% rParis6k
rpfiles = dir(strcat(datasetpath, '/rparis6k/jpg/*.jpg'));
rpqfiles = dir(strcat(datasetpath, '/rparis6k/query_images_crop/*.jpg'));

test_datasets = {'roxford5k', 'rparis6k'};
data_root = datasetpath;

%% 2 extracting representation

disp(datetime);
% % 2.1 Oxford5k dataset
ofeatures = zeros(ofile_num, dim);
tic;
parfor i = 1:ofile_num
    img = imread(strcat(ofiles(i).folder,'/',ofiles(i).name));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    ofeatures(i,:) = feat;
end
toc
ofeatures_norm = normalize(ofeatures,2,'norm');

qofeatures = zeros(oqfile_num,dim);
tic;
parfor i = 1:oqfile_num
    img = imread(strcat(oqfiles(i).folder,'/',oqueryPic{i,2},'.jpg'));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    qofeatures(i,:) = feat;
end
toc
qofeatures_norm = normalize(qofeatures,2,"norm");

% % 2.2 Paris6k
pfeatures = zeros(pfile_num,dim);
tic;
parfor i = 1:pfile_num
    img = imread(strcat(pfiles(i).folder,'/',pfiles(i).name));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    pfeatures(i,:) = feat;
end
pfeatures_norm = normalize(pfeatures,2,'norm');
toc

pqfeatures = zeros(pqfile_num,dim);
tic;
parfor i = 1:pqfile_num
    img = imread(strcat(pqfiles(i).folder,'/',pqueryPic{i,2},'.jpg'));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);
    
    feat = uifh(deepf, lowf);

    pqfeatures(i,:) = feat;
end
toc
pqfeatures_norm = normalize(pqfeatures,2,'norm');

%% 2.3 Holidays
hfeatures = zeros(hfile_num,dim);
tic;
parfor i = 1:hfile_num
    img = imread(strcat(hfiles(i).folder,'\',imlist{i}, '.jpg'));
    img = imresize(img, 0.5);
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    hfeatures(i,:) = feat;
end
toc
hfeatures_norm = normalize(hfeatures,2,'norm');

%% 2.4 Oxford105k dataset

ffeatures = zeros(ffile_num, dim);
tic;
parfor i = 1:ffile_num
    img = imread(strcat(ffiles(i).folder,'\',ffiles(i).name));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    ffeatures(i,:) = feat;        
end
toc
wofeatures = [ofeatures; ffeatures];
wofeatures_norm = normalize(wofeatures,2,'norm');
omfiles = [ofiles; ffiles];

% % 2.5 Paris106k
wpfeatures = [pfeatures; ffeatures];
wpfeatures_norm = normalize(wpfeatures,2,'norm');
pmfiles = [pfiles; ffiles];

%% 2.6 rOxford5k
cfg = configdataset (test_datasets{1}, data_root);
imp = cfg.imlist;
qimp = cfg.qimlist;

rofile_num = numel(imp);
rofeatures = zeros(rofile_num,dim);
tic;
parfor i = 1:rofile_num
    img = imread(strcat(rofiles(i).folder,'\',imp{i},'.jpg'));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    rofeatures(i,:) = feat;
end
toc
rofeatures_norm = normalize(rofeatures,2,'norm');

roqfile_num = numel(qimp);
roqfeatures = zeros(roqfile_num,dim);
tic;
parfor i = 1:roqfile_num
    img = imread(strcat(roqfiles(i).folder,'\',qimp{i},'.jpg'));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    roqfeatures(i,:) = feat;
end
toc
roqfeatures_norm = normalize(roqfeatures,2,"norm");

% % 2.7 rParis6k
cfg = configdataset (test_datasets{2}, data_root);
imp = cfg.imlist;
qimp = cfg.qimlist;

rpfile_num = numel(imp);
rpfeatures = zeros(rpfile_num,dim);
tic;
parfor i = 1:rpfile_num
    img = imread(strcat(rpfiles(i).folder,'\',imp{i},'.jpg'));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    rpfeatures(i,:) = feat;
end
toc
rpfeatures_norm = normalize(rpfeatures,2,'norm');

rpqfile_num = numel(qimp);
rpqfeatures = zeros(rpqfile_num,dim);
tic;
parfor i = 1:rpqfile_num
    img = imread(strcat(rpqfiles(i).folder,'\',qimp{i},'.jpg'));
    img = imgpreprocess(img);
    deepf = activations(net, img, layer);
    lowf = LowF(img);

    feat = uifh(deepf, lowf);

    rpqfeatures(i,:) = feat;
end
toc
rpqfeatures_norm = normalize(rpqfeatures,2,'norm');

%% 3 computeing mAP

ogt_query_filename = oqueryPic(:,1);
pgt_query_filename = pqueryPic(:,1);
qe_num = 0;

if netmodel == "vgg16"
    n = 5;
else
    n = 3;
end

% % 3.1 Oxford5k mAP
tic;
for i = 3:n
    if netmodel == "vgg16"
        dd = 32*2^(i-1);
    else
        dd = dim;
    end
    [oxford_feature_pca,query_feature_pca] = he_pca(pfeatures_norm,ofeatures_norm,qofeatures_norm,dd);
    resultfiles = rank_list(oxford_feature_pca,query_feature_pca,ofiles,ogt_query_filename,rank_list_path,qe_num);
    query_num = size(ogt_query_filename,1);
    sum_ap = 0;
    ap = 0;

    for j = 1:query_num
        resultfile = resultfiles(:,j);
        ap = compute_ap(ogt_query_filename{j},ogt_files,resultfile,rank_list_path);
        sum_ap = sum_ap+ap;
    end
    mAP = sum_ap/query_num;
    fprintf('%s,Oxford5k:dim= %d  mAP= %.4f\n',netmodel,dd,mAP);
end
toc

% % 3.2 Paris6k mAP
tic;
for i = 3:n
    if netmodel == "vgg16"
        dd = 32*2^(i-1);
    else
        dd = dim;
    end
    [paris_feature_pca,pquery_feature_pca] = he_pca(ofeatures_norm,pfeatures_norm,pqfeatures_norm,dd);
    resultfiles = rank_list(paris_feature_pca,pquery_feature_pca,pfiles,pgt_query_filename,rank_list_path,qe_num);
    query_num = size(pgt_query_filename,1);
    sum_ap = 0;
    ap = 0;
    
    parfor j = 1:query_num
        resultfile = resultfiles(:,j);
        ap = compute_ap(pgt_query_filename{j},pgt_files,resultfile,rank_list_path);
        sum_ap = sum_ap + ap;
    end
    mAP = sum_ap/query_num;
    fprintf('%s,Paris6k:dim= %d  mAP= %.4f\n',netmodel,dd,mAP);
end
toc

% % 3.3 Holidays mAP
tic;
for i = 3:n
    if netmodel == "vgg16"
        dd = 32*2^(i-1);
    else
        dd = dim;
    end
    vecs_test = hfeatures_norm';
    qvecs = vecs_test(:,qidx)';
    [hol_feature_pca,hq_feature_pca] = he_pca(wofeatures_norm,hfeatures_norm,qvecs,dd);
    vecs_test = hol_feature_pca';
    qvecs = hq_feature_pca';
    [ranks,sim] = yael_nn(vecs_test, qvecs, size(vecs_test,2), 'L2');
    [map,aps] = compute_map(ranks, gnd_h);
    fprintf('%s,Holidays:dim= %d  mAP= %.4f\n',netmodel,dd,map);
end
toc

%% 3.4 Oxford105k mAP
tic;
for i = 3:n
    if netmodel == "vgg16"
        dd = 32*2^(i-1);
    else
        dd = dim;
    end
    [oxford_feature_pca,query_feature_pca] = he_pca(pfeatures_norm,wofeatures_norm,qofeatures_norm,dd);
    resultfiles = rank_list(oxford_feature_pca,query_feature_pca,omfiles,ogt_query_filename,rank_list_path,qe_num);
    query_num = size(ogt_query_filename,1);
    sum_ap = 0;
    ap = 0;
    parfor j = 1:query_num
        resultfile = resultfiles(:,j);
        ap = compute_ap(ogt_query_filename{j},ogt_files,resultfile,rank_list_path);
        sum_ap = sum_ap+ap;
    end
     mAP = sum_ap/query_num;
     fprintf('%s,Oxford105k:dim= %d  mAP= %.4f\n',netmodel,dd,mAP);
end
toc

% % 3.5 Paris106k mAP
tic;
for i = 3:n
    if netmodel == "vgg16"
        dd = 32*2^(i-1);
    else
        dd = dim;
    end
    [paris_feature_pca,pquery_feature_pca] = he_pca(ofeatures_norm,wpfeatures_norm,pqfeatures_norm,dd);
    resultfiles = rank_list(paris_feature_pca,pquery_feature_pca,pmfiles,pgt_query_filename,rank_list_path,qe_num);
    query_num = size(pgt_query_filename,1);
    sum_ap = 0;
    ap = 0;

    for j = 1:query_num
        resultfile = resultfiles(:,j);
        ap = compute_ap(pgt_query_filename{j},pgt_files,resultfile,rank_list_path);
        sum_ap = sum_ap+ap;
    end
    mAP = sum_ap/query_num;
    fprintf('%s,Paris106k:dim= %d  mAP= %.4f\n',netmodel,dd,mAP);
end
toc

%% 3.6 roxford5k mAP

if netmodel == "vgg16"
    n = 5;
else
    n = 3;
end
d = 1;
cfg = configdataset (test_datasets{d}, char(data_root));
tic;
for j = 3:5
    if netmodel == "vgg16"
        dd = 32*2^(j-1);
    else
        dd = dim;
    end
    [vecsLw,qvecsLw] = he_pca(rpfeatures_norm,rofeatures_norm,roqfeatures_norm,dd);
 
    fprintf('>> %s: Retrieval...\n', test_datasets{d});
    if strcmp(test_datasets{d}, 'roxford5k') || strcmp(test_datasets{d}, 'rparis6k') 
		sim = vecsLw*qvecsLw';
        % sim = rofeatures_norm*roqfeatures_norm';
		[sim, ranks] = sort(sim, 'descend');
		% evaluate ranks
		ks = [1, 5, 10];
		% search for easy (E setup)
		for i = 1:numel(cfg.gnd), gnd(i).ok = [cfg.gnd(i).easy]; gnd(i).junk = [cfg.gnd(i).junk, cfg.gnd(i).hard]; end
		[mapE, apsE, mprE, prsE] = compute_map (ranks, gnd, ks);
		% search for easy & hard (M setup)
		for i = 1:numel(cfg.gnd), gnd(i).ok = [cfg.gnd(i).easy, cfg.gnd(i).hard]; gnd(i).junk = cfg.gnd(i).junk; end
		[mapM, apsM, mprM, prsM] = compute_map (ranks, gnd, ks);
		% search for hard (H setup)
		for i = 1:numel(cfg.gnd), gnd(i).ok = [cfg.gnd(i).hard]; gnd(i).junk = [cfg.gnd(i).junk, cfg.gnd(i).easy]; end
		[mapH, apsH, mprH, prsH] = compute_map (ranks, gnd, ks);
		fprintf('>> %s,%s: mAP %d E: %.4f, M: %.4f, H: %.4f\n', netmodel, test_datasets{d}, dd, mapE, mapM, mapH);
    end
end
toc

% % 3.7 rparis6k mAP
d=2;
cfg = configdataset (test_datasets{d}, char(data_root));
for j = 3:5
    if netmodel == "vgg16"
        dd = 32*2^(j-1);
    else
        dd = dim;
    end
    [vecsLw,qvecsLw] = he_pca(rofeatures_norm,rpfeatures_norm,rpqfeatures_norm,dd);
 
    fprintf('>> %s: Retrieval...\n', test_datasets{d});
    if strcmp(test_datasets{d}, 'roxford5k') || strcmp(test_datasets{d}, 'rparis6k') 
		sim = vecsLw*qvecsLw';
        % sim = rpfeatures_norm*rpqfeatures_norm';
		[sim, ranks] = sort(sim, 'descend');
		% evaluate ranks
		ks = [1, 5, 10];
		% search for easy (E setup)
		for i = 1:numel(cfg.gnd), gnd(i).ok = [cfg.gnd(i).easy]; gnd(i).junk = [cfg.gnd(i).junk, cfg.gnd(i).hard]; end
		[mapE, apsE, mprE, prsE] = compute_map (ranks, gnd, ks);
		% search for easy & hard (M setup)
		for i = 1:numel(cfg.gnd), gnd(i).ok = [cfg.gnd(i).easy, cfg.gnd(i).hard]; gnd(i).junk = cfg.gnd(i).junk; end
		[mapM, apsM, mprM, prsM] = compute_map (ranks, gnd, ks);
		% search for hard (H setup)
		for i = 1:numel(cfg.gnd), gnd(i).ok = [cfg.gnd(i).hard]; gnd(i).junk = [cfg.gnd(i).junk, cfg.gnd(i).easy]; end
		[mapH, apsH, mprH, prsH] = compute_map (ranks, gnd, ks);
		fprintf('>> %s,%s: mAP %d E: %.4f, M: %.4f, H: %.4f\n', netmodel, test_datasets{d}, dd, mapE, mapM, mapH);
    end
end
toc
disp(datetime);
