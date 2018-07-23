function [cascade_classifier, cascade_threshold] = ...
    train_cascade_classifier_hog(imp_list, imn_list, all_block_pos, block_cell, ...
    pretrained_cascade_classifier, pretrained_cascade_threshold, ...
    images_neg_dir, n_block, max_node, F_TARGET, f_MAX, d_min)
% Train the cascade classifier using HOG features
% imp_list: list of positive samples
% imn_list: list of negative samples
% all_block_pos: position and information of all blocks
% block_cell: cell layout of each block
% pretrained_cascade_classifier: pre-trained model for fine-tuning and growing
% pretrained_cascade_threshold: associated thresholds
% n_block: number of blocks to be evaluated per trial
% max_node: maximum number of nodes
fprintf('Training HOG Cascade Classifier\n')
% Initializing hyper params
D_i = 1;            % Initial detection rate
F_i = 1;            % Initial false positive rate

cascade_classifier = pretrained_cascade_classifier;
cascade_threshold = pretrained_cascade_threshold;

% Support fine-tuning and growing from pre-trained model
if (~isempty(cascade_classifier))
    imn_list_current = cascade_hog_detect(imn_list, cascade_classifier, cascade_threshold, block_cell);
    imp_list_current = cascade_hog_detect(imp_list, cascade_classifier, cascade_threshold, block_cell);
else
    imn_list_current = imn_list;
    imp_list_current = imp_list;
end

i = size(cascade_classifier,1);

wh = size(imp_list{1},1);
ww = size(imp_list{1},2);

while (F_i > F_TARGET || i <= max_node)
    i = i + 1; % Cascade node i
    fprintf('--Cascade Level %d: \n',i);
    
    n_pos = size(imp_list_current,1);
    n_neg = size(imn_list_current,1);
    
    if (~isempty(images_neg_dir))
        resample_count = 0;
        
        while (n_neg < 5000 && resample_count < 100 && ~isempty(cascade_classifier))
            fprintf('----Adding more negative samples ... ');
            resample_neg_list = prepare_training_input(images_neg_dir, [wh ww], 'exhaust', 5);
            resample_neg_list = cascade_hog_detect(resample_neg_list, cascade_classifier, cascade_threshold, block_cell);
            imn_list_current = [imn_list_current; resample_neg_list];
            n_neg = size(imn_list_current,1);
            fprintf('current: %d negative samples\n', n_neg);
            resample_count = resample_count + 1;
        end
        
        if n_neg > 15000
            fprintf('----Too many samples, reduce to 15000\n');
            imn_list_current = imn_list_current(1:15000);
            n_neg = size(imn_list_current,1);
        end
    end
    imx_list = [imp_list_current; imn_list_current];
    n_train = n_pos + n_neg;
    
    % Initializing AdaBoost weights
    ada_w = [ones(n_pos,1) / (2*n_pos);ones(n_neg,1) / (2*n_neg)];
    
    % Initial false positive rate of cascade node i
    f_i = 1;
    
    strong_classifier = [];
    prev_predicted = [];
    
    while (f_i > f_MAX)
        tic
        ada_w = ada_w ./ sum(ada_w); % Normalize weights
        
        random_block_list = randperm(size(all_block_pos,1), n_block);
        selected_pos = all_block_pos(random_block_list,:);
        
        error_best = inf;
        fprintf('----Processing blocks: ')
        for ns = 1:n_block
            if (mod(ns, 25) == 0)
                fprintf('%d ',ns);
            end
            
            region = selected_pos(ns,:);
            
            r_y = region(1):region(2);
            r_x = region(3):region(4);
            r_c = [region(5) region(6)];
                        
            unrolled_sample = [];
            for ni = 1:n_train
                unrolled_sample = [unrolled_sample imx_list{ni}(r_y,r_x)];
            end
            X = extractHOGFeatures(unrolled_sample,'CellSize',r_c, 'BlockSize', block_cell, 'BlockOverlap', [0 0]);
            X = reshape(X, 36, n_train);
            X = X';
            
            y = [ones(n_pos,1);zeros(n_neg,1)];
            
            tmp_model = fitcsvm(X,y);
            
            predicted_score = X * tmp_model.Beta + tmp_model.Bias;
            
            for nt = 1 : n_pos+n_neg
                predicted = predicted_score >= predicted_score(nt);
                match_non_match = abs(predicted - y); % 0 means matched, 1 means non-matched
                error = sum(ada_w .* match_non_match);
                
                if error < error_best
                    error_best = error;
                    best_weak_svm_classifier = [tmp_model.Beta; tmp_model.Bias; region'];
                    match_non_match_best = match_non_match;
                    predicted_best = predicted;
                    best_weak_threshold = predicted_score(nt);
                end
            end
        end
        fprintf('\n');
        
        ada_Beta = error_best / (1 - error_best);
        ada_w = ada_w .* (ada_Beta .^ (1 - match_non_match_best));
        ada_Alpha = -log(ada_Beta);
        
        min_n_pos = ceil(d_min * n_pos);
        
        prev_predicted = [prev_predicted ada_Alpha * predicted_best];
        
        strong_score = sum(prev_predicted,2);
        
        sort_strong_pos_score = sort(strong_score(1:n_pos), 'descend');
        threshold = sort_strong_pos_score(min_n_pos);
        
        strong_predicted = (strong_score >= threshold);
        
        d_i = sum(strong_predicted(1:n_pos)) / n_pos;
        
        strong_classifier = [strong_classifier [best_weak_svm_classifier; best_weak_threshold; ada_Alpha]];
        
        f_i = sum(strong_predicted(n_pos+1:end)) / n_neg;
        
        fprintf('----Threshold: %f -- Detection Rate: %f -- False Positive Rate: %f\n', threshold, d_i, f_i);
        toc
    end
    
    D_i = D_i * d_i;
    F_i = F_i * f_i;
    
    fprintf('--Next Di Fi: %f  %f\n',D_i, F_i);
    
    cascade_threshold = [cascade_threshold threshold];
    cascade_classifier = [cascade_classifier; strong_classifier];
    
    save(sprintf('backup_cascade_hog_%d',i),'i','cascade_classifier','cascade_threshold');
    
    false_positive_list = find(strong_predicted(n_pos+1:end) == 1);
    imn_list_current = imn_list_current(false_positive_list);
    
    true_positive_list = find(strong_predicted(1:n_pos) == 1);
    imp_list_current = imp_list_current(true_positive_list);
end