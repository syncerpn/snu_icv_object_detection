function [samples, scores, alive_idx] = cascade_hog_detect(samples, cascade_classifier, cascade_threshold, block_cell)
% Detect object using HOG features
% samples: N x 1 cell where N is number of windows
% cascade_classifier: pre-trained cascade classifier
% cascade_threshold: associated thresholds (unused now)
% block_cell: A x B number of cells in each block for HOG features (for example: [2 2])
% samples: output positive samples
% scores: scores associated with output samples
% alive_idx: indices of positive samples in the original set of input samples

n_node = size(cascade_classifier,1);
alive_idx = 1:size(samples,1);

for nn = 1:n_node
    n_samples = size(samples,1);
    
    % Extract information from cascade classifier
    svm_beta = cascade_classifier{nn}(1:end-9,:);
    feature_length = size(svm_beta,1);
    svm_bias = cascade_classifier{nn}(end-8,:);
    block = cascade_classifier{nn}(end-7:end-2,:);
    weak_threshold = cascade_classifier{nn}(end-1,:);
    ada_Alpha = cascade_classifier{nn}(end,:);
    threshold = cascade_threshold(nn);
    
    scores = 0;
    
    % May apply parfor here
    for nb = 1:size(block,2)
        region = block(:,nb);
        unrolled_sample = [];
        for ns = 1:n_samples
            unrolled_sample = [unrolled_sample samples{ns}(region(1):region(2),region(3):region(4))];
        end
        X = extractHOGFeatures(unrolled_sample,'CellSize',[region(5) region(6)], 'BlockSize', block_cell, 'BlockOverlap', [0 0]);
        X = reshape(X, feature_length, n_samples);
        X = X';
        scores = scores + ada_Alpha(nb) * (X * svm_beta(:,nb) + svm_bias(:,nb) >= weak_threshold(nb));
    end
    
    % Only positive samples survive
    alive = find(scores >= threshold);
    
    scores = scores(alive);
    samples = samples(alive);
    alive_idx = alive_idx(alive);
    
    if isempty(alive)
        break;
    end
end