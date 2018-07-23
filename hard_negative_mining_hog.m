function expanded_svm_model_compact = hard_negative_mining_hog(svm_model_compact, ...
    hog_window_size, hog_cell_size, hog_block_cell, ...
    prev_training_set, prev_training_label, images_neg_dir, depth, max_neg)
% Train HOG + Linear SVM model by hard negative mining technique
% svm_model_compact: compact (Beta + Bias only) of pre-trained SVM model
% hog_window_size: window size
% hog_cell_size: cell size
% hog_block_cell: cell layout in one block
% prev_training_set: previous training set
% prev_training_label: previous training label
% images_neg_dir: directory of all negative images
% depth: depth of mining (1 is recommended; higher value does not guarantee better result
% max_neg: maximum number of negative samples; memory sensitive
% expanded_svm_model_compact: new compact model

images_neg = dir(images_neg_dir);
images_neg = images_neg(3:end);
n_images_neg = size(images_neg,1);

svm_Beta = svm_model_compact(1:end-1);
feature_length = size(svm_Beta,1);
svm_Bias = svm_model_compact(end);

fprintf('Hard Negative mining:\n');
fprintf('--Negative set: %d images\n', n_images_neg);

X_expanded = prev_training_set;
y_expanded = prev_training_label;

for d = 1:depth
    fprintf('--Depth %d:\n----Processing: ', d);
    for i = 1:n_images_neg
        if (mod(i,50) == 0)
            fprintf('%d ', i);
        end
        im = rgb2gray(imread(strcat(images_neg_dir,images_neg(i).name)));
        [samples, ~] = sliding_window_search(im, hog_window_size, [8 8], inf, 1.2);
        
        n_samples = size(samples, 1);
        
        false_positives = zeros(n_samples, feature_length, 'single');
        
        parfor j = 1:n_samples
            false_positives(j,:) = extractHOGFeatures(samples{j},'CellSize',hog_cell_size, 'BlockSize',hog_block_cell);
        end
        
        scores = false_positives * svm_Beta + svm_Bias;
        labels = scores >= 0;
        
        hard_neg_idx = find(labels == 1);
        X_expanded = [X_expanded; false_positives(hard_neg_idx, :)];
        y_expanded = [y_expanded; zeros(size(hard_neg_idx,1),1)];
        
        if (size(X_expanded,1) > max_neg)
            X_expanded = X_expanded(1:max_neg,:);
            y_expanded = y_expanded(1:max_neg);
            break;
        end
    end
    fprintf('... done\n');
            
    fprintf('----Training with %d examples, including %d new negatives! ... ', size(y_expanded,1), size(y_expanded,1) - size(prev_training_label,1));
    
    svm_model = fitcsvm(X_expanded,y_expanded);
    
    fprintf('done\n');
    svm_Beta = svm_model.Beta;
    svm_Bias = svm_model.Bias;
end

fprintf('--Hard Negative mining finished!\n');
expanded_svm_model_compact = [svm_Beta; svm_Bias];