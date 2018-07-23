%%               HUMAN AND FACE DETECTION
% |===================================================|
% | Nguyen Tuan Nghia                                 |
% | nghianguyentuan@snu.ac.kr                         |
% | Seoul National University                         |
% | Department of Electrical and Computer Engineering |
% |===================================================|
%
%% HUMAN DETECTION
% SETTING UP:
close all;
clc;

% ==============================================
% |         HUMAN DETECTION SETTINGS           |
% ==============================================
% Training image directories
images_pos_dir = '../../datasets/INRIAPerson/train_64x128_H96/pos/';
images_neg_dir = '../../datasets/INRIAPerson/Train/neg/';

% Test image directory
images_test_dir = '../../datasets/INRIAPerson/Test/pos/';

% Ground-truth labels
human_label_path = 'human_gt.txt';

% Detection window size
human_window_size = [128 64];
human_window_stride = [8 8];

% HOG Settings
human_hog_cell_size = [8 8];
human_hog_block_cell = [2 2];

% LBP Settings
human_lbp_cell_size = [16 16];
% ==============================================

if (MODE == 2)
    % Load the pre-trained model
    load('model.mat');
end

% TRAINING
if (MODE == 1)
    % Create list of positive and negative images for common use
    [imp_list, n_imp] = prepare_training_input(images_pos_dir, human_window_size, 'center', []);
    [imn_list, n_imn] = prepare_training_input(images_neg_dir, human_window_size, 'random', 10);
    
    %======================================================== HOG + LINEAR SVM ==============================================================
    imx_list = [imp_list; imn_list];
    
    % Ground-truth for training
    % 1 ~ positive, 0 ~ negative
    y = [ones(n_imp,1); zeros(n_imn,1)];
    
    % Try to measure feature length on a toy sample
    unrolled_hog_feature_length = length(extractHOGFeatures(ones(human_window_size), 'CellSize', human_hog_cell_size, 'BlockSize', human_hog_block_cell));
    
    % Initialize training data
    X = zeros(n_imp + n_imn, unrolled_hog_feature_length, 'single');
    
    % Extract HOG features
    % Parallel computing: 2x faster
    parfor i = 1:n_imp + n_imn
        X(i,:) = extractHOGFeatures(imx_list{i}, 'CellSize', human_hog_cell_size, 'BlockSize', human_hog_block_cell);
    end
    
    % Train Linear SVM classifier
    fprintf('Training SVM Classifier with HOG ... ');
    svm_model = fitcsvm(X,y);
    fprintf('done\n');
    
    % Trained model contains a lot of redundant things
    % We only keep the main components: Beta and Bias
    % Score = X * Beta + Bias and Threshold = 0
    svm_human_hog = [svm_model.Beta; svm_model.Bias];
    %========================================================================================================================================
    
    %======================================================== MINING HARD NEGATIVE ==========================================================
    expanded_svm_human_hog = hard_negative_mining_hog(svm_human_hog, human_window_size, human_hog_cell_size, human_hog_block_cell, X, y, images_neg_dir, 1, 30000);
    %========================================================================================================================================
    
    %========================================================== CASCADE DETECTOR ============================================================
    % Get list of all blocks
    all_block_pos_human = get_all_var_block(12, human_window_size, human_hog_block_cell);
    
    F_TARGET_human = 0.001;    % Target false positive rate of final cascade classifier
    f_MAX_human = 0.7;        % False positive rate per weak classifier
    d_min_human = 0.9975;     % Minimum acceptable detection rate per cascade node
    n_block_human = 250;
    max_node_human = 40;
    
    % Train cascade classifier for human detection
    [cascade_classifier_human_hog, cascade_threshold_human_hog] = train_cascade_classifier_hog(imp_list, imn_list, all_block_pos_human, ...
        human_hog_block_cell, {}, [], images_neg_dir, n_block_human, max_node_human, F_TARGET_human, f_MAX_human, d_min_human);
    %========================================================================================================================================
    
    %========================================================== LBP + LINEAR SVM ============================================================
    unrolled_lbp_feature_length = length(extractLBPFeatures(ones(human_window_size), 'CellSize', human_lbp_cell_size));
    
    % Initialize training data
    % Use the same image list for training HOG + SVM model to train LBP + SVM one
    X = zeros(n_imp + n_imn, unrolled_lbp_feature_length);
    parfor i = 1:n_imp + n_imn % PARALLEL: 2x faster
        X(i,:) = extractLBPFeatures(imx_list{i},'CellSize',human_lbp_cell_size);
    end
    
    fprintf('Training SVM Classifier with LBP ... ');
    svm_model = fitcsvm(X,y);
    fprintf('done\n');
    svm_human_lbp = [svm_model.Beta; svm_model.Bias];
    %========================================================================================================================================
end

% TESTING

if (MODE == 2)
    % HOG + LINEAR SVM MODEL: PLEASE UNCOMMENT TO TEST
%     test_svm_classifier_hog(svm_human_hog, human_window_size, human_window_stride, human_hog_cell_size, human_hog_block_cell, images_test_dir, human_label_path)
    
    % EXPANDED HOG + LINEAR SVM MODEL: PLEASE UNCOMMENT TO TEST
%     test_svm_classifier_hog(expanded_svm_human_hog, human_window_size, human_window_stride, human_hog_cell_size, human_hog_block_cell, images_test_dir, human_label_path)
    
    % HOG CASCADE CLASSIFIER: BEST
    test_cascade_classifier_hog(cascade_classifier_human_hog, cascade_threshold_human_hog, human_window_size, human_window_stride, human_hog_block_cell, images_test_dir, human_label_path);
    
    % LBP + LINEAR SVM MODEL: PLEASE UNCOMMENT TO TEST
%     test_svm_classifier_lbp(svm_human_lbp, human_window_size, human_window_stride, human_lbp_cell_size, images_test_dir, human_label_path)
end

%% FACE DETECTION
% SETTING UP

% ==============================================
% |          FACE DETECTION SETTINGS           |
% ==============================================
% FDDB Dataset
images_fddb_dir = '../../datasets/FDDB/';
folds_dir = '../../datasets/FDDB/FDDB-folds/';

% Ground-truth labels
face_label_path = 'face_gt.txt';

% Detection window settings
face_window_size = [64 64];
face_window_stride = [4 4];

% HOG Settings
face_hog_cell_size = [8 8];
face_hog_block_cell = [2 2];

% LBP Settings
face_lbp_cell_size = [16 16];
% ==============================================

if (MODE == 2)
    % Load the pre-trained model
    load('model.mat');
end

% TRAINING
if (MODE == 1)
    %======================================================== HOG + LINEAR SVM ==============================================================
    % Prepare positive samples
    train_set = prepare_data_fddb(folds_dir, 1:8, 1);
    [imp_list, n_imp] = prepare_data_fddb_train_pos(images_fddb_dir, train_set, face_window_size);
    
    % COCO Dataset
    % Since the images in COCO Dataset are pretty large,
    % it is better that we sample only a part of the set,
    % and get multiple samples from each of them
    images_coco_dir = '../../datasets/COCO/train2014/';
    [imn_list, n_imn] = prepare_data_coco(images_coco_dir, face_window_size, 4000, 5);
    
    % Training data
    imx_list = [imp_list; imn_list];
    y = [ones(n_imp,1); zeros(n_imn,1)];
    unrolled_hog_feature_length = length(extractHOGFeatures(ones(face_window_size), 'CellSize', face_hog_cell_size, 'BlockSize', face_hog_block_cell));
    X = zeros(n_imp + n_imn, unrolled_hog_feature_length, 'single');
    
    parfor i = 1:n_imp + n_imn
        X(i,:) = extractHOGFeatures(imx_list{i}, 'CellSize', face_hog_cell_size, 'BlockSize', face_hog_block_cell);
    end
    
    fprintf('Training SVM Classifier with HOG ... ');
    svm_model = fitcsvm(X,y);
    fprintf('done\n');
    svm_face_hog = [svm_model.Beta; svm_model.Bias];
    %========================================================================================================================================
    
    %======================================================== MINING HARD NEGATIVE ==========================================================
    expanded_svm_face_hog = hard_negative_mining_hog(svm_face_hog, face_window_size, face_hog_cell_size, face_hog_block_cell, X, y, images_coco_dir, 1, 30000);
    %========================================================================================================================================
    
    %========================================================== CASCADE DETECTOR ============================================================
    
    all_block_pos_face = get_all_var_block(12, face_window_size, face_hog_block_cell);
    
    F_TARGET_face = 0.00001;    % Target false positive rate of final cascade classifier
    f_MAX_face = 0.7;        % False positive rate per weak classifier
    d_min_face = 0.9975;     % Minimum acceptable detection rate per cascade node
    n_block_face = 100;
    max_node_face = 40;
    
    [cascade_classifier_face_hog, cascade_threshold_face_hog] = train_cascade_classifier_hog(imp_list, imn_list, all_block_pos_face, ...
        face_hog_block_cell, {}, [], images_coco_dir, n_block_face, max_node_face, F_TARGET_face, f_MAX_face, d_min_face);
    %========================================================================================================================================
    
    %========================================================== LBP + LINEAR SVM ============================================================
    unrolled_lbp_feature_length = length(extractLBPFeatures(ones(face_window_size), 'CellSize', face_lbp_cell_size));
    X = zeros(n_imp + n_imn, unrolled_lbp_feature_length);
    
    parfor i = 1:n_imp + n_imn
        X(i,:) = extractLBPFeatures(imx_list{i},'CellSize',face_lbp_cell_size);
    end
    
    fprintf('Training SVM Classifier with LBP ... ');
    svm_model = fitcsvm(X,y);
    fprintf('done\n');
    svm_face_lbp = [svm_model.Beta; svm_model.Bias];
    %========================================================================================================================================
end

% TESTING
if (MODE == 2)
    % HOG + LINEAR SVM MODEL: PLEASE UNCOMMENT TO TEST
%     test_svm_classifier_hog(svm_face_hog, face_window_size, face_window_stride, face_hog_cell_size, face_hog_block_cell, images_fddb_dir, face_label_path)
    
    % EXPANDED HOG + LINEAR SVM MODEL: BEST
    test_svm_classifier_hog(expanded_svm_face_hog, face_window_size, face_window_stride, face_hog_cell_size, face_hog_block_cell, images_fddb_dir, face_label_path)
    
    % HOG CASCADE CLASSIFIER: PLEASE UNCOMMENT TO TEST
%     test_cascade_classifier_hog(cascade_classifier_face_hog, cascade_threshold_face_hog, face_window_size, face_window_stride, face_hog_block_cell, images_fddb_dir, face_label_path);
    
    % LBP + LINEAR SVM MODEL: PLEASE UNCOMMENT TO TEST
%     test_svm_classifier_lbp(svm_face_lbp, face_window_size, face_window_stride, face_lbp_cell_size, images_fddb_dir, face_label_path)
end