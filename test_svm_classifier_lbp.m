function test_svm_classifier_lbp(svm_model_compact, lbp_window_size, lbp_stride, lbp_cell_size, images_test_dir, label_path)
% Test the HOG + Linear SVM model
% svm_model_compact: pre-trained SVM model
% lbp_window_size: size of detection window
% lbp_cell_size: HOG cell size
% images_test_dir: directory of all test images
% label_path: path of ground-truth labels

fid = fopen(label_path);
gt_info = textscan(fid, '%s %d %d %d %d');
fclose(fid);
gt_ids = gt_info{1,1};
images_test = unique(gt_ids);
n_images_test = size(images_test,1);

filtered_bboxes_all = [];
image_ids_all = {};
confidences_all = [];

svm_beta = svm_model_compact(1:end-1);
feature_length = size(svm_beta,1);
svm_bias = svm_model_compact(end);

fprintf('Testing LBP + Linear SVM\n')

for i = 1:n_images_test
    fprintf('--Image %d:',i);
    im = imread(strcat(images_test_dir,images_test{i}));
    if (size(im,3) == 3)
        im = rgb2gray(im);
    end
    [imh, imw] = size(im);
    [samples, boxes] = sliding_window_search(im, lbp_window_size, lbp_stride, 10, 1.2);
    clear im; % Save memory
    n_samples = size(samples, 1);
    X_test = zeros(n_samples,feature_length, 'single');

    parfor j = 1:n_samples
        X_test(j,:) = extractLBPFeatures(samples{j}, 'CellSize', lbp_cell_size);
    end
    
    scores = X_test * svm_beta + svm_bias;
    clear X_test; % Save memory
    labels = scores >= 0;
    det_boxes_idx = find(labels);
    scores = scores(det_boxes_idx);
    boxes = boxes(det_boxes_idx,:);
    [is_valid_bbox] = non_max_supr_bbox(boxes, scores, [imh, imw], 0);
    valid = find(is_valid_bbox);
    filtered_bboxes = boxes(valid,:);
    scores = scores(valid,:);
    n_det = size(filtered_bboxes,1);

    image_ids = cell(n_det,1);
    [image_ids{:}] = deal(images_test{i});
    image_ids_all = [image_ids_all; image_ids];
    confidences_all = [confidences_all; scores];
    filtered_bboxes_all = [filtered_bboxes_all; filtered_bboxes];
end

save('test_result.mat','filtered_bboxes_all','confidences_all','image_ids_all');

[~, ~, ~, tp, fp, ~] = evaluate_detections(filtered_bboxes_all, confidences_all, image_ids_all, label_path, 1);
visualize_detections_by_image(filtered_bboxes_all, confidences_all, image_ids_all, tp, fp, images_test_dir, label_path);

fprintf('--Done!\n')