function test_cascade_classifier_hog(cascade_classifier, cascade_threshold, hog_window_size, window_stride, hog_block_cell, images_test_dir, label_path)
% Test the cascade classifier using HOG features
% cascade_classifier: pre-trained cascade classifier
% cascade_threshold: associated thresholds
% hog_window_size: size of detection window
% hog_block_cell: cell layout of each block
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

fprintf('Testing Cascade HOG + SVM Classifier\n')

for i = 1:n_images_test
    fprintf('Image %d:',i);
    im = imread(strcat(images_test_dir,images_test{i}));
    if (size(im,3) == 3)
        im = rgb2gray(im);
    end
    [samples, boxes] = sliding_window_search(im, hog_window_size, window_stride, inf, 1.2);
    [~, scores, alive_idx] = cascade_hog_detect(samples, cascade_classifier, cascade_threshold, hog_block_cell);
    boxes = boxes(alive_idx,:);
    
    [is_valid_bbox] = non_max_supr_bbox(boxes, scores, size(im), 0);

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

[~, ~, ~, tp, fp, ~] = evaluate_detections(filtered_bboxes_all, confidences_all, image_ids_all, label_path, 1);
visualize_detections_by_image(filtered_bboxes_all, confidences_all, image_ids_all, tp, fp, images_test_dir, label_path);

fprintf('--Done!\n')