function img_set = prepare_data_fddb(folds_dir, folds_list, rec_box)
% Create image set (either training or test set) with annotations
% folds_dir: directory of fold information file
% folds_list: list of fold (any combinations of 1 to 10) to be used for this image set
% rec_box: rectangle boxes or ellipses
% img_set: output set of images and annotations

fold_prefix = 'FDDB-fold-';
fold_ellipse = '-ellipseList';
fold_suffix = '.txt';

images_name = {};
images_box = {};

for i = folds_list
    fold_name = sprintf('%s%s%02d%s', folds_dir, fold_prefix, i, fold_suffix);
    fold_file = fopen(fold_name,'r');
    while 1
        tline = fgetl(fold_file);
        if ~ischar(tline), break, end
        images_name = [images_name; tline];
    end
    fclose(fold_file);
    
    fold_ellipse_name = sprintf('%s%s%02d%s%s', folds_dir, fold_prefix, i, fold_ellipse, fold_suffix);
    fold_ellipse_file = fopen(fold_ellipse_name,'r');
    while 1
        image_ellipse = [];
        tline = fgetl(fold_ellipse_file);
        if ~ischar(tline), break, end
        n_faces = str2double(fgetl(fold_ellipse_file));
        for j = 1:n_faces
            image_ellipse = [image_ellipse; sscanf(fgetl(fold_ellipse_file), '%f', [1, 6])];
        end
        if rec_box
            images_box = [images_box; ellipse_to_rect_box(image_ellipse)];
        else
            images_box = [images_box; image_ellipse];
        end
    end
    fclose(fold_ellipse_file);
end

img_set = [images_name images_box];