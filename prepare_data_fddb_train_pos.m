function [im_list, n_im] = prepare_data_fddb_train_pos(image_dir, train_set, window_size)
% Create postive training samples from annotations
% image_dir: root directory of all images
% train_set: contains list of images and annotations
% window_size: size of training samples
% im_list: output list of samples
% n_im: number of samples

image_ext = '.jpg';

n_image = size(train_set,1);

im_list = {};
n_im = 0;

for i = 1:n_image
    all_boxes = train_set{i,2};
    all_boxes = [all_boxes all_boxes(:,3)-all_boxes(:,1) all_boxes(:,4)-all_boxes(:,2)];
    
    im = imread(strcat(image_dir, train_set{i,1}, image_ext));
    
    if (size(im,3) == 3)
        im = rgb2gray(im);
    end
    
    [ih, iw] = size(im);
    
    n_boxes_cur = size(all_boxes,1);
    n_im = n_im + n_boxes_cur;
    for j = 1:n_boxes_cur
        xs = all_boxes(j,1);
        ys = all_boxes(j,2);
        
        xe = min(iw,all_boxes(j,3));
        ye = min(ih,all_boxes(j,4));
        
        h = ye - ys + 1;
        w = xe - xs + 1;
        
        if (h > w)
            xs = max(xs - floor((h-w)/2),1);
            xe = min(xs + h - 1, iw);
            xs = xs + (xs + h - 1 - xe);
        elseif (h < w)
            ys = max(ys - floor((w-h)/2),1);
            ye = min(ys + w - 1, ih);
            ys = ys + (ys + w - 1 - ye);
        end
        
        part = im(ys:ye,xs:xe);
        im_list = [im_list; imresize(part, window_size); imresize(part(:,end:-1:1), window_size)];
    end
end