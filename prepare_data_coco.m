function [im_list, n_samples] = prepare_data_coco(coco_dir, window_size, n_used_image, n_sample_per_image)
% Create negative samples from COCO dataset
% coco_dir: directory of COCO dataset
% window_size: size of detection window
% n_used_image: number of images to use (randomly select from the dataset)
% n_sample_per_image: number of samples to select from each image
% im_list: output list of images
% n_samples: output number of images

images = dir(coco_dir);
images = images(3:end);
n_images = size(images,1);

wh = window_size(1);
ww = window_size(2);

n_used_image = min(n_used_image, n_images);

idx = randperm(n_images, n_used_image);

n_samples = n_used_image * n_sample_per_image;

im_list = cell(n_samples, 1);

for ii = 1:n_used_image
    i = idx(ii);
	im = imread(strcat(coco_dir,images(i).name));
    h = size(im,1);
    w = size(im,2);
    
    for jj = 1:n_sample_per_image
        y = randi(h - wh + 1);
        x = randi(w - ww + 1);
        sample = im(y : y + wh - 1, x : x + ww - 1, :);
        if size(sample,3) == 3
            sample = rgb2gray(sample);
        end
        im_list{(ii-1)*n_sample_per_image + jj} = sample;
    end
end