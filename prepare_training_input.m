function [im_list, n_im] = prepare_training_input(images_dir, window_size, mode, n_random)
% Create training samples
% images_dir: directory of all images
% window_size: size of samples
% mode: 'center' ~ crop the center window or 'random' ~ crop randomly
% n_random: number of samples selected from each image
% im_list: output list of samples
% n_im: number of samples

images = dir(images_dir);
images = images(3:end);
n_images = size(images,1);

wh = window_size(1);
ww = window_size(2);

if (strcmp(mode, 'center'))
    n_im = n_images;
    im_list = cell(n_im,1);

    for i = 1:n_images
        im = imread(strcat(images_dir,images(i).name));
        h = size(im,1);
        w = size(im,2);
        ys = floor((h-wh)/2);
        xs = floor((w-ww)/2);
        im = im(ys+1:ys+wh,xs+1:xs+ww,:);
        if size(im,3) == 3
            im = rgb2gray(im);
        end
        im_list{i} = im;
    end
elseif (strcmp(mode, 'random'))
    n_im = n_random * n_images;
    im_list = cell(n_im,1);
    
    for i = 1:n_images
        im = imread(strcat(images_dir,images(i).name));
        h = size(im,1);
        w = size(im,2);
        for j = 1:n_random
            y = randi(h - wh + 1);
            x = randi(w - ww + 1);
            sample = im(y : y + wh - 1, x : x + ww - 1, :);
            if size(sample,3) == 3
                sample = rgb2gray(sample);
            end
            im_list{(i-1)*n_random+j} = sample;
        end
    end
elseif (strcmp(mode, 'exhaust'))
    list = randperm(n_images, n_random);
    im_list = {};
    for i = list
        im = imread(strcat(images_dir,images(i).name));
        if size(im,3) == 3
            im = rgb2gray(im);
        end
        im_list_i = sliding_window_search(im, window_size, [8 8], inf, 1.2);
        im_list = [im_list; im_list_i];
    end
    n_im = size(im_list,1);
end

