function [samples, boxes] = sliding_window_search(image_ori, window, stride, n_scale, scale_step)
% Collect all samples from image by a sliding window
% image_ori: original image
% window: window size
% stride: moving step of sliding window
% n_scale: number of scaling times
% scale_step: down-sampling ratio between current and preceeding images
% samples: output list of all samples
% boxes: output associated positions of all samples

h = size(image_ori,1);
w = size(image_ori,2);

winh = window(1);
winw = window(2);

samples = {};
boxes = [];
scale = 1/scale_step;

n = 1;
while (n <= n_scale)
    scale = scale * scale_step;
    
    image = imresize(image_ori,1/scale);
    
    h_c = floor(h/scale);
    w_c = floor(w/scale);
    
    if ((h_c < winh) || (w_c < winw))
        break;
    end
    
    y = 1:stride:h_c-winh+1;
    if (isempty(y))
        break;
    end
    if (y(end) ~= h_c-winh+1)
        y = [y h_c-winh+1];
    end
    
    x = 1:stride:w_c-winw+1;
    if (isempty(x))
        break;
    end
    if (x(end) ~= w_c-winw+1)
        x = [x w_c-winw+1];
    end
    
    ly = length(y);
    lx = length(x);
    
    currect_scale_samples = cell(ly * lx,1);
    currect_scale_boxes = zeros(ly * lx,4);
    
    for idx = 1:lx*ly
        xidx = mod(idx,lx);
        if ~xidx
            xs = x(end);
        else
            xs = x(xidx);
        end
        xe = xs + winw - 1;
        yidx = ceil(idx/lx);
        ys = y(yidx);
        ye = ys + winh - 1;
        currect_scale_samples{idx} = image(ys:ye, xs:xe);
        currect_scale_boxes(idx,:) = [round(xs*scale) round(ys*scale) round(xe*scale) round(ye*scale)];
    end
	
    samples = [samples; currect_scale_samples];
    boxes = [boxes; currect_scale_boxes];
    n = n+1;
end