function all_block_pos = get_all_var_block(start_scale, window_size, block_cell)

block_list = start_scale:2:window_size(1);
block_list = repmat(block_list,3,1);
block_list = block_list(:);
block_list = repmat(block_list,1,2);
block_list(1:3:end,2) = block_list(1:3:end,2) / 2;
block_list(3:3:end,2) = block_list(3:3:end,2) * 2;

idx = block_list(:,2) >= 12;
block_list = block_list(idx,:);

idx = block_list(:,2) <= window_size(2);
block_list = block_list(idx,:);

idx = mod(block_list(:,2),2) == 0;
block_list = block_list(idx,:);

HOG_params = [];

for i = 1:size(block_list,1)
    if (mod(window_size(1) - block_list(i,1), 4) == 0 && mod(window_size(2) - block_list(i,2), 4) == 0)
        HOG_params = [HOG_params; block_list(i,:)/2 4];
    end
    if (mod(window_size(1) - block_list(i,1), 6) == 0 && mod(window_size(2) - block_list(i,2), 6) == 0)
        HOG_params = [HOG_params; block_list(i,:)/2 6];
    end
end

% Get all position of those blocks
all_block_pos = [];
for f = 1:size(HOG_params,1)
    for ys = 1:HOG_params(f,3):window_size(1) - HOG_params(f,1)*block_cell(1) + 1
        for xs = 1:HOG_params(f,3):window_size(2) - HOG_params(f,2)*block_cell(2) + 1
            all_block_pos = [all_block_pos; ys ys + HOG_params(f,1)*block_cell(1) - 1 xs xs + HOG_params(f,2)*block_cell(2) - 1];
        end
    end
end

all_block_pos = unique(all_block_pos, 'rows');
all_block_pos = [all_block_pos (all_block_pos(:,2)-all_block_pos(:,1)+1)/block_cell(1) (all_block_pos(:,4)-all_block_pos(:,3)+1)/block_cell(2)];