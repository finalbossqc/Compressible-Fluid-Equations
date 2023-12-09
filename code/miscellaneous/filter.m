%{
Description:
getRGBarrs stores all the average r, g, and b values in three separate 
arrays and an additional array  of each sub-image and
the image names in four separate arrays. At each square patch of the
photomosaic, the average rgb value is calculated. It then finds the images
with the most similar rgb values within a certain threshold. Finally, it
will randomly select the sub-image. Increasing the threshold will further
distort the image because the average rgb will most likely not be as close
as it would with a lower threshold.
%}

close('all');

images_dir = 'images\';
fluid_im = 'https://comtessablog.files.wordpress.com/2018/04/ncview_s1.png';

[fluid_im, fluid_fig] = get_fig(fluid_im, 40);
figure
distorted = imshow(fluid_fig);
saveas(distorted, 'fluid.png');

function [original, fig] = get_fig(im, divisor)
    % returns photomosaic image
    original = imread(im);
    fig = low_pass_filter(original, divisor);
end

function res_im = low_pass_filter(main_im, divisor)
    [h, w, ~] = size(main_im);
    res_im = zeros(size(main_im), 'like', main_im);
    increment = [ceil(h/divisor) ceil(w/divisor)];
    
    for i = 1:increment(1):h
        for j = 1:increment(2):w
            patch = main_im(i:min(i+increment(1)-1,h), j:min(j+increment(2)-1,w), :);
            avg_rgb = mean(mean(patch, 1), 2);
            res_im(i:min(i+increment(1)-1,h), j:min(j+increment(2)-1,w), :) = repmat(avg_rgb, [size(patch, 1), size(patch, 2), 1]);
        end
    end
end