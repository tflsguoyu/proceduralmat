res = 256;

fn = 'heightmap';
im = rgb2gray(im2double(imread([fn '.jpg'])));
im = imresize(im, [res,res]);
im = im / sum(im(:));
exrwritechannels([fn '.exr'],'piz','single','Y',im);
        
