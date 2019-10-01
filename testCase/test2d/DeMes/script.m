res = 256;

fn = 'DeMes';
im = rgb2gray(im2double(imread([fn '.jpg'])));
im = imresize(im, [res,res]);
im = im / sum(im(:));
exrwritechannels([fn '.exr'],'piz','single','Y',im);
        
