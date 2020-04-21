fn = 'brush';

im = double(exrread([fn '.exr']));
im = (im + 1)/2;
exrwrite(im, [fn '_.exr']);