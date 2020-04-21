fn = 'metal';
im = double(exrread([fn '_normal.exr']));
im = (im + 1)/2;
exrwrite(im, [fn '_normal_.exr']);
im = double(exrread([fn '_roughy.exr']));
im = im.^2;
exrwrite(im, [fn '_roughy_.exr']);