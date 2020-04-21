im = double(exrread('flake_normal_flake.exr'));
im = (im + 1)/2;
exrwrite(im, 'flake_normal_flake_.exr');
im = double(exrread('flake_rough_flake.exr'));
im = im.^2;
exrwrite(im, 'flake_rough_flake_.exr');
im = double(exrread('flake_rough_top.exr'));
im = im.^2;
exrwrite(im, 'flake_rough_top_.exr');