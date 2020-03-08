# light, albedo(r,g,b), topF0, topRough, flakeF0(r,g,b), flakeRough, flakeNDF, flakeSize, iSigma

# python3 src/demo.py --in_dir in/wood/ --out_dir out/ --forward wood --operation generate --fn 1.png --para_all 1500 0.558 0.279 0.101 -5.674 -11.015 2.614 0.603 2.364 0.448 0.170 0.289 11.924 0.523 0.707 0.209 0.000 0.449 6.338 0.32 0.12 0.007 10


# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/optim/1/ --forward wood --operation optimize --fn 1.png --para_all 999 0.6 0.4 0.1 0 -1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig


# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/hmc/1/ --forward wood --operation sample --fn 1.png --para_all 999 0.6 0.4 0.1 0 -1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.01 0.01 --epochs 20001 --lf_lens 0.01 --lf_steps 4

# 1500.000 0.558 0.279 0.101 -5.674 -11.015 2.614 0.603 2.364 0.448 0.170 0.289 11.924 0.523 0.707 0.209 0.000 0.449 6.338 0.320 0.120 0.007



# python3 src/demo.py --in_dir in/wood/ --out_dir out/ --forward wood --operation generate --fn test.png --para_all 1500.000 0.558 0.279 0.101 -5.674 -11.015 2.614 0.603 2.364 0.448 0.170 0.289 11.924 0.523 0.707 0.209 0.000 0.449 6.338 0.320 0.120 0.007 10 5 0.05 0.2

# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/optim/wood1/ --forward wood --operation optimize --fn wood1.png --para_all 999 0.6 0.4 0.1 0 1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.1 0.1 0.005 20 5 0.05 0.2 --para_eval_idx 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 --err_sigma 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig

# 
# python3 src/demo.py --in_dir out/wood/optim/wood5-3 --out_dir out/ --forward wood --operation generate --fn 2.png --para_all 2000 0.331 0.053 0.015 -0.496 -3.162 0.000 0.648 1.283 0.150 0.364 0.265 18.287 0.275 0.589 1.350 0.558 1.359 16.731 0.33 0.014 0.000 12.921 --size 25 --camera 25 
# 2000 0.245 0.036 0.01 -0.496 -3.162 0.000 0.648 1.283 0.150 0.364 0.265 18.287 0.275 0.589 1.350 0.558 1.359 16.731 0.33 0.014 0.000 12.921

# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/optim/wood5-3/ --forward wood --operation optimize --fn wood5.png --para_all 2000 0.4 0.2 0.1 0 1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 15 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.001 0.01 0.001 0.001 --epochs 5001 --lr 0.01 --to_save fig --camera 25 --size 25

# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/hmc/wood5/ --forward wood --operation sample --fn wood5.png --para_all 2032.053 0.302 0.050 0.016 -4.816 13.257 0.000 0.477 1.265 0.115 0.262 0.076 6.638 0.172 0.589 0.803 0.318 0.588 6.044 0.259 0.058 0.005 5.609 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.001 0.01 0.001 0.001 --epochs 20001 --lf_lens 0.005 --lf_steps 4 --camera 25 --size 25

# aniso
# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/optim/wood1/ --forward wood --operation optimize --fn wood1.png --para_all 300 0.6 0.4 0.1 0 1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.1 0.005 15 0.3 0.3 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --err_sigma 0.01 0.01 --epochs 5001 --lr 0.05 --to_save fig --camera 9 --size 9


# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/optim/prior/ --forward wood --operation optimize --fn 1.png --para_all 999 0.6 0.4 0.1 0 -1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1 --lr 0.01


# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/hmc/test_time/ --forward wood --operation sample --fn 1.png --para_all 999 0.6 0.4 0.1 0 -1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.01 0.01 0.01 0.01 --epochs 101 --lf_lens 0.01 --lf_steps 4


# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/optim/1/ --forward wood --operation optimize --fn 1.png --para_all 2000 0.4 0.2 0.1 0 -1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 15 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig

# python3 src/demo.py --in_dir in/wood/ --out_dir out/wood/optim/wood5/ --forward wood --operation optimize --fn wood5.png --para_all 2000 0.4 0.2 0.1 0 1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 15 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.001 0.01 0.001 0.001 --epochs 2001 --lr 0.01 --to_save fig --camera 25 --size 25


##
python3 src/demo.py --in_dir in/wood/ --out_dir out_egsr/wood/mala/1_3/ --forward wood --operation sample --sampleMethod MALA --fn 1.png --para_all 2000 0.4 0.2 0.1 0 -1 0 0.5 2.5 0.5 0.2 0.2 10 0.5 0.5 0.5 0.3 0.5 5 0.3 0.1 0.005 15 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --err_sigma 0.01 0.01 0.01 0.01 --epochs 100000 --lr 0.001 --diminish 0.1 --to_save fig