#  lgiht, f0, roughx, roughy, fsigmax, fsigmay, fscale, iSigma
# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 0.png

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 1.png

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 2.png 

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 3.png 

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 4.png 

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 5.png  

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 6.png 

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 7.png 

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 8.png

# python3 src/demo.py --in_dir in/metal/ --out_dir out/ --forward metal --operation generate --fn 9.png 


# sample
# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/hmc/0/ --forward metal --operation sample --fn 0.png --para_all 999 0.4 0.4 0.4 0.1 0.5 0.5 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 --epochs 20001 --lf_lens 0.1 --lf_steps 4

# optimize
# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/0/ --forward metal --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.1 0.5 0.5 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 --epochs 20001 --lr 0.001


# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/real3/ --forward metal --operation optimize --fn metal.png --para_all 999 0.4 0.4 0.4 0.05 0.5 0.05 10 0.01 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 --epochs 2001 --lr 0.01 --size 7 --to_save fig


############ test time

# sample
# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/hmc/test_time/ --forward metal --operation sample --fn 0.png --para_all 999 0.4 0.4 0.4 0.1 0.5 0.5 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 0.001 0.001 --epochs 101 --lf_lens 0.01 --lf_steps 4

# # optimize
python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/test_time/ --forward metal --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.1 0.5 0.5 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1 --lr 0.001

# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/0/ --forward metal --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.1 0.4 0.05 10 0.01 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig

# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/real/ --forward metal --operation optimize --fn metal_new.png --para_all 999 0.4 0.4 0.4 0.2 0.2 0.01 20 0.01 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.01 0.01 0.01 0.1 --epochs 501 --lr 0.08 --size 12 --to_save fig --sum_func Grids


# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/sample/real/ --forward metal --operation sample --fn metal_new.png --para_all 597.889 0.227 0.224 0.223 0.175 0.621 0.053 18.908 0.049 18.660 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.01 0.01 0.01 0.1 --epochs 20001 --lf_lens 0.02 --lf_steps 8 --size 12 --sum_func Grids

