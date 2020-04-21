#  light, f0, roughx, roughy, fsigmax, fsigmay, fscale, iSigma
python src/bayesian.py \
	--forward metal \
	--operation generate \
	--out_dir in/real_estimate/5_metal_real1/ \
	--size 10 \
	--imres 1024 \
	--save_tex yes \
	--para_all 1382.704 0.285 0.285 0.285 0.204 0.438 0.038 14.532 0.036 13.343
	# --para_all 999 0.8 0.8 0.4 0.1 0.5 0.05 5 0.05 10 \

# optimize
# python src/bayesian.py \
# 	--forward metal \
# 	--operation optimize \
# 	--in_dir in/5_metal_real1/ \
# 	--out_dir out/5_metal_real1_optim1/ \
# 	--para_eval_idx 0 1 2 3 4 5 6 7 8 9 \
# 	--epochs 1000 \
# 	--lr 0.01 \
# 	--size 10 \
# 	--imres 512 \
# 	--sum_func Grids


# python src/bayesian.py \
# 	--forward metal \
# 	--operation sample \
# 	--in_dir in/5_metal_real1/ \
# 	--out_dir out/5_metal_real1/ \
# 	--para_eval_idx 0 1 2 3 4 5 6 7 8 9 \
# 	--epochs 100 \
# 	--lr 0.002 \
# 	--size 10 \
# 	--imres 512 \
# 	--sum_func Grids \
# 	--mcmc MALA \
# 	--para_all 1382.704 0.085 0.085 0.085 0.204 0.438 0.038 14.532 0.036 13.343


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
# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/test_time/ --forward metal --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.1 0.5 0.5 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1 --lr 0.001

# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/0/ --forward metal --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.1 0.4 0.05 10 0.01 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig

# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/optim/real/ --forward metal --operation optimize --fn metal_new.png --para_all 999 0.4 0.4 0.4 0.2 0.2 0.01 20 0.01 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.01 0.01 0.01 0.1 --epochs 501 --lr 0.08 --size 12 --to_save fig --sum_func Grids


# python3 src/demo.py --in_dir in/metal/ --out_dir out/metal/sample/real/ --forward metal --operation sample --fn metal_new.png --para_all 597.889 0.227 0.224 0.223 0.175 0.621 0.053 18.908 0.049 18.660 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 --err_sigma 0.01 0.01 0.01 0.1 --epochs 20001 --lf_lens 0.02 --lf_steps 8 --size 12 --sum_func Grids

