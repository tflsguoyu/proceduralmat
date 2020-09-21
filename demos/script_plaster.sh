#  light, albedo, rough, rough_var, height, slope, scale, iSigma

# generate
# python src/bayesian.py \
# 	--forward plaster \
# 	--operation generate \
# 	--out_dir in/2_plaster_syn2_5/ \
# 	--para_all 2000.000 0.655 0.667 0.828 0.394 0.068 0.134 2.128 0.253 0.222 -0.274 15.219 2 # plaster 2
	# --para_all 2000.000 0.65 0.66 0.82 0.39 0.06 0.13 2.12 0.25 0.22 -0.27 15 # plaster 2
	# --para_all 2000 0.65 0.65 0.8 0.4 0.08 0.2 2 0.4 0 0 15 3 # plaster 2
	# --para_all 2000.000 0.655 0.667 0.828 0.394 0.068 0.134 2.128 0.253 0.222 -0.274 15.219 # plaster 2


# sample
# python src/bayesian.py \
# 	--in_dir in/real/6_wood_real3/ \
# 	--out_dir out_tmp/6_wood_real3_optim_plaster/ \
# 	--forward plaster \
# 	--operation sample \
# 	--mcmc MALA \
# 	--para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
# 	--epochs 100 \
# 	--lr 0.005 \
# 	--to_save fig \
# 	--imres 256 \
# 	--para_all 1948.993 0.824 0.760 0.774 0.79 0.208 0.308 2.142 0.888 0.007 -0.038 17.460

## optim
python src/bayesian.py \
	--forward plaster \
	--operation optimize \
	--in_dir in/real/6_wood_real3/ \
	--out_dir out_tmp/6_wood_real3_optim_plaster/ \
	--para_eval_idx 0 1 2 3 5 6 7 8 9 10 11 \
	--epochs 1001 \
	--imres 256 \
	--size 5 \
	--lr 0.01 \
	--save_tex yes

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/2/ --forward plaster --operation sample --fn 2.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/3/ --forward plaster --operation sample --fn 3.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/4/ --forward plaster --operation sample --fn 4.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/5/ --forward plaster --operation sample --fn 5.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/6/ --forward plaster --operation sample --fn 6.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001 --lf_lens 0.01 --lf_steps 4



# optimize
# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/1/ --forward plaster --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 1001

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/2/ --forward plaster --operation optimize --fn 2.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 1001

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/3/ --forward plaster --operation optimize --fn 3.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 1001

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/4/ --forward plaster --operation optimize --fn 4.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 1001

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/5/ --forward plaster --operation optimize --fn 5.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 1001

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/6/ --forward plaster --operation optimize --fn 6.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 1001


##########################
# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/1/ --forward plaster --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 --epochs 20001 --lr 0.001

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/1/ --forward plaster --operation sample --fn 1.png --para_all 2000.000 0.612 0.155 0.506 0.411 0.316 0.171 2.559 0.418 0 0 10.667 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --err_sigma 0.001 0.01 --epochs 20001 --lf_lens 0.1 --lf_steps 4



# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/green-wall/ --forward plaster --operation sample --fn green-wall.png --para_all 1674.206 0.080 0.090 0.043 0.597 0.377 0.039 2.190 0.500 0 0 9.018 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --err_sigma 0.001 0.001 --epochs 20001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/green-wall2/ --forward plaster --operation optimize --fn green-wall.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 --epochs 5001 --lr 0.05 --to_save fig


###### test time

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/prior/ --forward plaster --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1 --lr 0.001

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/hmc/test_time/ --forward plaster --operation sample --fn 1.png --para_all 2000.000 0.612 0.155 0.506 0.411 0.316 0.171 2.559 0.418 0 0 10.667 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --err_sigma 0.001 0.01 0.001 0.001 --epochs 101 --lf_lens 0.1 --lf_steps 4


# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/1/ --forward plaster --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig

# python3 src/demo.py --in_dir in/plaster --out_dir out/plaster/optim/green-wall/ --forward plaster --operation optimize --fn green-wall.png --para_all 999 0.4 0.4 0.4 0.4 0.2 0.2 3 0.1 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig
