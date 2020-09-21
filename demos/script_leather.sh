#  light, albedo, rough, rough_var, height, power, scale, shiftx, shifty, iSigma

# generate
# python src/bayesian.py \
# 	--forward leather \
# 	--operation generate \
# 	--out_dir in/real_estimate/2_leather_real1/ \
# 	--imres 1024\
# 	--size 10 \
# 	--save_tex yes \
# 	--para_all 1705.436 0.028 0.024 0.035 0.257 0.088 0.888 0.010 0.632 0.018 0.120 0.282 5.893 # leather real 3

# sample
# python src/bayesian.py \
# 	--forward leather \
# 	--operation sample \
# 	--in_dir in/real/2_leather_real3/ \
# 	--out_dir out/2_leather_real3_time/ \
# 	--mcmc MALA \
# 	--para_eval_idx 0 1 2 3 4 5 6 7 9 12 \
# 	--epochs 100 \
# 	--lr 0.001 \
# 	--to_save fig \
# 	--imres 512 \
# 	--size 10 \
#     --para_all 1530.617 0.723 0.305 0.009 0.281 0.096 0.283 0.059 0.42 0.024 -0.029 0.020 15.571
	# --para_all 1489.686 0.319 0.490 0.406 0.464 0.194 0.595 0.053 0.844 0.034 -0.054 0.088 7.756 # leather 4
	# --para_all 1750.453 0.245 0.003 0.002 0.264 0.024 0.038 0.305 0.933 0.009 -0.097 -0.373 14.202 # leather 2
	# --para_all 1552.595 0.697 0.304 0.003 0.317 0.063 0.098 0.451 0.420 0.035 -0.300 -0.135 15.573 # leather 3
# optimize
python src/bayesian.py \
	--forward leather \
	--operation optimize \
	--in_dir in/real/6_wood_real3/ \
	--out_dir out_tmp/6_wood_real3_optim_leather/ \
	--para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 \
	--epochs 1000 \
	--imres 256 \
	--size 5 \
	--lr 0.05 \
	--save_tex yes

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/2/ --forward leather --operation sample --fn 2.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001  --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/3/ --forward leather --operation sample --fn 3.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001  --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/4/ --forward leather --operation sample --fn 4.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001  --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/5/ --forward leather --operation sample --fn 5.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001  --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/6/ --forward leather --operation sample --fn 6.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001  --lf_lens 0.01 --lf_steps 4


# optimize
# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/1/ --forward leather --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 501

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/2/ --forward leather --operation optimize --fn 2.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 501

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/3/ --forward leather --operation optimize --fn 3.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 501

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/4/ --forward leather --operation optimize --fn 4.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 501

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/5/ --forward leather --operation optimize --fn 5.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 501

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/6/ --forward leather --operation optimize --fn 6.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 501

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/1/ --forward leather --operation sample --fn 1.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --epochs 5001  --lf_lens 0.1 --lf_steps 4


####################################
# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/3/ --forward leather --operation optimize --fn 3.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 --epochs 20001 --lr 0.001

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/3/ --forward leather --operation sample --fn 3.png --para_all 901.948 0.305 0.523 0.274 0.241 0.203 0.184 2.145 0.48 0 0 10.382 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --err_sigma 0.001 0.01 --epochs 20001  --lf_lens 0.1 --lf_steps 4

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/leather_black/ --forward leather --operation optimize --fn leather_black.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --size 10 --epochs 5001 --lr 0.05 --to_save fig

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/leather_black/ --forward leather --operation sample --fn leather_black.png --para_all 609.647 0.003 0.002 0.006 0.187 0.124 0.004 1.945 0.723 -0.735 -0.710 5.263 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --err_sigma 0.001 0.01 0.001 0.001 --epochs 20001  --lf_lens 0.005 --lf_steps 4 --size 10



######## test time 

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/prior/ --forward leather --operation optimize --fn leather_black.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --size 10 --epochs 101 --lr 0.05

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/test_time/ --forward leather --operation sample --fn leather_black.png --para_all 609.647 0.003 0.002 0.006 0.187 0.124 0.004 1.945 0.723 -0.735 -0.710 5.263 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --err_sigma 0.001 0.01 0.001 0.001 --epochs 101 --lf_lens 0.005 --lf_steps 4 --size 10


# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/3/ --forward leather --operation optimize --fn 3.png --para_all 999 0.4 0.4 0.4 0.2 0.2 0.2 1.5 0.2 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --epochs 501 --lr 0.01  --to_save fig

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/leather_black/ --forward leather --operation optimize --fn leather_black.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --size 10 --epochs 2001 --lr 0.05 --to_save fig