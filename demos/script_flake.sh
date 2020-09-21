# light, albedo(r,g,b), topF0, topRough, flakeF0(r,g,b), flakeRough, flakeNDF, flakeSize, iSigma

# generate
# python src/bayesian.py \
# 	--forward flake \
# 	--operation generate \
# 	--size 15 \
# 	--imres 1024 \
# 	--save_tex yes \
# 	--out_dir in/real_estimate/2_flake_real1/ \
# 	--para_all 1194.088 0.129 0.142 0.127 0.032 0.101 0.04 0.04 0.04 0.171 0.30 1 0.000 0.000 11.909
	# --para_all 1782.909 0.774 0.003 0.037 0.042 0.174 0.496 0.044 0.030 0.398 0.380 0.864 0.291 0.649 16.416 # real2

	# --para_all 1200 0.72 0.05 0.61 0.04 0.3 0.6 0.3 0.1 0.16 0.5 0.35 0 0 10 # flake 2
	# --para_all 1194.095 0.722 0.051 0.608 0.044 0.288 0.619 0.295 0.087 0.144 0.470 0.300 0 0 10.640 # flake 2

#999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10

# sample
# python src/bayesian.py \
# 	--in_dir in/4_flake_real2/ \
# 	--out_dir out/4_flake_real2_2/ \
# 	--forward flake \
# 	--operation sample \
# 	--mcmc MALA \
# 	--para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
# 	--epochs 100 \
# 	--lr 0.0005 \
# 	--to_save fig \
# 	--imres 1024 \
# 	--size 15 \
# 	--para_all 1782.909 0.774 0.003 0.037 0.042 0.174 0.496 0.044 0.030 0.398 0.380 0.864 0.291 0.649 16.416

### optim
python src/bayesian.py \
	--forward flake \
	--operation optimize \
	--in_dir in/real/6_wood_real3/ \
	--out_dir out_tmp/6_wood_real3_optim_flake/ \
	--para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
	--epochs 1000 \
	--imres 256 \
	--size 15 \
	--lr 0.05 \
	--save_tex yes

# sample
# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/1/ --forward flake --operation sample --fn 1.png --para_all 720.772 0.089 0.000 0.804 0.045 0.115 0.000 0.512 0.371 0.260 0.497 0.481 0 0 8.381 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/2/ --forward flake --operation sample --fn 2.png --para_all 1288.460 0.621 0.100 0.000 0.035 0.113 0.385 0.187 0.008 0.047 0.274 0.438 0 0 7.102 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/3/ --forward flake --operation sample --fn 3.png --para_all 1194.095 0.722 0.051 0.608 0.044 0.288 0.619 0.295 0.087 0.144 0.470 0.300 0 0 10.640 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/4/ --forward flake --operation sample --fn 4.png --para_all 1092.295 0.032 0.960 0.663 0.035 0.145 0.247 0.141 0.108 0.180 0.384 0.571 0 0 10.652 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/5/ --forward flake --operation sample --fn 5.png --para_all 1077.085 0.542 0.762 0.080 0.041 0.300 0.455 0.407 0.087 0.028 0.451 0.398 0 0 5.772 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 5001 --lf_lens 0.01 --lf_steps 4

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/6/ --forward flake --operation sample --fn 6.png --para_all 1335.586 0.547 0.192 0.108 0.042 0.246 0.648 0.448 0.109 0.221 0.409 0.536 0 0 7.271 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 5001 --lf_lens 0.01 --lf_steps 4


# optimize
# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/0/ --forward flake --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 2001

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/2/ --forward flake --operation optimize --fn 2.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 2001

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/3/ --forward flake --operation optimize --fn 3.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 2001

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/4/ --forward flake --operation optimize --fn 4.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 2001

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/5/ --forward flake --operation optimize --fn 5.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 2001

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/6/ --forward flake --operation optimize --fn 6.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 2001



# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/1/ --forward flake --operation sample --fn 1.png --para_all 720.772 0.089 0.000 0.804 0.045 0.115 0.000 0.512 0.371 0.260 0.497 0.481 0 0 8.381 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 14 --epochs 5001 --lf_lens 0.1 --lf_steps 4

#############################
# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/1/ --forward flake --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --err_sigma 0.001 0.001 --epochs 20001 --lr 0.001

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/1/ --forward flake --operation sample --fn 1.png --para_all 720.772 0.089 0.000 0.804 0.045 0.115 0.000 0.512 0.371 0.260 0.497 0.481 0 0 8.381 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 14 --err_sigma 0.001 0.01 --epochs 20001 --lf_lens 0.05 --lf_steps 4

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/realflake2/ --forward flake --operation optimize --fn flake.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --err_sigma 0.001 0.001 --epochs 5001 --lr 0.05 --to_save fig --size 10.6



###### test time 

# optimize
# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/prior/ --forward flake --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.04 0.15 0.4 0.4 0.4 0.15 0.3 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 1 --err_sigma 0.001 0.001 0.001 0.001

# sample
# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/hmc/test_time/ --forward flake --operation sample --fn 0.png --para_all 720.772 0.089 0.000 0.804 0.045 0.115 0.000 0.512 0.371 0.260 0.497 0.481 0 0 8.381 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 101 --lf_lens 0.01 --lf_steps 4  --err_sigma 0.01 0.01 0.01 0.01

# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/1/ --forward flake --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.04 0.1 0.4 0.4 0.4 0.1 0.3 0.5 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --err_sigma 0.001 0.001 --epochs 2001 --lr 0.01 --to_save fig


# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/optim/real/ --forward flake --operation optimize --fn flake.png --para_all 999 0.04 0.04 0.04 0.04 0.1 0.1 0.1 0.1 0.2 0.2 0.9 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --err_sigma 0.001 0.01 0.001 0.01 --epochs 1001 --lr 0.01 --to_save fig --sum_func Bins


# python3 src/demo.py --in_dir in/flake/ --out_dir out/flake/sample/real/ --forward flake --operation sample --fn flake.png --para_all 1313.787 0.094 0.116 0.106 0.032 0.100 0.119 0.128 0.119 0.488 0.323 0.788 0.047 -0.043 13.237 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 14 --err_sigma 0.001 0.01 0.01 0.05 --epochs 20001 --lf_lens 0.01 --lf_steps 8 --sum_func Bins