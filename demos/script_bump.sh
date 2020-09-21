# light, albedo(r,g,b), rough, fsigma, fscale, iSigma

## generate
# python src/bayesian.py \
# 	--forward bump \
# 	--operation generate \
# 	--out_dir in/1_bump_syn0/ \
# 	--imres 256 \
# 	--save_tex yes \
# 	--para_all 1000 0.1 0.1 0.9 0.2 2 0.1 15 # bump 0
# 	# --para_all 1887.856 0.643 0.612 0.567 0.356 1.798 0.060 17.305
# 	# --para_all 2024.807 0.617 0.139 0.159 0.310 3.262 0.025 6.297
# 	# --para_all 1355.653 0.000 0.239 0.109 0.406 1.137 0.149 9.862 # bump 2
# 	# --para_all 1355.653 0.000 0.239 0.109 0.406 1.137 0.149 9.862 # bump 2

### optim
# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/1/ --forward bump --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 501



### sample
python src/bayesian.py \
	--in_dir in/1_bump_syn0/ \
	--out_dir out/1_bump_syn0_hu_sample/ \
	--forward bump \
	--operation sample \
	--mcmc MALA \
	--para_eval_idx 5 6 \
	--epochs 2000 \
	--lr 0.001 \
	--to_save fig \
	--imres 256 \
	--para_all 1000 0.1 0.1 0.9 0.2 2.0553 0.1138 15 \
	# --para_all 1000 0.1 0.1 0.9 0.2 4 0.15 15 \

# python src/bayesian.py \
# 	--in_dir in/1_bump_real1/ \
# 	--out_dir out/1_bump_real1/ \
# 	--forward bump \
# 	--operation sample \
# 	--mcmc MALA \
# 	--para_eval_idx 0 1 2 3 4 5 6 7 \
# 	--epochs 1000 \
# 	--lr 0.01 \
# 	--to_save fig \
# 	--imres 512 \
# 	--para_all 2021.697 0.590 0.132 0.158 0.327 3.687 0.019 6.606

### optim
# python src/bayesian.py \
# 	--forward bump \
# 	--operation optimize \
# 	--in_dir in/real/1_bump_real2/ \
# 	--out_dir out/1_bump_real2_optim2/ \
# 	--para_eval_idx 0 1 2 3 4 5 6 7 \
# 	--epochs 1000 \
# 	--imres 512 \
# 	--lr 0.01 \
# 	--to_save fig \
