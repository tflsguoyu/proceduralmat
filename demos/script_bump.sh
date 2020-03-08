# light, albedo(r,g,b), rough, fsigma, fscale, iSigma

# python3 src/demo.py --in_dir in/bump/ --out_dir out/ --forward bump --operation generate --fn 0.png --para_all 1058.332 0.623 0.332 0.104 0.247 0.821 0.600 9.796


# sample
# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/1/ --forward bump --operation sample --fn 1.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 5001 --lf_lens 0.05 --lf_steps 4

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/2/ --forward bump --operation sample --fn 2.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 5001 --lf_lens 0.05 --lf_steps 4

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/3/ --forward bump --operation sample --fn 3.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 5001 --lf_lens 0.05 --lf_steps 4

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/4/ --forward bump --operation sample --fn 4.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 5001 --lf_lens 0.05 --lf_steps 4

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/5/ --forward bump --operation sample --fn 5.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 5001 --lf_lens 0.05 --lf_steps 4

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/6/ --forward bump --operation sample --fn 6.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 5001 --lf_lens 0.05 --lf_steps 4

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/real1/ --forward bump --operation sample --fn real1.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 4001 --lf_lens 1 --lf_steps 5


# optim
# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/1/ --forward bump --operation optimize --fn 1.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 501

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/2/ --forward bump --operation optimize --fn 2.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 501

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/3/ --forward bump --operation optimize --fn 3.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 501

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/4/ --forward bump --operation optimize --fn 4.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 501

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/5/ --forward bump --operation optimize --fn 5.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 501

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/6/ --forward bump --operation optimize --fn 6.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --epochs 501

########################################

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/0/ --forward bump --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.2 1 0.2 10 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.001 --epochs 20001 --lr 0.001

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/0/ --forward bump --operation sample --fn 0.png --para_all 1058.332 0.623 0.332 0.104 0.247 0.821 0.600 9.796 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.01 --epochs 20001 --lf_lens 0.1 --lf_steps 4


# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/red-wall3/ --forward bump --operation sample --fn red-wall2.png --para_all 1965.031 0.494 0.091 0.103 0.382 3.090 0.027 7.748 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.01 0.001 0.001 --epochs 20001 --lf_lens 0.005 --lf_steps 4

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/red-wall3/ --forward bump --operation optimize --fn red-wall2.png --para_all 999 0.4 0.4 0.4 0.2 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.001 0.001 0.001 --epochs 2001 --lr 0.01 --to_save fig




#########  test time


# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/prior/ --forward bump --operation optimize --fn red-wall.png --para_all 999 0.4 0.4 0.4 0.2 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1 --lr 0.01

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/hmc/test_time/ --forward bump --operation sample --fn red-wall2.png --para_all 1965.031 0.494 0.091 0.103 0.382 3.090 0.027 7.748 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.01 0.001 0.001 --epochs 101 --lf_lens 0.005 --lf_steps 4


##### video
# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/0/ --forward bump --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.2 5 0.5 10 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1001 --lr 0.01 --to_save fig

# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/red-wall/ --forward bump --operation optimize --fn red-wall.png --para_all 999 0.4 0.4 0.4 0.2 5 0.05 10 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.001 0.001 0.001 --epochs 2001 --lr 0.01 --to_save fig
# python3 src/demo.py --in_dir in/bump/ --out_dir out/bump/optim/0/ --forward bump --operation optimize --fn 0.png --para_all 999 0.4 0.4 0.4 0.2 5 0.5 10 --para_eval_idx 0 1 2 3 4 5 6 7 --err_sigma 0.001 0.001 0.001 0.001 --epochs 1001 --lr 0.1 --to_save fig


#######
python3 src/demo.py \
	--in_dir in/bump/ \
	--out_dir out_egsr/bump/mala/0_1/ \
	--forward bump \
	--operation sample \
	--sampleMethod MALA \
	--fn 0.png \
	--para_all 999 0.4 0.4 0.4 0.2 0.1 0.1 10 \
	--para_eval_idx 0 1 2 3 4 5 6 7 \
	--err_sigma 0.01 0.01 \
	--epochs 10000 \
	--lr 0.01 \
	--diminish 0.1 \
	--to_save fig

# python3 src/demo.py --in_dir in/bump/ --out_dir out_egsr/bump/hmc/0/ --forward bump --operation sample --sampleMethod HMC --fn 0.png --para_all 1058.332 0.623 0.332 0.104 0.247 0.1 0.1 9.796 --para_eval_idx 5 6 --err_sigma 0.01 0.01 --epochs 10000 --lf_lens 0.2 --lf_steps 2 --to_save fig


