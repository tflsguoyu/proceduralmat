#  light, albedo, rough, rough_var, height, power, scale, shiftx, shifty, iSigma
# python3 src/demo.py --in_dir in/leather/ --out_dir out/ --forward leather --operation generate --fn 0.png

# sample
# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/1/ --forward leather --operation sample --fn 1.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 5001  --lf_lens 0.01 --lf_steps 4

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

python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/prior/ --forward leather --operation optimize --fn leather_black.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --size 10 --epochs 101 --lr 0.05

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/hmc/test_time/ --forward leather --operation sample --fn leather_black.png --para_all 609.647 0.003 0.002 0.006 0.187 0.124 0.004 1.945 0.723 -0.735 -0.710 5.263 --para_eval_idx 0 1 2 3 4 5 6 7 8 11 --err_sigma 0.001 0.01 0.001 0.001 --epochs 101 --lf_lens 0.005 --lf_steps 4 --size 10


# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/3/ --forward leather --operation optimize --fn 3.png --para_all 999 0.4 0.4 0.4 0.2 0.2 0.2 1.5 0.2 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --epochs 501 --lr 0.01  --to_save fig

# python3 src/demo.py --in_dir in/leather/ --out_dir out/leather/optim/leather_black/ --forward leather --operation optimize --fn leather_black.png --para_all 999 0.4 0.4 0.4 0.3 0.2 0.2 2 0.3 0 0 10 --para_eval_idx 0 1 2 3 4 5 6 7 8 9 10 11 --err_sigma 0.001 0.001 0.001 0.001 --size 10 --epochs 2001 --lr 0.05 --to_save fig