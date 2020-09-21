# python src/hu2019.py \
# 	--in_dir hu2019/in/real/ \
# 	--out_dir hu2019/out/bump/ \
# 	--forward bump \
# 	--operation train \
# 	--epochs 100 \
# 	--iters_train 800 \
# 	--iters_val 200 \
# 	--lr 0.0001 \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 \
# 	--para     0 0 0 0 0 0 0 0 \
# 	# --resume hu2019/out/bump/checkpoint_0050.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/ \
# 	--out_dir hu2019/out/leather/ \
# 	--forward leather \
# 	--operation train \
# 	--epochs 100 \
# 	--iters_train 800 \
# 	--iters_val 200 \
# 	--lr 0.0001 \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0  0 \
# 	# --resume hu2019/out/bump/checkpoint_0050.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/ \
# 	--out_dir hu2019/out/plaster/ \
# 	--forward plaster \
# 	--operation train \
# 	--epochs 100 \
# 	--iters_train 800 \
# 	--iters_val 200 \
# 	--lr 0.0001 \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0 \
# 	# --resume hu2019/out/bump/checkpoint_0050.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/ \
# 	--out_dir hu2019/out/flake/ \
# 	--forward flake \
# 	--operation train \
# 	--epochs 100 \
# 	--iters_train 800 \
# 	--iters_val 200 \
# 	--lr 0.0001 \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0  0  0  0 \
# 	# --resume hu2019/out/bump/checkpoint_0050.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/ \
# 	--out_dir hu2019/out/metal/ \
# 	--forward metal \
# 	--operation train \
# 	--epochs 100 \
# 	--iters_train 800 \
# 	--iters_val 200 \
# 	--lr 0.0001 \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 \
# 	--para     0 0 0 0 0 0 0 0 0 0 \
# 	# --resume hu2019/out/bump/checkpoint_0050.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/ \
# 	--out_dir hu2019/out/wood/ \
# 	--forward wood \
# 	--operation train \
# 	--epochs 100 \
# 	--iters_train 800 \
# 	--iters_val 200 \
# 	--lr 0.0001 \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0 \
# 	# --resume hu2019/out/wood/checkpoint_0095.pth

python src/hu2019.py \
	--in_dir hu2019/in/1_bump_syn0/input.png \
	--out_dir hu2019/out/bump/ \
	--forward bump \
	--operation test \
	--cuda \
	--para_idx 0 1 2 3 4 5 6 7 \
	--para     0 0 0 0 0 0 0 0 \
	--resume hu2019/out/bump/checkpoint_0100.pth
	# --in_dir hu2019/in/1_bump_syn0/input.png \

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/flake_2/target.jpg \
# 	--out_dir hu2019/out/flake/ \
# 	--forward flake \
# 	--operation test \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0  0  0  0 \
# 	--resume hu2019/out/flake/checkpoint_0100.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/leather_4/target.jpg \
# 	--out_dir hu2019/out/leather/ \
# 	--forward leather \
# 	--operation test \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0  0 \
# 	--resume hu2019/out/leather/checkpoint_0100.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/metal_1/target.jpg \
# 	--out_dir hu2019/out/metal/ \
# 	--forward metal \
# 	--operation test \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 \
# 	--para     0 0 0 0 0 0 0 0 0 0 \
# 	--resume hu2019/out/metal/checkpoint_0100.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/plaster_2/target.jpg \
# 	--out_dir hu2019/out/plaster/ \
# 	--forward plaster \
# 	--operation test \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0 \
# 	--resume hu2019/out/plaster/checkpoint_0100.pth

# python src/hu2019.py \
# 	--in_dir hu2019/in/real/wood_1/target.jpg \
# 	--out_dir hu2019/out/wood/ \
# 	--forward wood \
# 	--operation test \
# 	--cuda \
# 	--para_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
# 	--para     0 0 0 0 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0 \
# 	--resume hu2019/out/wood/checkpoint_0100.pth