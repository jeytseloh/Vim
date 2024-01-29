#!/bin/bash
conda activate /media/hdd/jiaxi/anaconda3/envs/vim2
cd /home/jiaxi/Vim/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token --batch-size 128 --num_workers 25 --data-path /media/hdd/jiaxi/VimDataset --data-set 'CT' --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual --no_amp
