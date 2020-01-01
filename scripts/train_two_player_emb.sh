source_dataset_name=$1
target_dataset_name=$2

set -ex
python train.py --name lsgan_emb \
       --dataset_mode embedding --batch_size 32 --dataroot None \
       --max_dataset_size 200000 --preprocess center \
       --source_dataset_name $source_dataset_name --target_dataset_name $target_dataset_name \
       --model two_player_gan --gan_mode unconditional \
       --gpu_ids 0 \
       --d_loss_mode vanilla --g_loss_mode vanilla --which_D S \
       --optim_type Adam --lr_g 0.001 --lr_d 0.001 \
       --netD fc --z_dim 10 --netG fc --ngf 128 --ndf 128 --g_norm none --d_norm none \
       --init_type diagonal --init_gain 0.02 \
       --no_dropout --no_flip \
       --D_iters 1 \
       --score_name muse-csls-en \
       --print_freq 2000 --display_freq 2000 --score_freq 10000 \
       --save_latest_freq 100000 --save_giters_freq 100000 --most_frequent 75000
       # good hyper param for wgan
       # --d_loss_mode wgan --g_loss_mode wgan --which_D S --use_gp \
       # --optim_type Adam --lr_g 0.001 --lr_d 0.001 \ 
