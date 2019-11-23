set -ex
python train.py --name lsgan_cifar10 \
       --dataset_mode embedding --batch_size 32 --dataroot None \
       --model two_player_gan --gan_mode unconditional \
       --gpu_ids 0 \
       --download_root /shares/Public/rays_data/embedding --source_dataset_name skipgram --target_dataset_name skipgram \
       --crop_size 32 --load_size 32 \
       --d_loss_mode lsgan --g_loss_mode lsgan --which_D S \
       --netD fc --netG fc --ngf 128 --ndf 128 --g_norm none --d_norm batch \
       --init_type normal --init_gain 0.02 \
       --no_dropout --no_flip \
       --D_iters 3 \
       --use_pytorch_scores --score_name IS --evaluation_size 50000 --fid_batch_size 500 \
       --print_freq 2000 --display_freq 2000 --score_freq 5000 --display_id -1 --save_giters_freq 100000
