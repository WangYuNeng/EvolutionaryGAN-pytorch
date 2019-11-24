set -ex
python train.py --name egan_emb \
       --dataset_mode embedding --batch_size 32 --eval_size 256 --dataroot None \
       --model egan --gan_mode unconditional \
       --gpu_ids 0 \
       --download_root ./datasets/embedding --source_dataset_name cbow --target_dataset_name skipgram \
       --crop_size 32 --load_size 32 \
       --d_loss_mode vanilla --g_loss_mode nsgan vanilla lsgan --which_D S \
       --lambda_f 0.05 --candi_num 1 --z_type Uniform --z_dim 100 \
       --netD fc --netG fc --ngf 128 --ndf 128 --g_norm none --d_norm batch \
       --init_type normal --init_gain 0.02 \
       --no_dropout --no_flip \
       --D_iters 3 \
       --use_pytorch_scores --score_name IS --evaluation_size 50000 --fid_batch_size 500 \
       --print_freq 2000 --display_freq 2000 --score_freq 5000 --display_id -1 --save_giters_freq 100000
