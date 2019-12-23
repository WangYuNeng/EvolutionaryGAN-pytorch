"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training.
During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
"""
import time
import json
import os

from dotenv import load_dotenv
load_dotenv('./.env')
import wandb

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from evaluators import get_evaluator


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    evaluator = get_evaluator(opt, model=model, dataset=dataset)
    total_iters = 0  # the total number of training iterations
    epoch = 0

    if opt.wandb:
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(config=opt)

    while total_iters < opt.total_num_giters:
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            epoch_iter += 1
            total_iters += 1

            if total_iters % opt.display_freq == 0 and opt.dataset_mode == 'torchvision':
                samples = model.get_output()
                if opt.wandb and opt.dataset_mode=="torchvision":
                    wandb.log(
                        {
                            "fake-samples": [wandb.Image((im + 1) / 2) for im in samples],
                            "real-samples": [wandb.Image((im + 1) / 2) for im in data['data']],
                        },
                        step=total_iters,
                    )

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                print('iters: ', total_iters, end='')
                print(json.dumps(losses, indent=4))
                if opt.wandb:
                    wandb.log(losses, step=total_iters)
                    if opt.model == "egan":
                        print('loss_mode: ', model.current_loss_mode)
                        wandb.log({"loss_mode": model.current_loss_mode}, step=total_iters)


            if total_iters % opt.score_freq == 0:  # print generation scores and save logging information to the disk
                scores = evaluator.get_current_scores()
                print('iters: ', total_iters, end='')
                print(json.dumps(scores, indent=4))
                if opt.wandb:
                    wandb.log(scores, step=total_iters)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

            if total_iters % opt.save_giters_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(total_iters)

        epoch += 1
        print('(epoch_%d) End of giters %d / %d \t Time Taken: %d sec \t %d sample / s' % (
        epoch, total_iters, opt.total_num_giters, time.time() - epoch_start_time, len(dataset) // (time.time() - epoch_start_time)))
