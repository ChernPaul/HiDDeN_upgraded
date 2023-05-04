import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter


def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherw  ise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)
    # early stopping
    last_loss = 100
    # loss min difference value
    delta = 0.0001
    # trigger times >= patience
    patience_abs = 15
    patience_improve = 20
    patience_jumps = 7
    seq_trigger_times_abs = 0
    seq_trigger_times_imp = 0
    jumps_trigger = 0
    min_loss_value = 100
    # early stopping
    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):

        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            # print(image.numpy().min())
            # print(image.numpy().max())
            losses, _ = model.train_on_batch([image, message])


            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False

        # early stopping block started

        current_loss = validation_losses['loss           '].avg
        print('The current loss:', current_loss)

        if epoch % 15 == 0:
            jumps_trigger = 0

        if last_loss - current_loss < 0:
            seq_trigger_times_imp += 1
            if jumps_trigger >= patience_jumps:
                print('trigger times for jumps:', jumps_trigger)
                print('Early stopping!\n')
                utils.log_progress(validation_losses)
                logging.info('-' * 40)
                utils.save_checkpoint(model, train_options.experiment_name, epoch,
                                      os.path.join(this_run_folder, 'checkpoints'))
                utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                                   time.time() - epoch_start)
                break
        # last - current > delta means improvement
        if last_loss - current_loss <= delta:
            seq_trigger_times_abs += 1
            if seq_trigger_times_abs >= patience_abs:
                print('trigger times for delta:', seq_trigger_times_abs)
                print('Early stopping!\n')
                utils.log_progress(validation_losses)
                logging.info('-' * 40)
                utils.save_checkpoint(model, train_options.experiment_name, epoch,
                                      os.path.join(this_run_folder, 'checkpoints'))
                utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                                   time.time() - epoch_start)
                break
        else:
            seq_trigger_times_abs = 0

        if min_loss_value < current_loss:
            seq_trigger_times_imp += 1
            if seq_trigger_times_imp >= patience_improve:
                print('trigger times for improve:', seq_trigger_times_imp)
                print('Early stopping!\n')
                utils.log_progress(validation_losses)
                logging.info('-' * 40)
                utils.save_checkpoint(model, train_options.experiment_name, epoch,
                                      os.path.join(this_run_folder, 'checkpoints'))
                utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                                   time.time() - epoch_start)
                break
        else:
            seq_trigger_times_imp = 0
            min_loss_value = current_loss

        # set current loss as last lost for next step
        last_loss = current_loss

        # early stopping block ended
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
