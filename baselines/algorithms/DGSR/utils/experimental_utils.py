import logging
import random
from copy import deepcopy
from time import time

import numpy as np
import torch

logger = logging.getLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: Check not needed.
# def seed_all(seed=None):
#     """
#     Set the torch, numpy, and random module seeds based on the seed
#     specified in config. If there is no seed or it is None, a time-based
#     seed is used instead and is written to config.
#     """
#     # random.seed(seed)
#     # np.random.seed(seed)
#     # torch.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed)

#     # Default uses current time in milliseconds, modulo 1e9
#     if seed is None:
#         seed = round(time() * 1000) % int(1e9)

#     # Shift the seed based on task name
#     # This ensures a specified seed doesn't have similarities across different task names
#     task_name = Program.task.name
#     shifted_seed = seed + zlib.adler32(task_name.encode("utf-8"))

#     # Set the seeds using the shifted seed
#     torch.manual_seed(shifted_seed)
#     np.random.seed(shifted_seed)
#     random.seed(shifted_seed)


def seed_data_gen_manual(seed):
    random.seed(seed)
    np.random.seed(seed)


# def set_seeds():
#     """
#     Set the torch, numpy, and random module seeds based on the seed
#     specified in config. If there is no seed or it is None, a time-based
#     seed is used instead and is written to config.
#     """

#     seed = config_experiment.get("seed")

#     # Default uses current time in milliseconds, modulo 1e9
#     if seed is None:
#         seed = round(time() * 1000) % int(1e9)
#         config_experiment["seed"] = seed

#     # Shift the seed based on task name
#     # This ensures a specified seed doesn't have similarities across different task names
#     task_name = Program.task.name
#     shifted_seed = seed + zlib.adler32(task_name.encode("utf-8"))

#     # Set the seeds using the shifted seed
#     torch.manual_seed(shifted_seed)
#     np.random.seed(shifted_seed)
#     random.seed(shifted_seed)


def train_and_test(
    model,
    dltrain,
    dlval,
    dltest,
    optim,
    scheduler=None,
    EPOCHS=1000,
    patience=None,
    gradient_clip=1,
    snapshot_every_epochs=100,
):
    # Model is an super class of the actual model used - to give training methods,
    # Training loop parameters
    if not patience:
        patience = EPOCHS
    best_loss = float("inf")
    waiting = 0
    durations = []
    train_losses = []
    val_losses = []
    epoch_num = []

    for epoch in range(EPOCHS):
        iteration = 0
        epoch_train_loss_it_cum = 0

        model.train()
        start_time = time()

        for batch in dltrain:
            nbatch = [batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2]]
            # Single training step
            optim.zero_grad()
            train_loss = model.training_step(nbatch, None)
            train_loss.backward()
            # Optional gradient clipping
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), gradient_clip)
            optim.step()
            epoch_train_loss_it_cum += train_loss.item()

            logger.info(f"[epoch={epoch+1:04d}|iter={iteration+1:04d}] train_loss={train_loss:.5f}")
            iteration += 1
        epoch_train_loss = epoch_train_loss_it_cum / iteration

        epoch_duration = time() - start_time
        durations.append(epoch_duration)
        train_losses.append(epoch_train_loss)
        epoch_num.append(epoch)

        # Validation step
        model.eval()
        val_loss = 0
        for batch in dlval:
            nbatch = [batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2]]
            val_loss_t = model.validation_step(nbatch, None)
            val_loss += val_loss_t.item()
        val_losses.append(val_loss)
        logger.info(
            "[epoch={}] epoch_duration={:.2f} | train_loss={}\t| val_loss={}\t".format(
                epoch, epoch_duration, epoch_train_loss, val_loss
            )
        )

        # Learning rate scheduler
        if scheduler:
            scheduler.step()

        # Early stopping procedure
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model.state_dict())
            waiting = 0
        elif waiting > patience:
            break
        else:
            waiting += 1

        # if epoch % snapshot_every_epochs == 0:
        #     preds = system.predict(dltest)
        #     logger.info('Test trajectory ...')
        #     logger.info(preds[0, :, :].detach().cpu(
        #     ).numpy().tolist())

    logger.info(f"epoch_duration_mean={np.mean(durations):.5f}")

    # Load best model
    model.load_state_dict(best_model)  # pyright: ignore

    # Held-out test set step
    # test_loss, test_mse = system.test_step(dltest)
    # test_mse = test_mse.item()
    # logger.info('test_mse= {}'.format(test_mse))
    # return test_mse, torch.Tensor(train_losses).to(DEVICE),
    # torch.Tensor(train_nfes).to(DEVICE), torch.Tensor(epoch_num).to(DEVICE)


def test_pg(
    model, test_task, optim, scheduler=None, EPOCHS=1000, patience=None, gradient_clip=1, snapshot_every_epochs=100
):
    # Model is an super class of the actual model used - to give training methods,
    # Training loop parameters
    if not patience:
        patience = EPOCHS
    # best_loss = float("inf")
    # waiting = 0
    durations = []
    train_losses = []
    # val_losses = []
    epoch_num = []

    for epoch in range(EPOCHS):
        iteration = 0
        epoch_train_loss_it_cum = 0

        model.train()
        start_time = time()

        # nbatch = [batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2]]

        # Single training step
        optim.zero_grad()
        torch.autograd.set_detect_anomaly(True)  # pyright: ignore

        # , cfg_params=params_fit)
        train_loss = model.test_step(test_task)
        train_loss.backward()
        # Optional gradient clipping
        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(), gradient_clip)
        optim.step()
        epoch_train_loss_it_cum += train_loss.item()

        logger.info(f"[epoch={epoch+1:04d}|iter={iteration+1:04d}] train_loss={train_loss:.5f}")
        iteration += 1
        #

        epoch_train_loss = epoch_train_loss_it_cum / iteration

        epoch_duration = time() - start_time
        durations.append(epoch_duration)
        train_losses.append(epoch_train_loss)
        epoch_num.append(epoch)

        # # Validation step
        # model.eval()
        # val_loss = 0
        # for batch in dlval:
        #     nbatch = [batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2]]
        #     val_loss_t = model.validation_step(nbatch, None)
        #     val_loss += val_loss_t.item()
        # val_losses.append(val_loss)
        # logger.info('[epoch={}] epoch_duration={:.2f} | train_loss={}\t| val_loss={}\t'.format(
        #     epoch, epoch_duration, epoch_train_loss, val_loss))

        # # Learning rate scheduler
        # if scheduler:
        #     scheduler.step()

        # # Early stopping procedure
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_model = deepcopy(model.state_dict())
        #     waiting = 0
        # elif waiting > patience:
        #     break
        # else:
        #     waiting += 1

        # if epoch % snapshot_every_epochs == 0:
        #     preds = system.predict(dltest)
        #     logger.info('Test trajectory ...')
        #     logger.info(preds[0, :, :].detach().cpu(
        #     ).numpy().tolist())

    logger.info(f"epoch_duration_mean={np.mean(durations):.5f}")

    # Load best model
    # model.load_state_dict(best_model)

    # Held-out test set step
    # test_loss, test_mse = system.test_step(dltest)
    # test_mse = test_mse.item()
    # logger.info('test_mse= {}'.format(test_mse))
    # return test_mse, torch.Tensor(train_losses).to(DEVICE),
    # torch.Tensor(train_nfes).to(DEVICE), torch.Tensor(epoch_num).to(DEVICE)
