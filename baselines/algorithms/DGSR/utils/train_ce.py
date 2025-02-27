import logging
import time

import torch
from dso.utils import log_and_print

from .train import process_raw_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger()


def train_encoder_ce_controller(
    dltrain,
    dlval,
    dltest,
    controller,
    pool,
    gp_controller,
    output_file,
    pre_train,
    training_epochs,
    batch_outer_datasets,
    batch_inner_equations,
    load_pre_trained_path,
    config,
    controller_saved_path,
    n_epochs=None,
    n_samples=2000000,
    batch_size=1000,
    complexity="token",
    const_optimizer="scipy",
    const_params=None,
    alpha=0.5,
    epsilon=0.05,
    n_cores_batch=1,
    verbose=True,
    save_summary=False,
    save_all_epoch=False,
    baseline="R_e",
    b_jumpstart=False,
    early_stopping=True,
    hof=100,
    eval_all=False,
    save_pareto_front=True,
    debug=0,
    use_memory=False,
    memory_capacity=1e3,
    warm_start=None,
    memory_threshold=None,
    save_positional_entropy=False,
    save_top_samples_per_batch=0,
    save_cache=False,
    save_cache_r_min=0.9,
    save_freq=None,
    save_token_count=False,
    learning_rate=0.001,
    gradient_clip=1,
):
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)

    for epoch in range(training_epochs):
        it = 0
        t2 = time.time()
        for raw_batch in dltrain:
            if raw_batch[0].nelement() == 0:
                log_and_print("WARNING no data in batch skipping")
                continue
            data, eqs = process_raw_batch(raw_batch, controller)
            if data.nelement() == 0:
                log_and_print("WARNING no data in batch filtered eqs skipping")
                continue
            t0 = time.time()
            data_to_encode = data.permute(0, 2, 1)
            batch_sos_tokens = torch.ones((1, data.shape[0]), dtype=torch.long).to(DEVICE) * controller.sos_token
            token_eqs = eqs[:, : controller.max_length].to(torch.long).to(DEVICE)
            tgt = torch.cat((batch_sos_tokens, token_eqs.T), 0)
            optimizer.zero_grad()
            # try:
            optimizer.zero_grad()
            loss = controller.train_mle_loss(data_to_encode.to(DEVICE), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), gradient_clip)  # pyright: ignore
            optimizer.step()
            if it % 100 == 0:
                torch.save(controller.state_dict(), controller_saved_path)
            if it % 20 == 0:
                log_and_print(
                    f"[epoch={epoch+1:04d}|iter={it+1:04d}] ce_train_loss={loss.item():.5f} \t| "
                    f"s/it={time.time() - t0:.5f} \t| s/batch={time.time() - t2:.5f}"
                )
            it += 1
            t2 = time.time()
            # except Exception as e:
            #     log_and_print('[ERROR] Error {}'.format(str(e)))
            #     controller_saved_path = controller_saved_path.replace(
            #         'controller', 'controller_errored')
            #     continue
        # End of iter
        torch.cuda.empty_cache()
