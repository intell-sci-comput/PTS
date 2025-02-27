# pylint: disable=unused-variable

import json
import logging
from typing import Any

import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nesymres.dclasses import BFGSParams, FitParams
from torch import nn
from torch.distributions import Categorical

from .nesymres.architectures.beam_search import BeamHypotheses
from .nesymres.architectures.bfgs import bfgs
from .nesymres.architectures.set_encoder import SetEncoder
from .nesymres.dataset.generator import InvalidPrefixExpression

logger = logging.getLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_nesrymres_model(load_paper_model=True):
    if load_paper_model:
        # eq_setting_path = "./libs/NeuralSymbolicRegressionThatScales/jupyter/10MPaper/equation_config.json"
        # config_path = "./libs/NeuralSymbolicRegressionThatScales/jupyter/10MPaper/config.yaml"
        weights_path = "./models/nesymres_pre_train/10M.ckpt"
    else:
        weights_path = "./models/nesymres_pre_train/100M.ckpt"
    eq_setting_path = "./libs/NeuralSymbolicRegressionThatScales/jupyter/100M/eq_setting.json"
    config_path = "./libs/NeuralSymbolicRegressionThatScales/jupyter/100M/config.yaml"

    with open(eq_setting_path, "r") as json_file:
        eq_setting = json.load(json_file)

    cfg = omegaconf.OmegaConf.load(config_path)

    # Set up BFGS load rom the hydra config yaml
    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    model.set_bfgs(bfgs)
    model.set_eq_setting(eq_setting)
    model.cfg  # pylint: disable=pointless-statement
    return model


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.enc = SetEncoder(cfg)
        self.trg_pad_idx = cfg.trg_pad_idx
        self.tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
        if cfg.sinuisodal_embeddings:
            self.create_sinusoidal_embeddings(cfg.length_eq, cfg.dim_hidden, out=self.pos_embedding.weight)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_hidden,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dec_pf_dim,
            dropout=cfg.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.dec_layers)
        self.fc_out = nn.Linear(cfg.dim_hidden, cfg.output_dim)
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.eq = None

    def set_bfgs(self, bfgs):
        self.bfgs = bfgs

    def set_eq_setting(self, eq_setting):
        self.eq_setting = eq_setting

    def sample(self, X, y, beamsize=None, samples=100):
        if beamsize is not None:
            cfg_params = FitParams(
                word2id=self.eq_setting["word2id"],
                id2word={int(k): v for k, v in self.eq_setting["id2word"].items()},
                una_ops=self.eq_setting["una_ops"],
                bin_ops=self.eq_setting["bin_ops"],
                total_variables=list(self.eq_setting["total_variables"]),
                total_coefficients=list(self.eq_setting["total_coefficients"]),
                rewrite_functions=list(self.eq_setting["rewrite_functions"]),
                bfgs=bfgs,  # pyright: ignore
                # This parameter is a tradeoff between accuracy and fitting time
                beam_size=beamsize,
            )
            return self.fitfunc(X, y, cfg_params=cfg_params)
        else:
            cfg_params = FitParams(
                word2id=self.eq_setting["word2id"],
                id2word={int(k): v for k, v in self.eq_setting["id2word"].items()},
                una_ops=self.eq_setting["una_ops"],
                bin_ops=self.eq_setting["bin_ops"],
                total_variables=list(self.eq_setting["total_variables"]),
                total_coefficients=list(self.eq_setting["total_coefficients"]),
                rewrite_functions=list(self.eq_setting["rewrite_functions"]),
                bfgs=bfgs,  # pyright: ignore
                # This parameter is a tradeoff between accuracy and fitting time
                beam_size=beamsize,  # pyright: ignore
            )
            return self.fitfunc_samp(X, y, samples=samples, cfg_params=cfg_params)

    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # Odd indices
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))  # Even indices
        out.detach_()
        out.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0)).type_as(trg)
        return trg_pad_mask, mask

    # batch(res [equations that are valid, 4 (x's+y), support points],
    # tokens [eqs that are valid, token length], [eq.expr for eq in filtered_eqs - str expression])
    # pylint: disable-next=arguments-differ
    def forward(self, batch):
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, : (size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)
        trg = batch[1].long()  # tokens
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])  # trg_pad_mask, mask
        src_mask = None  # noqa: F841
        encoder_input = torch.cat((src_x, src_y), dim=-1)  # Why ? - same as b
        enc_src = self.enc(encoder_input)
        assert not torch.isnan(enc_src).any()
        pos = self.pos_embedding(
            torch.arange(0, batch[1].shape[1] - 1).unsqueeze(0).repeat(batch[1].shape[0], 1).type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        trg_ = self.dropout(te + pos)
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_src.permute(1, 0, 2),
            trg_mask2.bool(),
            tgt_key_padding_mask=trg_mask1.bool(),
        )
        output = self.fc_out(output)  # (seq_len, bs, vocab_size)
        return output, trg

    def compute_loss(self, output, trg):
        output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return loss

    # pylint: disable-next=arguments-differ
    def training_step(self, batch, _):
        output, trg = self.forward(batch)  # [21, 9, 60], [9, 22]
        loss = self.compute_loss(output, trg)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    # pylint: disable-next=arguments-differ
    def validation_step(self, batch, _):
        output, trg = self.forward(batch)
        loss = self.compute_loss(output, trg)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def sample_equations(self, samples, encoder_input, greedy=False):
        enc_src = self.enc(encoder_input)
        enc_src = enc_src.repeat(samples, 1, 1)
        assert not torch.isnan(enc_src).any()
        generated = torch.ones(
            [samples, 1],
            dtype=torch.long,
            device=self.device,
        )
        cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
        # Adapt for arity
        while cur_len < self.cfg.length_eq:
            generated_mask1, generated_mask2 = self.make_trg_mask(generated[:, :cur_len])
            pos = self.pos_embedding(
                torch.arange(0, cur_len)  # pyright: ignore
                .unsqueeze(0)
                .repeat(generated.shape[0], 1)
                .type_as(generated)  # attention here
            )
            te = self.tok_embedding(generated[:, :cur_len])
            trg_ = self.dropout(te + pos)
            output = self.decoder_transfomer(
                trg_.permute(1, 0, 2),  # Target
                enc_src.permute(1, 0, 2),  # Memory
                generated_mask2.float(),
                tgt_key_padding_mask=generated_mask1.bool(),
            )
            output = self.fc_out(output)
            output = output.permute(1, 0, 2).contiguous()

            # Masked log_softmax:
            # https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
            log_scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)
            scores = F.softmax(output[:, -1:, :], dim=-1).squeeze(1)  # noqa: F841

            if greedy:
                next_word = torch.argmax(log_scores, dim=1)
            else:
                multi_dist = Categorical(logits=log_scores)
                next_word = multi_dist.sample()
            generated = torch.cat([generated, next_word.unsqueeze(1)], dim=1)

            # update current length
            cur_len = cur_len + torch.tensor(1, device=self.device, dtype=torch.int64)
        return generated.detach()

    def compute_neg_log_likelihood(self, X, y, true_action):
        # X = X
        y = y[:, None]

        X = torch.tensor(X, device=self.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=self.device)
            X = torch.cat((X, pad), dim=2)
        y = torch.tensor(y, device=self.device).unsqueeze(0)
        with torch.no_grad():
            encoder_input = torch.cat((X, y), dim=2)  # .permute(0, 2, 1)
            # if self.device.type == "cuda":
            #     encoder_input = encoder_input.cuda()
            enc_src = self.enc(encoder_input)
            size = 1
            trg = torch.tensor(true_action).to(torch.int64).to(DEVICE)
            trg = trg.view(1, -1)
            # trg = trg.tile(1,1,1)
            # trg = batch[1].long()  # tokens
            trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])  # trg_pad_mask, mask
            src_mask = None  # noqa: F841
            enc_src = self.enc(encoder_input)
            assert not torch.isnan(enc_src).any()
            pos = self.pos_embedding(  # noqa: F841
                torch.arange(0, trg.shape[0] - 1).unsqueeze(0).repeat(size, 1).type_as(trg)
            )
            te = self.tok_embedding(trg[:, :-1])
            trg_ = self.dropout(te)  # + pos)
            output = self.decoder_transfomer(trg_.permute(1, 0, 2), enc_src.permute(1, 0, 2))
            # trg_mask2.bool(),
            # tgt_key_padding_mask=trg_mask1.bool(),
            # )
            output = self.fc_out(output)  # (seq_len, bs, vocab_size)
            loss = self.compute_loss(output, trg)
            return loss

    def fitfunc_samp(self, X, y, samples=100, greedy=False, cfg_params: Any = None, return_prefix_skeleton=True):
        """Same API as fit functions in sklearn:
        X [Number_of_points, Number_of_features],
        Y [Number_of_points]
        """
        # X = X
        y = y[:, None]

        X = torch.tensor(X, device=self.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=self.device)
            X = torch.cat((X, pad), dim=2)
        y = torch.tensor(y, device=self.device).unsqueeze(0)
        with torch.no_grad():
            encoder_input = torch.cat((X, y), dim=2)  # .permute(0, 2, 1)
            out = self.sample_equations(samples, encoder_input, greedy=greedy)

            # perc = 0
            # cnt = 0
            # gts = []
            best_preds = []  # noqa: F841
            best_preds_bfgs = []
            # best_L = []
            best_L_bfgs = []

            # flag = 0
            L_bfgs = []
            P_bfgs = []
            # counter = 1

            # fun_args = ",".join(chain(cfg_params.total_variables,"constant"))
            cfg_params.id2word[3] = "constant"
            if return_prefix_skeleton:
                from nesymres.architectures import data

                prefixes = []
                for ww in out:
                    try:
                        raw = data.de_tokenize(ww[1:].tolist(), cfg_params.id2word)
                        prefixes.append(raw)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"Error {e}")
                        continue
                return prefixes
            for ww in out:
                try:
                    # pylint: disable-next=no-member
                    pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(ww, X, y, cfg_params)

                except InvalidPrefixExpression:
                    continue
                # L_bfgs = loss_bfgs
                P_bfgs.append(str(pred_w_c))
                L_bfgs.append(loss_bfgs)

            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = None
            else:
                best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                best_L_bfgs.append(np.nanmin(L_bfgs))

            output = {
                "all_bfgs_preds": P_bfgs,
                "all_bfgs_loss": L_bfgs,
                "best_bfgs_preds": best_preds_bfgs,
                "best_bfgs_loss": best_L_bfgs,
            }
            self.eq = output["best_bfgs_preds"]
            return output

    def fitfunc(self, X, y, cfg_params: Any = None, return_prefix_skeleton=True):
        """Same API as fit functions in sklearn:
        X [Number_of_points, Number_of_features],
        Y [Number_of_points]
        """
        # X = X
        y = y[:, None]

        X = torch.tensor(X, device=self.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=self.device)
            X = torch.cat((X, pad), dim=2)
        y = torch.tensor(y, device=self.device).unsqueeze(0)
        with torch.no_grad():
            encoder_input = torch.cat((X, y), dim=2)  # .permute(0, 2, 1)
            # if self.device.type == "cuda":
            #     encoder_input = encoder_input.cuda()
            enc_src = self.enc(encoder_input)
            src_enc = enc_src
            shape_enc_src = (cfg_params.beam_size,) + src_enc.shape[1:]
            enc_src = (
                src_enc.unsqueeze(1)
                .expand((1, cfg_params.beam_size) + src_enc.shape[1:])
                .contiguous()
                .view(shape_enc_src)
            )
            # print(
            #     "Memory footprint of the encoder: {}GB \n".format(
            #         enc_src.element_size() * enc_src.nelement() / 10 ** (9)
            #     )
            # )
            assert enc_src.size(0) == cfg_params.beam_size
            generated = torch.zeros(
                [cfg_params.beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )
            generated[:, 0] = 1
            # trg_indexes = [[1] for i in range(bs*self.beam_size)]
            cache = {"slen": 0}
            # generated = torch.tensor(trg_indexes,device=self.device,dtype=torch.long)
            generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)
            done = False
            # Beam Scores
            beam_scores = torch.zeros(cfg_params.beam_size, device=self.device, dtype=torch.long)
            beam_scores[1:] = -1e9
            # beam_scores = beam_scores.view(-1)

            cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
            while cur_len < self.cfg.length_eq:
                # breakpoint()
                generated_mask1, generated_mask2 = self.make_trg_mask(generated[:, :cur_len])

                # dec_args = (generated, enc_src, generated_mask, src_mask)

                pos = self.pos_embedding(
                    torch.arange(0, cur_len)  # pyright: ignore  # attention here
                    .unsqueeze(0)
                    .repeat(generated.shape[0], 1)
                    .type_as(generated)
                )
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = self.dropout(te + pos)

                output = self.decoder_transfomer(
                    trg_.permute(1, 0, 2),  # Target
                    enc_src.permute(1, 0, 2),  # Memory
                    generated_mask2.float(),
                    tgt_key_padding_mask=generated_mask1.bool(),
                )
                output = self.fc_out(output)
                output = output.permute(1, 0, 2).contiguous()
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)

                assert output[:, -1:, :].shape == (
                    cfg_params.beam_size,
                    1,
                    self.cfg.length_eq,
                )

                n_words = scores.shape[-1]
                # select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
                # (bs, beam_size * n_words)
                _scores = _scores.view(cfg_params.beam_size * n_words)

                next_scores, next_words = torch.topk(
                    _scores, 2 * cfg_params.beam_size, dim=0, largest=True, sorted=True
                )
                assert len(next_scores) == len(next_words) == 2 * cfg_params.beam_size
                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words, next_scores):
                    # get beam and word IDs
                    beam_id = idx // n_words
                    # ^ https://github.com/huggingface/transformers/blob/088c1880b7bfd47777778d0d0fcc20e921bcf21e/src/transformers/generation_tf_utils.py#L1062
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == cfg_params.word2id["F"] or cur_len + 1 == self.cfg.length_eq:
                        generated_hyps.add(
                            generated[
                                beam_id,
                                :cur_len,
                            ]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append((value, word_id, beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == cfg_params.beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == self.cfg.length_eq else cfg_params.beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.trg_pad_idx, 0)] * cfg_params.beam_size  # pad the batch

                # next_batch_beam.extend(next_sent_beam)
                assert len(next_sent_beam) == cfg_params.beam_size

                beam_scores = torch.tensor(
                    [x[0] for x in next_sent_beam], device=self.device
                )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                beam_words = torch.tensor(
                    [x[1] for x in next_sent_beam], device=self.device
                )  # generated.new([x[1] for x in next_batch_beam])
                beam_idx = torch.tensor([x[2] for x in next_sent_beam], device=self.device)
                generated = generated[
                    beam_idx, :
                ]  # Extend here ! - overwrite previous 2 beams with best beam now found
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])  # pyright: ignore

                # update current length
                cur_len = cur_len + torch.tensor(1, device=self.device, dtype=torch.int64)

            # perc = 0
            # cnt = 0
            # gts = []
            best_preds = []  # noqa: F841
            best_preds_bfgs = []
            # best_L = []
            best_L_bfgs = []

            # flag = 0
            L_bfgs = []
            P_bfgs = []
            # counter = 1

            # fun_args = ",".join(chain(cfg_params.total_variables,"constant"))
            cfg_params.id2word[3] = "constant"
            if return_prefix_skeleton:
                from nesymres.architectures import data

                prefixes = []
                for __, ww in sorted(generated_hyps.hyp, key=lambda x: x[0], reverse=True):
                    try:
                        raw = data.de_tokenize(ww[1:].tolist(), cfg_params.id2word)
                        prefixes.append(raw)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"Error {e}")
                        continue
                return prefixes
            for __, ww in sorted(generated_hyps.hyp, key=lambda x: x[0], reverse=True):
                try:
                    # pylint: disable-next=no-member
                    pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(ww, X, y, cfg_params)
                except InvalidPrefixExpression:
                    continue
                # L_bfgs = loss_bfgs
                P_bfgs.append(str(pred_w_c))
                L_bfgs.append(loss_bfgs)

            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = None
            else:
                best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                best_L_bfgs.append(np.nanmin(L_bfgs))

            output = {
                "all_bfgs_preds": P_bfgs,
                "all_bfgs_loss": L_bfgs,
                "best_bfgs_preds": best_preds_bfgs,
                "best_bfgs_loss": best_L_bfgs,
            }
            self.eq = output["best_bfgs_preds"]
            return output

    def get_equation(
        self,
    ):
        return self.eq


# if __name__ == "__main__":
#     model = SetTransformer(
#         n_l_enc=2,
#         src_pad_idx=0,
#         trg_pad_idx=0,
#         dim_input=6,
#         output_dim=20,
#         dim_hidden=40,
#         dec_layers=1,
#         num_heads=8,
#         dec_pf_dim=40,
#         dec_dropout=0,
#         length_eq=30,
#         lr=0.001,
#         num_inds=20,
#         ln=True,
#         num_features=10,
#         is_sin_emb=False,
#         bit32=True,
#         norm=False,
#         activation="linear",
#         linear=False,
#         mean=torch.Tensor([1.0]),
#         std=torch.Tensor([1.0]),
#         input_normalization=False,
#     )
#     src_x = torch.rand([2, 5, 20])
#     src_y = torch.sin(torch.norm(src_x, dim=1)).unsqueeze(1)
#     inp_1 = torch.cat([src_x, src_y], dim=1)
#     inp_2 = torch.randint(0, 13, [2, 10])
#     batch = (inp_1, inp_2)
#     print(model)
