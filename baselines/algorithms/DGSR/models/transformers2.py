"""Controller used to generate distribution over hierarchical, variable-length objects."""
import logging
import math

import numpy as np
import torch
from dso.memory import Batch
from dso.prior import LengthConstraint
from dso.state_manager import TorchHierarchicalStateManager
from dso.utils import log_and_print
from nesymres.architectures.set_encoder import SetEncoder
from torch import nn
from torch.autograd import Function

logger = logging.getLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(DEVICE)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


def safe_cross_entropy(p, logq, dim=-1):
    safe_logq = torch.where(p == 0, torch.ones_like(logq).to(DEVICE), logq)
    return -torch.sum(p * safe_logq, dim)


def numpy_batch_to_tensor_batch(batch):
    if batch is None:
        return None
    else:
        return Batch(
            actions=torch.from_numpy(batch.actions).to(DEVICE),
            obs=torch.from_numpy(batch.obs).to(DEVICE),
            priors=torch.from_numpy(batch.priors).to(DEVICE),
            lengths=torch.from_numpy(batch.lengths).to(DEVICE),
            rewards=torch.from_numpy(batch.rewards).to(DEVICE),
            on_policy=torch.from_numpy(batch.on_policy).to(DEVICE),
            data_to_encode=torch.from_numpy(batch.data_to_encode).to(DEVICE),
            tgt=torch.from_numpy(batch.tgt).to(DEVICE),
        )


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# pylint: disable-next=abstract-method
class GetNextObs(Function):
    @staticmethod
    # pylint: disable-next=arguments-differ
    def forward(ctx, obj, actions, obs):
        np_actions = actions.detach().cpu().numpy()
        np_obs = obs.detach().cpu().numpy()
        next_obs, next_prior = obj.task.get_next_obs(np_actions, np_obs)
        return obs.new(next_obs), obs.new(next_prior)

    @staticmethod
    # pylint: disable-next=arguments-differ
    def backward(ctx, grad_output):
        return grad_output


class PositionalEncoding(nn.Module):
    # Also works for non-even dimensions
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if (d_model % 2) == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.scale * self.pe[: x.size(0), :]  # pyright: ignore
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden,
        input_pdt=0,
        output_pdt=0,
        enc_layers=3,
        dec_layers=1,
        dropout=0.1,
        input_already_encoded=True,
        max_len=1024,
    ):
        super(TransformerModel, self).__init__()
        self.input_pdt = input_pdt
        self.output_pdt = output_pdt
        self.input_already_encoded = input_already_encoded
        self.out_dim = out_dim
        self.in_dim = in_dim
        if not self.input_already_encoded:
            self.encoder = nn.Embedding(in_dim, hidden)
        else:
            log_and_print(f"Tranformer overwritting hidden size to input_dim {in_dim}")
            hidden = in_dim
        self.decoder = nn.Embedding(out_dim, hidden)
        self.pos_encoder = PositionalEncoding(in_dim, dropout, max_len=max_len)
        self.pos_decoder = PositionalEncoding(hidden, dropout, max_len=max_len)

        self.hidden = hidden
        nhead = int(np.ceil(hidden / 64))
        if (hidden % nhead) != 0:
            nhead = 1
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=hidden * 4,  # 4,
            dropout=dropout,
            activation="relu",
        )
        self.fc_out = nn.Linear(hidden, out_dim)

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def make_len_mask(
        self,
        inp,
        input=False,  # pylint: disable=redefined-builtin
    ):
        if input:
            return (inp == self.input_pdt).transpose(0, 1)
        else:
            return (inp == self.output_pdt).transpose(0, 1)

    def forward(self, src, tgt):
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            self.tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)

        # src_pad_mask = self.make_len_mask(src, input=True)
        src_pad_mask = None
        # tgt_pad_mask = self.make_len_mask(tgt)
        tgt_pad_mask = None

        if not self.input_already_encoded:
            src = self.encoder(src)
        src = self.pos_encoder(src)

        tgt = self.decoder(tgt)
        tgt = self.pos_decoder(tgt)

        output = self.transformer(
            src,
            tgt,
            src_mask=self.src_mask,
            tgt_mask=self.tgt_mask,
            memory_mask=self.memory_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        output = self.fc_out(output)

        return output


class TransformerCustomEncoderModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden,
        cfg,
        input_pdt=None,
        output_pdt=None,
        enc_layers=3,
        dec_layers=1,
        dropout=0.1,
        input_already_encoded=True,
        max_len=1024,
        has_encoder=True,
    ):
        super(TransformerCustomEncoderModel, self).__init__()
        self.has_encoder = has_encoder
        self.input_pdt = input_pdt
        self.output_pdt = output_pdt
        self.input_already_encoded = input_already_encoded
        self.out_dim = out_dim
        self.in_dim = in_dim
        if not self.input_already_encoded:
            self.encoder = nn.Embedding(in_dim, hidden)
        else:
            log_and_print(f"Transformer overwriting hidden size to input_dim {in_dim}")
            hidden = in_dim
        self.decoder = nn.Embedding(out_dim, hidden)
        self.pos_encoder = PositionalEncoding(in_dim, dropout, max_len=max_len)
        self.pos_decoder = PositionalEncoding(hidden, dropout, max_len=max_len)

        self.hidden = hidden
        nhead = int(np.ceil(hidden / 64))
        if (hidden % nhead) != 0:
            nhead = 1
        self.nhead = nhead
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=hidden * 4,  # 4,
            dropout=dropout,
            activation="relu",
        )
        self.fc_out = nn.Linear(hidden, out_dim)

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None
        cfg["num_heads"] = nhead
        cfg["dim_hidden"] = hidden
        cfg["num_features"] = 1
        cfg["linear"] = True
        cfg["bit16"] = False
        cfg["n_l_enc"] = 3
        cfg["num_inds"] = 64
        log_and_print(f"Encoder params: {cfg}")
        if has_encoder:
            self.data_encoder = SetEncoder(cfg)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def make_len_mask(
        self,
        inp,
        input=False,  # pylint: disable=redefined-builtin
    ):
        if input:
            return (inp == self.input_pdt).transpose(0, 1)
        else:
            return (inp == self.output_pdt).transpose(0, 1)

    def forward(self, data_src, src, tgt):
        seq_length = tgt.shape[0]  # pylint: disable=unused-variable  # noqa: F841
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            self.tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)

        # src_pad_mask = self.make_len_mask(src, input=True)
        src_pad_mask = None
        if self.output_pdt:
            tgt_pad_mask = self.make_len_mask(tgt)

        tgt = self.decoder(tgt)
        tgt = self.pos_decoder(tgt)

        if src is not None:
            if not self.input_already_encoded:
                src = self.encoder(src)
            src = self.pos_encoder(src)
            memory = self.transformer.encoder(src, mask=self.src_mask, src_key_padding_mask=src_pad_mask)
        if data_src is not None and data_src.nelement() != 0:
            if self.has_encoder:
                data_memory = self.data_encoder(data_src)
            else:
                data_memory = torch.zeros(memory.shape[1], 1, memory.shape[2]).to(memory.device)  # pyright: ignore
            assert not torch.isnan(data_memory).any()
            if src is not None:
                memory = torch.cat([data_memory.permute(1, 0, 2), memory], axis=0)  # pyright: ignore
                # memory = memory + data_memory.permute(1, 0, 2)
                # Memory grows in (seq_len,1,1)
                # Data_memory does not grow () - therefore broadcasting operation - instead should concatenate them
            else:
                memory = data_memory.permute(1, 0, 2)
                # memory = data_memory
            # out_memory = data_memory.permute(1, 0, 2).tile(memory.shape[0], 1, 1)

        output = self.transformer.decoder(
            tgt,
            memory,  # pyright: ignore
            tgt_mask=self.tgt_mask,
            memory_mask=self.memory_mask,
            tgt_key_padding_mask=tgt_pad_mask,  # pyright: ignore
            memory_key_padding_mask=src_pad_mask,
        )
        # output = self.transformer(
        #     src,
        #     tgt,
        #     src_mask=self.src_mask,
        #     tgt_mask=self.tgt_mask,
        #     memory_mask=self.memory_mask,
        #     src_key_padding_mask=src_pad_mask,
        #     tgt_key_padding_mask=tgt_pad_mask,
        #     memory_key_padding_mask=src_pad_mask
        # )
        output = self.fc_out(output)

        return output


# pylint: disable-next=abstract-method
class TransformerTreeController(nn.Module):
    """
    Recurrent neural network (RNN) controller used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees. It is trained using REINFORCE with baseline.

    Parameters
    ----------
    prior : dso.prior.JointPrior
        JointPrior object used to adjust probabilities during sampling.

    state_manager: dso.state_manager.StateManager
        Object that handles the state features to be used

    summary : bool
        Write tensorboard summaries?

    debug : int
        Debug level, also used in learn(). 0: No debug. 1: logger.info shapes and
        number of parameters for each variable.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    initializer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

    optimizer : str
        Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

    learning_rate : float
        Learning rate for optimizer.

    entropy_weight : float
        Coefficient for entropy bonus.

    entropy_gamma : float or None
        Gamma in entropy decay. None (or
        equivalently, 1.0) turns off entropy decay.

    pqt : bool
        Train with priority queue training (PQT)?

    pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?

    max_length : int or None
        Maximum sequence length. This will be overridden if a LengthConstraint
        with a maximum length is part of the prior.

    batch_size: int
        Most likely 500 or 1000
    """

    def __init__(
        self,
        prior,
        library,
        task,
        config_state_manager=None,
        encoder_input_dim=None,
        debug=0,
        summary=False,
        # RNN cell hyperparameters
        cell="lstm",
        num_layers=1,
        num_units=32,
        # Optimizer hyperparameters
        optimizer="adam",
        initializer="zeros",
        learning_rate=0.001,
        # Loss hyperparameters
        entropy_weight=0.005,
        entropy_gamma=1.0,
        # PQT hyperparameters
        pqt=False,
        pqt_k=10,
        pqt_batch_size=1,
        pqt_weight=200.0,
        pqt_use_pg=False,
        # Other hyperparameters
        max_length=30,
        batch_size=1000,
        n_objects=1,
        rl_weight=1.0,
    ):
        super(TransformerTreeController, self).__init__()
        self.encoder_input_dim = encoder_input_dim
        self.learning_rate = learning_rate
        self.rl_weight = rl_weight

        self.prior = prior
        self.summary = summary
        # self.rng = np.random.RandomState(0) # Used for PPO minibatch sampling
        self.n_objects = n_objects
        self.num_units = num_units

        lib = library

        # Find max_length from the LengthConstraint prior, if it exists
        # Both priors will never happen in the same experiment
        prior_max_length = None
        for single_prior in self.prior.priors:
            if isinstance(single_prior, LengthConstraint):
                if single_prior.max is not None:
                    prior_max_length = single_prior.max
                    self.max_length = prior_max_length
                break

        if prior_max_length is None:
            assert max_length is not None, "max_length must be specified if " "there is no LengthConstraint."
            self.max_length = max_length
            logger.info(
                "WARNING: Maximum length not constrained. Sequences will "
                "stop at {} and complete by repeating the first input "
                "variable.".format(self.max_length)
            )
        elif max_length is not None and max_length != self.max_length:
            logger.info(
                "WARNING: max_length ({}) will be overridden by value from "
                "LengthConstraint ({}).".format(max_length, self.max_length)
            )
        self.max_length *= self.n_objects
        max_length = self.max_length

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.pqt = pqt
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg

        self.n_choices = lib.L

        self.batch_size = batch_size
        # self.baseline = torch.zeros(1)

        # Entropy decay vector
        if entropy_gamma is None:
            entropy_gamma = 1.0
        # Could make this a tensor
        self.entropy_gamma_decay = torch.Tensor([entropy_gamma**t for t in range(max_length)]).to(DEVICE)

        # Build controller RNN
        # Create recurrent cell
        if isinstance(num_units, int):
            num_units = num_units * num_layers

        if "type" in config_state_manager:  # pyright: ignore
            del config_state_manager["type"]  # pyright: ignore
        self.state_manager = TorchHierarchicalStateManager(
            library,
            max_length,
            **config_state_manager,  # pyright: ignore
        )

        # Calculate input size
        self.input_dim_size = self.state_manager.input_dim_size

        # (input_size, hidden_size)

        self.task = task

        # Transformer
        # if self.input_dim_size % 2 != 0:
        #     log_and_print('Error, input_size should be divisible by 2')
        #     raise ValueError("Error, input_size should be divisible by 2")

        self.tgt_padding_token = self.n_choices
        self.sos_token = self.n_choices + 1
        out_dim = self.n_choices + 2  # EOS token at last index
        # if out_dim % 2 != 0:
        #     log_and_print('Error, out_dim should be divisible by 2')
        #     raise ValueError("Error, out_dim should be divisible by 2")
        self.out_dim = out_dim
        # in_dim = int(np.ceil(in_dim / 2) * 2)
        # out_dim = int(np.ceil(out_dim / 2) * 2)
        # num_units = hidden_units
        self.model = TransformerModel(
            self.input_dim_size,
            self.out_dim,
            num_units,
            enc_layers=3,
            dec_layers=2,
            dropout=0,
            input_already_encoded=True,
        )

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tgt_padding_token)

    def _sample(self, batch_size=None, input_=None):
        if batch_size is None:
            batch_size = self.batch_size
            self.batch_size = batch_size
        initial_obs = self.task.reset_task(self.prior)
        initial_obs = initial_obs.expand(batch_size, initial_obs.shape[0])
        initial_obs = self.state_manager.process_state(initial_obs)

        # Get initial prior
        initial_prior = torch.from_numpy(self.prior.initial_prior()).to(DEVICE)
        initial_prior = initial_prior.expand(batch_size, self.n_choices)

        # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
        current_length = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        obs = initial_obs
        next_input = self.state_manager.get_tensor_input(obs)
        # Could add a start token for inputs - none at present
        inputs = next_input.unsqueeze(0)
        obs_ta = []
        priors_ta = []
        prior = initial_prior
        lengths = torch.ones(batch_size, dtype=torch.int32).to(DEVICE)
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        # Initial state
        while not all(finished):
            current_length += 1
            output = self.model(inputs, tgt_actions)
            cell_output = output[-1, :, :-2]
            # Adapt prior for EOS token
            logits = cell_output + prior
            action = torch.distributions.categorical.Categorical(logits=logits).sample()
            tgt_actions = torch.cat((tgt_actions, action.view(1, -1)), 0)

            # Compute obs and prior
            next_obs, next_prior = GetNextObs.apply(self, tgt_actions[1:, :].permute(1, 0), obs)  # pyright: ignore
            next_obs = self.state_manager.process_state(next_obs)
            next_input = self.state_manager.get_tensor_input(next_obs)
            inputs = torch.cat((inputs, next_input.unsqueeze(0)), 0)

            obs_ta.append(obs)
            priors_ta.append(prior)
            # finished = torch.logical_or(
            #     finished,
            #     current_length >= max_length)
            finished = finished + (current_length >= self.max_length)
            next_lengths = torch.where(finished, lengths, (current_length + 1).expand(batch_size))  # Ever finished
            # (current_length + 1).unsqueeze(0).expand(batch_size, 1))
            obs = next_obs
            prior = next_prior
            lengths = next_lengths

        actions = tgt_actions[1:, :].permute(1, 0)
        # (?, obs_dim, max_length)
        obs = torch.stack(obs_ta).permute(1, 2, 0)
        # (?, max_length, n_choices)
        priors = torch.stack(priors_ta, 1)
        return actions, obs, priors

    def compute_neg_log_likelihood(self, data_to_encode, true_action, B=None):
        inputs_ = torch.Tensor([0]).to(DEVICE)
        batch_size = 1
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int64).to(DEVICE) * self.sos_token
        actions = torch.tensor(true_action).to(DEVICE).view(1, -1)
        tgt_actions = torch.cat((tgt_actions, actions.permute(1, 0)), 0)

        outputs = self.model(inputs_, tgt_actions[:-1,])
        logits = outputs[:, :, :-2]
        neg_log_likelihood = self.ce_loss(logits.permute(1, 2, 0), tgt_actions[1:, :].T)
        return neg_log_likelihood

    def make_neglogp_and_entropy(self, B, test=False):
        # Generates tensor for neglogp of a given batch
        # Loop_fn is defined in the function:
        # Essentially only calculating up to the sequence_lengths given:
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/raw_rnn
        inputs = self.state_manager.get_tensor_input(B.obs)
        sequence_length = B.lengths  # pylint: disable=unused-variable  # noqa: F841
        batch_size = B.obs.shape[0]
        data_to_encode = B.data_to_encode  # pylint: disable=unused-variable  # noqa: F841
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        actions = B.actions
        tgt_actions = torch.cat((tgt_actions, actions.permute(1, 0)), 0)

        outputs = self.model(inputs.permute(1, 0, 2), tgt_actions[:-1,])
        logits = outputs[:, :, :-2].permute(1, 0, 2)
        logits += B.priors
        probs = torch.nn.Softmax(dim=2)(logits)
        if any(torch.isnan(torch.reshape(probs, (-1,)))):
            raise ValueError
            # probs[torch.isinf(logits)] = 0
        logprobs = torch.nn.LogSoftmax(dim=2)(logits)
        if any(torch.isnan(torch.reshape(logprobs, (-1,)))):
            raise ValueError

        # Generate mask from sequence lengths
        # NOTE: Using this mask for neglogp and entropy actually does NOT
        # affect training because gradients are zero outside the lengths.
        # However, the mask makes tensorflow summaries accurate.
        mask = sequence_mask(B.lengths, maxlen=self.max_length, dtype=torch.float32)

        # Negative log probabilities of sequences
        actions_one_hot = torch.nn.functional.one_hot(B.actions.to(torch.long), num_classes=self.n_choices)
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, dim=2)  # Sum over action dim

        neglogp = torch.sum(neglogp_per_step * mask, dim=1)  # Sum over current_length dim

        # NOTE 1: The above implementation is the same as the one below:
        # neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=actions)
        # neglogp = tf.reduce_sum(neglogp_per_step, dim=1) # Sum over current_length
        # NOTE 2: The above implementation is also the same as the one below, with a few caveats:
        #   Exactly equivalent when removing priors.
        #   Equivalent up to precision when including clipped prior.
        #   Crashes when prior is not clipped due to multiplying zero by -inf.
        # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(priors, -2.4e38, 0)) * actions_one_hot
        # neglogp_per_step = tf.reduce_sum(neglogp_per_step, dim=2)
        # neglogp = tf.reduce_sum(neglogp_per_step, dim=1) # Sum over current_length

        # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        entropy_gamma_decay_mask = self.entropy_gamma_decay * mask  # ->(batch_size, max_length)
        # Sum over action dim -> (batch_size, max_length)
        entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)
        # Sum over current_length dim -> (batch_size, )
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1)

        return neglogp, entropy

    def _train_loss(self, b, sampled_batch_ph, pqt_batch_ph=None, test=False):
        # Setup losses
        neglogp, entropy = self.make_neglogp_and_entropy(sampled_batch_ph, test=test)
        r = sampled_batch_ph.rewards

        # Entropy loss
        entropy_loss = -self.entropy_weight * torch.mean(entropy)
        loss = entropy_loss

        if not self.pqt or (self.pqt and self.pqt_use_pg):
            # Baseline is the worst of the current samples r
            pg_loss = torch.mean((r - b) * neglogp)
            # Loss already is set to entropy loss
            loss += pg_loss

        # Priority queue training loss
        if self.pqt:
            pqt_neglogp, _ = self.make_neglogp_and_entropy(pqt_batch_ph, test=test)
            pqt_loss = self.pqt_weight * torch.mean(pqt_neglogp)
            loss += pqt_loss

        return loss, None

    def _compute_probs(self, memory_batch_ph, log=False):
        # Memory batch
        memory_neglogp, _ = self.make_neglogp_and_entropy(memory_batch_ph)
        if log:
            return -memory_neglogp
        else:
            return torch.exp(-memory_neglogp)

    def sample(self, n, input_=None):
        """Sample batch of n expressions"""

        actions, obs, priors = self._sample(n, input_=input_)
        return actions.cpu().numpy(), obs.cpu().numpy(), priors.cpu().numpy()

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""
        probs = self._compute_probs(numpy_batch_to_tensor_batch(memory_batch), log=log)
        return probs.cpu().numpy()

    def train_loss(self, b, sampled_batch, pqt_batch, test=False):
        """Computes loss, trains model, and returns summaries."""
        loss, summaries = self._train_loss(
            torch.tensor(b).to(DEVICE),
            numpy_batch_to_tensor_batch(sampled_batch),
            numpy_batch_to_tensor_batch(pqt_batch),
            test=test,
        )
        return loss, summaries


# pylint: disable-next=abstract-method
class TransformerTreeEncoderController(nn.Module):
    """
    Recurrent neural network (RNN) controller used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees. It is trained using REINFORCE with baseline.

    Parameters
    ----------
    prior : dso.prior.JointPrior
        JointPrior object used to adjust probabilities during sampling.

    state_manager: dso.state_manager.StateManager
        Object that handles the state features to be used

    summary : bool
        Write tensorboard summaries?

    debug : int
        Debug level, also used in learn(). 0: No debug. 1: logger.info shapes and
        number of parameters for each variable.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    initializer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

    optimizer : str
        Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

    learning_rate : float
        Learning rate for optimizer.

    entropy_weight : float
        Coefficient for entropy bonus.

    entropy_gamma : float or None
        Gamma in entropy decay. None (or
        equivalently, 1.0) turns off entropy decay.

    pqt : bool
        Train with priority queue training (PQT)?

    pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?

    max_length : int or None
        Maximum sequence length. This will be overridden if a LengthConstraint
        with a maximum length is part of the prior.

    batch_size: int
        Most likely 500 or 1000
    """

    def __init__(
        self,
        prior,
        library,
        task,
        cfg,
        config_state_manager=None,
        encoder_input_dim=None,
        debug=0,
        summary=False,
        # RNN cell hyperparameters
        cell="lstm",
        num_layers=1,
        num_units=32,
        # Optimizer hyperparameters
        optimizer="adam",
        initializer="zeros",
        learning_rate=0.001,
        # Loss hyperparameters
        entropy_weight=0.005,
        entropy_gamma=1.0,
        # PQT hyperparameters
        pqt=False,
        pqt_k=10,
        pqt_batch_size=1,
        pqt_weight=200.0,
        pqt_use_pg=False,
        # Other hyperparameters
        max_length=30,
        batch_size=1000,
        n_objects=1,
        has_encoder=True,
        rl_weight=1.0,
        randomize_ce=False,
    ):
        super(TransformerTreeEncoderController, self).__init__()
        self.encoder_input_dim = encoder_input_dim
        self.learning_rate = learning_rate
        self.rl_weight = rl_weight
        self.randomize_ce = randomize_ce

        self.prior = prior
        self.summary = summary
        # self.rng = np.random.RandomState(0) # Used for PPO minibatch sampling
        self.n_objects = n_objects
        self.num_units = num_units

        lib = library

        # Find max_length from the LengthConstraint prior, if it exists
        # Both priors will never happen in the same experiment
        prior_max_length = None
        for single_prior in self.prior.priors:
            if isinstance(single_prior, LengthConstraint):
                if single_prior.max is not None:
                    prior_max_length = single_prior.max
                    self.max_length = prior_max_length
                break

        if prior_max_length is None:
            assert max_length is not None, "max_length must be specified if " "there is no LengthConstraint."
            self.max_length = max_length
            logger.info(
                "WARNING: Maximum length not constrained. Sequences will "
                "stop at {} and complete by repeating the first input "
                "variable.".format(self.max_length)
            )
        elif max_length is not None and max_length != self.max_length:
            logger.info(
                "WARNING: max_length ({}) will be overridden by value from "
                "LengthConstraint ({}).".format(max_length, self.max_length)
            )
        self.max_length *= self.n_objects
        max_length = self.max_length

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.pqt = pqt
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg

        self.n_choices = lib.L

        self.batch_size = batch_size
        # self.baseline = torch.zeros(1)

        # Entropy decay vector
        if entropy_gamma is None:
            entropy_gamma = 1.0
        # Could make this a tensor
        self.entropy_gamma_decay = torch.Tensor([entropy_gamma**t for t in range(max_length)]).to(DEVICE)

        # Build controller RNN
        # Create recurrent cell
        if isinstance(num_units, int):
            num_units = num_units * num_layers

        if "type" in config_state_manager:  # pyright: ignore
            del config_state_manager["type"]  # pyright: ignore
        self.state_manager = TorchHierarchicalStateManager(
            library,
            max_length,
            **config_state_manager,  # pyright: ignore
        )

        # Calculate input size
        self.input_dim_size = self.state_manager.input_dim_size

        # (input_size, hidden_size)

        self.task = task

        # Transformer
        # if self.input_dim_size % 2 != 0:
        #     log_and_print('Error, input_size should be divisible by 2')
        #     raise ValueError("Error, input_size should be divisible by 2")

        self.tgt_padding_token = self.n_choices
        self.sos_token = self.n_choices + 1
        out_dim = self.n_choices + 2  # EOS token at last index
        # if out_dim % 2 != 0:
        #     log_and_print('Error, out_dim should be divisible by 2')
        #     raise ValueError("Error, out_dim should be divisible by 2")
        self.out_dim = out_dim

        # in_dim = int(np.ceil(in_dim / 2) * 2)
        # out_dim = int(np.ceil(out_dim / 2) * 2)
        # num_units = hidden_units
        self.model = TransformerCustomEncoderModel(
            self.input_dim_size,
            self.out_dim,
            num_units,
            cfg,
            enc_layers=3,
            dec_layers=2,
            dropout=0,
            input_already_encoded=True,
            output_pdt=self.tgt_padding_token,
            has_encoder=has_encoder,
        )

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tgt_padding_token)

    def _sample(self, batch_size=None, input_=None):
        if batch_size is None:
            batch_size = self.batch_size
            self.batch_size = batch_size
        initial_obs = self.task.reset_task(self.prior)
        initial_obs = initial_obs.expand(batch_size, initial_obs.shape[0])
        initial_obs = self.state_manager.process_state(initial_obs)

        # Get initial prior
        initial_prior = torch.from_numpy(self.prior.initial_prior()).to(DEVICE)
        initial_prior = initial_prior.expand(batch_size, self.n_choices)

        # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
        current_length = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        obs = initial_obs
        next_input = self.state_manager.get_tensor_input(obs)
        # Could add a start token for inputs - none at present
        inputs = next_input.unsqueeze(0)
        obs_ta = []
        priors_ta = []
        prior = initial_prior
        lengths = torch.ones(batch_size, dtype=torch.int32).to(DEVICE)
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        # Initial state
        while not all(finished):
            current_length += 1
            output = self.model(input_, inputs, tgt_actions)
            cell_output = output[-1, :, :-2]
            logits = cell_output + prior
            action = torch.distributions.categorical.Categorical(logits=logits).sample()
            tgt_actions = torch.cat((tgt_actions, action.view(1, -1)), 0)

            # Compute obs and prior
            next_obs, next_prior = GetNextObs.apply(self, tgt_actions[1:, :].permute(1, 0), obs)  # pyright: ignore
            next_obs = self.state_manager.process_state(next_obs)
            next_input = self.state_manager.get_tensor_input(next_obs)
            inputs = torch.cat((inputs, next_input.unsqueeze(0)), 0)

            obs_ta.append(obs)
            priors_ta.append(prior)
            # finished = torch.logical_or(
            #     finished,
            #     current_length >= max_length)
            finished = finished + (current_length >= self.max_length)
            next_lengths = torch.where(finished, lengths, (current_length + 1).expand(batch_size))  # Ever finished
            # (current_length + 1).unsqueeze(0).expand(batch_size, 1))
            obs = next_obs
            prior = next_prior
            lengths = next_lengths

        actions = tgt_actions[1:, :].permute(1, 0)
        # (?, obs_dim, max_length)
        obs = torch.stack(obs_ta).permute(1, 2, 0)
        # (?, max_length, n_choices)
        priors = torch.stack(priors_ta, 1)
        return actions, obs, priors

    def compute_neg_log_likelihood(self, data_to_encode, true_action, B=None):
        inputs_ = None
        batch_size = 1
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int64).to(DEVICE) * self.sos_token
        actions = torch.tensor(true_action).to(DEVICE).view(1, -1)
        tgt_actions = torch.cat((tgt_actions, actions.permute(1, 0)), 0)

        outputs = self.model(data_to_encode, inputs_, tgt_actions[:-1,])
        logits = outputs[:, :, :-2]
        neg_log_likelihood = self.ce_loss(logits.permute(1, 2, 0), tgt_actions[1:, :].T)
        return neg_log_likelihood

    def train_mle_loss(self, input_, token_eqs):
        # inputs_ = None
        # outputs = self.model(data_to_encode, inputs_, token_eqs[:-1, :])
        # logits = outputs[:, :, :-2]
        # # token_eqs[1:, :]
        # loss = self.ce_loss(logits.permute(1, 2, 0), token_eqs[1:, :].T)
        batch_size = input_.shape[0]
        initial_obs = self.task.reset_task(self.prior)
        initial_obs = initial_obs.expand(batch_size, initial_obs.shape[0])
        initial_obs = self.state_manager.process_state(initial_obs)

        # Get initial prior
        initial_prior = torch.from_numpy(self.prior.initial_prior()).to(DEVICE)
        initial_prior = initial_prior.expand(batch_size, self.n_choices)

        # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
        current_length = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        obs = initial_obs
        next_input = self.state_manager.get_tensor_input(obs)
        # Could add a start token for inputs - none at present
        inputs = next_input.unsqueeze(0)
        obs_ta = []
        priors_ta = []
        step_losses = []
        prior = initial_prior
        lengths = torch.ones(batch_size, dtype=torch.int32).to(DEVICE)
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        # Initial state
        while not all(finished):
            current_length += 1
            if self.randomize_ce:
                use_ground_truth = (
                    (torch.rand(batch_size) > 0.25).to(DEVICE).long()
                )  # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
                tgt_actions = use_ground_truth * token_eqs[:current_length, :] + (1 - use_ground_truth) * tgt_actions
            else:
                tgt_actions = token_eqs[:current_length, :]
            output = self.model(input_, inputs, tgt_actions)
            cell_output = output[-1, :, :-2]
            logits = cell_output + prior
            logits[logits == float("-inf")] = 0
            step_losses.append(self.ce_loss(logits, token_eqs[current_length, :]))
            action = torch.distributions.categorical.Categorical(logits=logits).sample()
            tgt_actions = torch.cat((tgt_actions, action.view(1, -1)), 0)

            # Compute obs and prior
            next_obs, next_prior = GetNextObs.apply(self, tgt_actions[1:, :].permute(1, 0), obs)  # pyright: ignore
            next_obs = self.state_manager.process_state(next_obs)
            next_input = self.state_manager.get_tensor_input(next_obs)
            inputs = torch.cat((inputs, next_input.unsqueeze(0)), 0)

            obs_ta.append(obs)
            priors_ta.append(prior)
            # finished = torch.logical_or(
            #     finished,
            #     current_length >= max_length)
            finished = finished + ((current_length >= self.max_length) or (current_length == (token_eqs.shape[0] - 1)))
            next_lengths = torch.where(finished, lengths, (current_length + 1).expand(batch_size))  # Ever finished
            # (current_length + 1).unsqueeze(0).expand(batch_size, 1))
            obs = next_obs
            prior = next_prior
            lengths = next_lengths
        mle_loss = torch.stack(step_losses).mean()
        return mle_loss

    def make_neglogp_and_entropy(self, B, test=False):
        # Generates tensor for neglogp of a given batch
        # Loop_fn is defined in the function:
        # Essentially only calculating up to the sequence_lengths given:
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/raw_rnn
        inputs = self.state_manager.get_tensor_input(B.obs)
        sequence_length = B.lengths  # pylint: disable=unused-variable  # noqa: F841
        batch_size = B.obs.shape[0]
        data_to_encode = B.data_to_encode
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        actions = B.actions
        tgt_actions = torch.cat((tgt_actions, actions.permute(1, 0)), 0)

        outputs = self.model(data_to_encode, inputs.permute(1, 0, 2).float(), tgt_actions[:-1,])
        logits = outputs[:, :, :-2].permute(1, 0, 2)
        logits += B.priors
        probs = torch.nn.Softmax(dim=2)(logits)
        if any(torch.isnan(torch.reshape(probs, (-1,)))):
            raise ValueError
            # probs[torch.isinf(logits)] = 0
        logprobs = torch.nn.LogSoftmax(dim=2)(logits)
        if any(torch.isnan(torch.reshape(logprobs, (-1,)))):
            raise ValueError

        # Generate mask from sequence lengths
        # NOTE: Using this mask for neglogp and entropy actually does NOT
        # affect training because gradients are zero outside the lengths.
        # However, the mask makes tensorflow summaries accurate.
        mask = sequence_mask(B.lengths, maxlen=self.max_length, dtype=torch.float32)

        # Negative log probabilities of sequences
        actions_one_hot = torch.nn.functional.one_hot(B.actions.to(torch.long), num_classes=self.n_choices)
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, dim=2)  # Sum over action dim

        neglogp = torch.sum(neglogp_per_step * mask, dim=1)  # Sum over current_length dim

        # NOTE 1: The above implementation is the same as the one below:
        # neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=actions)
        # neglogp = tf.reduce_sum(neglogp_per_step, dim=1) # Sum over current_length
        # NOTE 2: The above implementation is also the same as the one below, with a few caveats:
        #   Exactly equivalent when removing priors.
        #   Equivalent up to precision when including clipped prior.
        #   Crashes when prior is not clipped due to multiplying zero by -inf.
        # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(priors, -2.4e38, 0)) * actions_one_hot
        # neglogp_per_step = tf.reduce_sum(neglogp_per_step, dim=2)
        # neglogp = tf.reduce_sum(neglogp_per_step, dim=1) # Sum over current_length

        # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        entropy_gamma_decay_mask = self.entropy_gamma_decay * mask  # ->(batch_size, max_length)
        # Sum over action dim -> (batch_size, max_length)
        entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)
        # Sum over current_length dim -> (batch_size, )
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1)

        return neglogp, entropy

    def _train_loss(self, b, sampled_batch_ph, pqt_batch_ph=None, test=False):
        # Setup losses
        neglogp, entropy = self.make_neglogp_and_entropy(sampled_batch_ph, test=test)
        r = sampled_batch_ph.rewards

        # Entropy loss
        entropy_loss = -self.entropy_weight * torch.mean(entropy)
        loss = entropy_loss

        if not self.pqt or (self.pqt and self.pqt_use_pg):
            # Baseline is the worst of the current samples r
            pg_loss = torch.mean((r - b) * neglogp)
            # Loss already is set to entropy loss
            loss += pg_loss

        # Priority queue training loss
        if self.pqt:
            pqt_neglogp, _ = self.make_neglogp_and_entropy(pqt_batch_ph, test=test)
            pqt_loss = self.pqt_weight * torch.mean(pqt_neglogp)
            loss += pqt_loss

        if self.rl_weight != 1.0 and sampled_batch_ph.tgt.size != 0:
            mle_loss = self.train_mle_loss(sampled_batch_ph.data_to_encode, sampled_batch_ph.tgt)
            total_loss = self.rl_weight * loss + (1 - self.rl_weight) * mle_loss
            mle_loss_out = mle_loss.item()
        else:
            total_loss = loss
            mle_loss_out = None
        return total_loss, mle_loss_out

    def _compute_probs(self, memory_batch_ph, log=False):
        # Memory batch
        memory_neglogp, _ = self.make_neglogp_and_entropy(memory_batch_ph)
        if log:
            return -memory_neglogp
        else:
            return torch.exp(-memory_neglogp)

    def sample(self, n, input_=None):
        """Sample batch of n expressions"""

        actions, obs, priors = self._sample(n, input_=input_)
        return actions.cpu().numpy(), obs.cpu().numpy(), priors.cpu().numpy()

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""
        probs = self._compute_probs(numpy_batch_to_tensor_batch(memory_batch), log=log)
        return probs.cpu().numpy()

    def train_loss(self, b, sampled_batch, pqt_batch, test=False):
        """Computes loss, trains model, and returns mle_loss if not None."""
        loss, mle_loss = self._train_loss(
            torch.tensor(b).to(DEVICE),
            numpy_batch_to_tensor_batch(sampled_batch),
            numpy_batch_to_tensor_batch(pqt_batch),
            test=test,
        )
        return loss, mle_loss


if __name__ == "__main__":
    in_dim = 19
    out_dim = 11

    # in_dim = int(np.ceil(in_dim / 2) * 2)
    # out_dim = int(np.ceil(out_dim / 2) * 2)
    hidden = 32
    model = TransformerModel(
        in_dim, out_dim, hidden, enc_layers=3, dec_layers=1, dropout=0.1, input_already_encoded=True
    )
    # Data : (seq, batch, feature)
    batch_size = 300
    src = torch.randint(0, 1 + 1, (1, batch_size, in_dim))
    tgt = torch.randint(0, out_dim, (1, batch_size))

    out = model(src, tgt)
    assert out.shape[0] == 1  # Length
    assert out.shape[1] == batch_size  # Batch size
    assert out.shape[2] == out_dim
    print("passed tests")
