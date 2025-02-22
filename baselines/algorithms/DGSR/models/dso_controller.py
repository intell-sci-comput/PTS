"""Controller used to generate distribution over hierarchical, variable-length objects."""
import logging

import torch
from dso.memory import Batch
from dso.prior import LengthConstraint
from dso.state_manager import TorchHierarchicalStateManager
from torch import nn

logger = logging.getLogger()


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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# pylint: disable-next=abstract-method
class DsoController(nn.Module):
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
        super(DsoController, self).__init__()
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
        self.cell = nn.LSTMCell(self.input_dim_size, num_units)
        self.linear_out = nn.Linear(num_units, self.n_choices)

        self.task = task

    def _sample(self, batch_size=None):
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
        time = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        obs = initial_obs
        next_input = self.state_manager.get_tensor_input(obs)
        actions_ta = []
        obs_ta = []
        priors_ta = []
        prior = initial_prior
        lengths = torch.ones(batch_size, dtype=torch.int32).to(DEVICE)
        # Initial state
        hx = torch.zeros(batch_size, self.num_units).to(DEVICE)  # (batch, hidden_size)
        cx = torch.zeros(batch_size, self.num_units).to(DEVICE)
        while not all(finished):
            time += 1
            next_hx, next_cx = self.cell(next_input, (hx, cx))
            cell_output = self.linear_out(next_hx)
            logits = cell_output + prior
            action = torch.distributions.categorical.Categorical(logits=logits).sample()
            actions_ta.append(action)
            actions = torch.transpose(torch.stack(actions_ta), 0, 1)

            # Compute obs and prior
            next_obs, next_prior = self.task.get_next_obs(actions.cpu().numpy(), obs.cpu().numpy())
            next_prior = torch.from_numpy(next_prior).to(DEVICE).view(-1, self.n_choices)
            next_obs = torch.from_numpy(next_obs).to(DEVICE).view(-1, self.task.OBS_DIM)
            next_obs = self.state_manager.process_state(next_obs)
            next_input = self.state_manager.get_tensor_input(next_obs)
            obs_ta.append(obs)
            priors_ta.append(prior)
            # finished = torch.logical_or(
            #     finished,
            #     time >= max_length)
            finished = finished + (time >= self.max_length)
            next_lengths = torch.where(finished, lengths, (time + 1).expand(batch_size))  # Ever finished
            # (time + 1).unsqueeze(0).expand(batch_size, 1))
            obs = next_obs
            prior = next_prior
            lengths = next_lengths

            # Emit zeros and copy forward state for minibatch entries that are finished.
            hx = torch.where(finished.view(-1, 1), hx, next_hx)
            cx = torch.where(finished.view(-1, 1), cx, next_cx)
            # if any(finished):
            #     pass

        actions = torch.stack(actions_ta, 1)
        # (?, obs_dim, max_length)
        obs = torch.stack(obs_ta).permute(1, 2, 0)
        # (?, max_length, n_choices)
        priors = torch.stack(priors_ta, 1)
        return actions, obs, priors

    def compute_neg_log_likelihood(self, data_to_encode, true_action, B_=None):
        B = numpy_batch_to_tensor_batch(B_)
        inputs = self.state_manager.get_tensor_input(B.obs)  # pyright: ignore
        sequence_length = B.lengths  # pyright: ignore
        batch_size = B.obs.shape[0]  # pyright: ignore

        time = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        # Initial state
        # (batch, hidden_size)
        hx = torch.zeros(batch_size, self.num_units).to(DEVICE)
        cx = torch.zeros(batch_size, self.num_units).to(DEVICE)
        finished_loop = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        elements_finished = time >= sequence_length
        if all(elements_finished):
            next_input = torch.zeros(batch_size, self.input_dim_size).to(DEVICE)
        else:
            next_input = inputs[:, time, :]

        emit_ta = []
        while not all(finished_loop):
            time += 1
            next_hx, next_cx = self.cell(next_input, (hx, cx))
            cell_output = self.linear_out(next_hx)
            elements_finished = time >= sequence_length
            if all(elements_finished):
                next_input = torch.zeros(batch_size, self.input_dim_size).to(DEVICE)
            else:
                next_input = inputs[:, time, :]
            finished_loop = finished_loop + (time >= self.max_length)
            # Emit zeros and copy forward state for minibatch entries that are finished.
            hx = torch.where(finished_loop.view(-1, 1), hx, next_hx)
            cx = torch.where(finished_loop.view(-1, 1), cx, next_cx)
            emit = torch.where(finished_loop.view(-1, 1), torch.zeros_like(cell_output), cell_output).to(DEVICE)
            emit_ta.append(emit)
            # If any new minibatch entries are marked as finished, mark these.
            # finished = tf.logical_or(finished, next_finished) # Skipping may need ?!
            # if any(finished_loop):
            #     hx = hx + 0
            #     pass
            # if any(elements_finished):
            #     hx = hx + 0
            #     pass

        logits = torch.transpose(torch.stack(emit_ta), 0, 1)
        B.actions.to(torch.long)  # pyright: ignore

        actions = torch.tensor(true_action).to(DEVICE).view(1, -1)
        actions_in = torch.zeros(1, logits.shape[1]).to(torch.int64).to(DEVICE)
        actions_in[0][: actions.shape[1]] = actions[0]
        logits_ = torch.mean(logits, axis=0)  # pyright: ignore
        neg_log_likelihood = nn.CrossEntropyLoss()(logits_, actions_in.ravel())
        return neg_log_likelihood

    def make_neglogp_and_entropy(self, B):
        # Generates tensor for neglogp of a given batch
        # Loop_fn is defined in the function:
        # Essentially only calculating up to the sequence_lengths given:
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/raw_rnn
        inputs = self.state_manager.get_tensor_input(B.obs)
        sequence_length = B.lengths
        batch_size = B.obs.shape[0]

        time = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        # Initial state
        # (batch, hidden_size)
        hx = torch.zeros(batch_size, self.num_units).to(DEVICE)
        cx = torch.zeros(batch_size, self.num_units).to(DEVICE)
        finished_loop = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        elements_finished = time >= sequence_length
        if all(elements_finished):
            next_input = torch.zeros(batch_size, self.input_dim_size).to(DEVICE)
        else:
            next_input = inputs[:, time, :]

        emit_ta = []
        while not all(finished_loop):
            time += 1
            next_hx, next_cx = self.cell(next_input, (hx, cx))
            cell_output = self.linear_out(next_hx)
            elements_finished = time >= sequence_length
            if all(elements_finished):
                next_input = torch.zeros(batch_size, self.input_dim_size).to(DEVICE)
            else:
                next_input = inputs[:, time, :]
            finished_loop = finished_loop + (time >= self.max_length)
            # Emit zeros and copy forward state for minibatch entries that are finished.
            hx = torch.where(finished_loop.view(-1, 1), hx, next_hx)
            cx = torch.where(finished_loop.view(-1, 1), cx, next_cx)
            emit = torch.where(finished_loop.view(-1, 1), torch.zeros_like(cell_output), cell_output).to(DEVICE)
            emit_ta.append(emit)
            # If any new minibatch entries are marked as finished, mark these.
            # finished = tf.logical_or(finished, next_finished) # Skipping may need ?!
            # if any(finished_loop):
            #     hx = hx + 0
            #     pass
            # if any(elements_finished):
            #     hx = hx + 0
            #     pass

        logits = torch.transpose(torch.stack(emit_ta), 0, 1)
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

        neglogp = torch.sum(neglogp_per_step * mask, dim=1)  # Sum over time dim

        # NOTE 1: The above implementation is the same as the one below:
        # neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=actions)
        # neglogp = tf.reduce_sum(neglogp_per_step, dim=1) # Sum over time
        # NOTE 2: The above implementation is also the same as the one below, with a few caveats:
        #   Exactly equivalent when removing priors.
        #   Equivalent up to precision when including clipped prior.
        #   Crashes when prior is not clipped due to multiplying zero by -inf.
        # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(priors, -2.4e38, 0)) * actions_one_hot
        # neglogp_per_step = tf.reduce_sum(neglogp_per_step, dim=2)
        # neglogp = tf.reduce_sum(neglogp_per_step, dim=1) # Sum over time

        # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        entropy_gamma_decay_mask = self.entropy_gamma_decay * mask  # ->(batch_size, max_length)
        # Sum over action dim -> (batch_size, max_length)
        entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)
        # Sum over time dim -> (batch_size, )
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1)

        return neglogp, entropy

    def _train_loss(self, b, sampled_batch_ph, pqt_batch_ph=None):
        # Setup losses
        neglogp, entropy = self.make_neglogp_and_entropy(sampled_batch_ph)
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
            pqt_neglogp, _ = self.make_neglogp_and_entropy(pqt_batch_ph)
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

        actions, obs, priors = self._sample(n)
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
        )
        return loss, summaries


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
