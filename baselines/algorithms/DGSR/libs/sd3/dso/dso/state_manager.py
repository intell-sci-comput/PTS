from abc import ABC, abstractmethod
import torch

from dso.program import Program
from torch import nn


class StateManager(ABC):
    """
    An interface for handling the tf.Tensor inputs to the Controller.
    """

    def setup_manager(self, controller):
        """
        Function called inside the controller to perform the needed initializations (e.g., if the tf context is needed)
        :param controller the controller class
        """
        self.controller = controller
        self.max_length = controller.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tesnor input for the
        Controller, e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : np.ndarray (dtype=np.float32)
            Observation coming from the Task.

        Returns
        --------
        input_ : tf.Tensor (dtype=tf.float32)
            Tensor to be used as input to the Controller.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(library, config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the Controller.
    """
    manager_dict = {
        "hierarchical": HierarchicalStateManager
    }

    if config is None:
        config = {}

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(library, **config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, library, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_size=8):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = library

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size

    def setup_manager(self, controller):
        super().setup_manager(controller)
        # Create embeddings if needed
        if self.embedding:
            if self.observe_action:
                self.action_embeddings = torch.nn.Embedding(
                    self.library.n_action_inputs, self.embedding_size)
            if self.observe_parent:
                self.parent_embeddings = torch.nn.Embedding(
                    self.library.n_parent_inputs, self.embedding_size)
            if self.observe_sibling:
                self.sibling_embeddings = torch.nn.Embedding(
                    self.library.n_sibling_inputs, self.embedding_size)
        return self

    def get_tensor_input(self, obs):
        observations = []
        action, parent, sibling, dangling = torch.unbind(obs, dim=1)

        # Cast action, parent, sibling to int for embedding_lookup or nn.functional.one_hot
        action = action.to(torch.int64)
        parent = parent.to(torch.int64)
        sibling = sibling.to(torch.int64)

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = self.action_embeddings(action)
            else:
                x = torch.nn.functional.one_hot(
                    action, num_classes=self.library.n_action_inputs).to(torch.float32)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = self.parent_embeddings(parent)
            else:
                x = torch.nn.functional.one_hot(
                    parent, num_classes=self.library.n_parent_inputs).to(torch.float32)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = self.sibling_embeddings(sibling)
            else:
                x = torch.nn.functional.one_hot(
                    sibling, num_classes=self.library.n_sibling_inputs).to(torch.float32)
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            x = torch.unsqueeze(dangling, dim=-1)
            observations.append(x)

        input_ = torch.cat(observations, -1)
        return input_


class TorchHierarchicalStateManager(nn.Module):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, library, max_length, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_size=8):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        super(TorchHierarchicalStateManager, self).__init__()
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = library
        self.max_length = max_length

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size
        if self.embedding:
            if self.observe_action:
                self.action_embeddings = torch.nn.Embedding(
                    self.library.n_action_inputs, self.embedding_size)
            if self.observe_parent:
                self.parent_embeddings = torch.nn.Embedding(
                    self.library.n_parent_inputs, self.embedding_size)
            if self.observe_sibling:
                self.sibling_embeddings = torch.nn.Embedding(
                    self.library.n_sibling_inputs, self.embedding_size)

        if self.embedding:
            input_dim_size = 0
            input_dim_size += self.embedding_size if self.observe_action else 0
            input_dim_size += self.embedding_size if self.observe_parent else 0
            input_dim_size += self.embedding_size if self.observe_sibling else 0
            input_dim_size += 1 if self.observe_dangling else 0
        else:
            input_dim_size = 0
            input_dim_size += self.library.n_action_inputs if self.observe_action else 0
            input_dim_size += self.library.n_parent_inputs if self.observe_parent else 0
            input_dim_size += self.library.n_sibling_inputs if self.observe_sibling else 0
            input_dim_size += 1 if self.observe_dangling else 0
        self.input_dim_size = input_dim_size

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs

    def get_tensor_input(self, obs):
        observations = []
        action, parent, sibling, dangling = torch.unbind(obs, dim=1)

        # Cast action, parent, sibling to int for embedding_lookup or nn.functional.one_hot
        action = action.to(torch.int64)
        parent = parent.to(torch.int64)
        sibling = sibling.to(torch.int64)

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = self.action_embeddings(action)
            else:
                x = torch.nn.functional.one_hot(
                    action, num_classes=self.library.n_action_inputs).to(torch.float32)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = self.parent_embeddings(parent)
            else:
                x = torch.nn.functional.one_hot(
                    parent, num_classes=self.library.n_parent_inputs).to(torch.float32)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = self.sibling_embeddings(sibling)
            else:
                x = torch.nn.functional.one_hot(
                    sibling, num_classes=self.library.n_sibling_inputs).to(torch.float32)
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            x = torch.unsqueeze(dangling, dim=-1)
            observations.append(x)

        input_ = torch.cat(observations, -1)
        return input_
