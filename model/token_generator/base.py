from abc import abstractmethod, ABC


class TokenGenerator(ABC):
    """
    Init: TokenGenerator()

    It has two method: .step(n_tokens) and .reward(reward)
    """

    @abstractmethod
    def step(self, n_tokens):
        pass

    @abstractmethod
    def reward(self, reward):
        pass
