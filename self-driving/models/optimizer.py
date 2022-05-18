import torch


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        """
            Returns the state of the warmup scheduler as dict.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """
            Loads the warmup scheduler's state.

            Args:
                state_dict: warmup scheduler state.
        """
        self.__dict__.update(state_dict)

    def step(self):
        """
            Updates parameters and rate.
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
            Changing the rate.
        """
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        """
            Zero-ing out the gradients.
        """
        self.optimizer.zero_grad()


def get_optimizer(model):
    """Intializing the optimizer."""

    def get_model_size(model):
        """Computing the size of the model"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buff in model.buffers():
            buffer_size += buff.nelement() * buff.element_size()

        return (param_size + buffer_size) / 1024**2

    return NoamOpt(get_model_size(model), 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
