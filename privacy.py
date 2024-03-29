import torch
import numpy as np

class RR:
    def __init__(self, args):
        self.p = torch.exp(torch.tensor(args.epsilon))*((1+torch.exp(torch.tensor(args.epsilon)))**(-1))

    def __call__(self, input):
        randomize_bits = np.random.binomial(1, self.p, input.shape[0])
        randomize_bits[randomize_bits == 0] = -1
        return input*torch.tensor(randomize_bits, device=input.device).unsqueeze(-1) if len(input.shape) != 1 \
            else input*torch.tensor(randomize_bits, device=input.device)


class Laplace:
    def __init__(self, args):
        self.noise = torch.distributions.laplace.Laplace(loc=0, scale=2/args.epsilon)

    def __call__(self, input):

        # compute statistics
        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.norm(input - mean) / (input.shape[-1] ** 0.5)

        std = 3 * std
        input = (input - mean) / std

        # normalize
        input = input + self.noise.sample(input.shape).to(input.dtype).to(input.device)

        # denormalize
        input = (input * std) + mean

        return input
