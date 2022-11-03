import torch

class ScalarQuantization:  # mid-tread
    def __init__(self, args):
        self.codebook_len = 2 ** args.R + 1
        self.delta = (2 * args.gamma) / self.codebook_len  # quantization levels spacing
        self.egde = args.gamma - (self.delta/2)
        self.codebook = [-self.egde + i * self.delta for i in range(self.codebook_len)]
        self.hot_vector = torch.ones(self.codebook_len)
        self.hot_vector[0:int((self.codebook_len-1)/2)] = -1

    def __call__(self, input, malicious):
        # quantization
        q_input = self.delta * torch.round(input / self.delta)
        q_input[q_input >= self.egde] = self.egde
        q_input[q_input <= -self.egde] = -self.egde

        # 1-bit compression
        random_vecs = torch.zeros(len(q_input), self.codebook_len)
        for i in range(len(q_input)):
            index = self.codebook.index(q_input[i])
            random_vec = self.hot_vector[torch.randperm(self.codebook_len)]
            if malicious is None:
                random_vecs[i, :] = random_vec if random_vec[index] == 1 else (-1) * random_vec
            elif malicious == 'ones':
                random_vecs[i, :] = random_vec
            elif malicious == 'flip':
                random_vecs[i, :] = random_vec if random_vec[index] == 1 else (-1) * random_vec
                random_vecs[i, :] = (-1) * random_vecs[i, :]
        return random_vecs

def signSGD(input, malicious):
    output = torch.sign(input)
    output[output == 0] = 1
    if malicious is None:
        return output
    elif malicious == 'ones':
        return torch.abs(output)
    elif malicious == 'flip':
        return (-1) * output
