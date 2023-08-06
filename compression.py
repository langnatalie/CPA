import numpy as np
import torch
import privacy


class scalar:  # mid-tread
    def __init__(self, args):
        self.codebook_len = 2 ** args.R + 1
        self.delta = (2 * args.gamma) / self.codebook_len  # quantization levels spacing
        self.codebook = torch.tensor([i * self.delta for i in np.arange(self.codebook_len) - int(self.codebook_len / 2)]).float().to(args.device)
        self.hot_vector = torch.ones(self.codebook_len).to(args.device)
        self.hot_vector[0:int((self.codebook_len - 1) / 2)] = -1
        self.privacy = privacy.RR(args) if args.privacy == 'RR' else None
        self.malicious = args.malicious_type

    def __call__(self, input, user_idx=None):
        # quantization
        q_input = self.quantization(input)

        # 1-bit compression
        repeated_codebook = self.codebook.repeat(len(q_input), 1)
        repeated_hot_vector = self.hot_vector.repeat(len(q_input), 1)

        # get the idx of the quantized words
        q_input_words_idx = ((q_input.unsqueeze(-1) == repeated_codebook).nonzero())[:, -1]

        # randomize Bernoulli vectors
        indices = torch.argsort(torch.rand(repeated_hot_vector.shape), dim=-1).to(repeated_hot_vector.device)
        random_vecs = torch.gather(repeated_hot_vector, dim=-1, index=indices)

        # decide upon sending to sever 1 or -1 depending on the quantized word index entry
        quantized_word_index_entry = torch.gather(random_vecs, dim=-1, index=q_input_words_idx.unsqueeze(-1))
        if self.privacy is not None:
            quantized_word_index_entry = self.privacy(quantized_word_index_entry)
        if user_idx is not None:
            if self.malicious == 'ones':
                quantized_word_index_entry[:] = 1
            else:
                flippings = torch.bernoulli(0.5*torch.ones_like(quantized_word_index_entry))
                flippings[flippings == 0] = -1
                quantized_word_index_entry *= flippings
        random_vecs = quantized_word_index_entry * random_vecs

        return random_vecs

    def quantization(self, input):
        q_input = self.delta * torch.round(input / self.delta).int()
        q_input[q_input >= self.codebook[-1]] = self.codebook[-1]
        q_input[q_input <= self.codebook[0]] = self.codebook[0]
        return q_input

class nested:
    def __init__(self, args):
        # outer (coarse) quantizer - small quantizer 1
        args.R = args.R_coarse
        self.quantizer_coarse = scalar(args)

        # inner (fine) quantizer - small quantizer 2
        args.R = np.log2(args.R_fine - 1).astype(int)
        args.gamma = self.quantizer_coarse.delta / 2
        self.quantizer_fine = scalar(args)

    def __call__(self, input, user_idx=None):
        result_coarse = self.quantizer_coarse(input)
        result_fine = self.quantizer_fine(input - self.quantizer_coarse.quantization(input))
        return result_coarse, result_fine


def sign(input):
    output = torch.sign(input)
    output[output == 0] = 1
    return output


