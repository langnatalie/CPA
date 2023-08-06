import torch
import copy
import compression
from numpy.random import randint

class CPA:  # Compression Privacy class
    def __init__(self, args):
        # choose quantization method: scalar, vector, nested
        self.cpa = compression.nested(args) if args.compression == 'nested' else compression.scalar(args)
        self.device = args.device
        if args.malicious:
            self.malicious_idx = randint(low=0, high=args.num_users, size=round(args.malicious_users_percent*args.num_users))
        else:
            self.malicious_idx = []

    def __call__(self, input, user_idx=None):
        return self.cpa(input, user_idx)

    def single(self, local_models, global_model):
        state_dict = copy.deepcopy(global_model.state_dict())
        SNR_layers = []
        for key in state_dict.keys():
            histogram = torch.zeros(len(state_dict[key].view(-1)),
                                    self.cpa.codebook_len,
                                    len(local_models)).to(self.device)
            local_weights_orig_average = torch.zeros_like(state_dict[key].view(-1))

            for user_idx in range(0, len(local_models)):
                local_weights_orig = local_models[user_idx]['model'].state_dict()[key].view(-1) - state_dict[key].view(
                    -1)
                local_weights_orig_average += local_weights_orig
                local_weights = self(local_weights_orig, user_idx) if user_idx in self.malicious_idx else self(local_weights_orig)
                histogram[:, :, user_idx] = local_weights

            histogram = torch.mean(histogram, dim=-1)  # averaging over the users
            histogram *= self.cpa.codebook
            histogram = torch.sum(histogram, dim=-1)  # weighted average

            local_weights_orig_average = local_weights_orig_average / len(local_models)
            SNR_layers.append(torch.var(local_weights_orig_average) / torch.var(local_weights_orig_average - histogram))
            state_dict[key] += histogram.reshape(state_dict[key].shape).to(state_dict[key].dtype)  # global model update

        global_model.load_state_dict(copy.deepcopy(state_dict))
        return torch.mean(torch.stack(SNR_layers))

    def nested(self, local_models, global_model):
        state_dict = copy.deepcopy(global_model.state_dict())
        SNR_layers = []
        for key in state_dict.keys():
            histogram1 = torch.zeros(len(state_dict[key].view(-1)),
                                     self.cpa.quantizer_coarse.codebook_len,
                                     len(local_models)).to(self.device)

            histogram2 = torch.zeros(len(state_dict[key].view(-1)),
                                     self.cpa.quantizer_fine.codebook_len,
                                     len(local_models)).to(self.device)
            local_weights_orig_average = torch.zeros_like(state_dict[key].view(-1))

            for user_idx in range(0, len(local_models)):
                local_weights_orig = local_models[user_idx]['model'].state_dict()[key].view(-1) - state_dict[key].view(
                    -1)
                local_weights_orig_average += local_weights_orig
                local_weights1, local_weights2 = self(local_weights_orig)
                histogram1[:, :, user_idx] = local_weights1
                histogram2[:, :, user_idx] = local_weights2

            # averaging over the users
            histogram1 = torch.mean(histogram1, dim=-1)
            histogram2 = torch.mean(histogram2, dim=-1)

            histogram1 *= self.cpa.quantizer_coarse.codebook.clone().detach()
            histogram1 = torch.sum(histogram1, dim=-1)  # weighted average

            histogram2 *= self.cpa.quantizer_fine.codebook.clone().detach()
            histogram2 = torch.sum(histogram2, dim=-1)  # weighted average

            histogram = histogram1 + histogram2

            local_weights_orig_average = local_weights_orig_average / len(local_models)
            SNR_layers.append(torch.var(local_weights_orig_average) / torch.var(local_weights_orig_average - histogram))
            state_dict[key] += histogram.reshape(state_dict[key].shape).to(state_dict[key].dtype)  # global model update

        global_model.load_state_dict(copy.deepcopy(state_dict))
        return torch.mean(torch.stack(SNR_layers))

