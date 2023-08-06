import copy
import torch
import compression
import privacy


class FedAvg:  # Compression Privacy class
    def __init__(self, args):
        # choose quantization method: scalar, vector, nested
        self.compression = compression.sign if args.compression is not None else None

        if args.privacy is not None:
            self.privacy = privacy.RR(args) if args.privacy == 'RR' else privacy.Laplace(args)
        else:
            self.privacy = None

    def __call__(self, input):
        if self.compression is not None:
            input = self.compression(input)
        if self.privacy is not None:
            input = self.privacy(input)
        return input

    def FL(self, local_models, global_model):
        state_dict = copy.deepcopy(global_model.state_dict())
        SNR_layers = []
        for key in state_dict.keys():
            local_weights_average = torch.zeros_like(state_dict[key].view(-1))
            local_weights_orig_average = torch.zeros_like(state_dict[key].view(-1))

            for user_idx in range(0, len(local_models)):
                local_weights_orig = local_models[user_idx]['model'].state_dict()[key].view(-1) - state_dict[key].view(
                    -1)
                local_weights_orig_average += local_weights_orig
                local_weights = self(local_weights_orig)
                local_weights_average += local_weights

            local_weights_orig_average = local_weights_orig_average / len(local_models)
            local_weights_average = local_weights_average / len(local_models)
            SNR_layers.append(
                torch.var(local_weights_orig_average) / torch.var(local_weights_orig_average - local_weights_average))
            state_dict[key] += local_weights_average.reshape(state_dict[key].shape).to(state_dict[key].dtype)

        global_model.load_state_dict(copy.deepcopy(state_dict))
        return torch.mean(torch.stack(SNR_layers))