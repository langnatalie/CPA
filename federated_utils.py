import torch
import torch.optim as optim
import copy
import privacy
import compression


def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = 5  # math.floor(len(train_data) / args.max_num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))


class FedAvg:
    def __init__(self, args):
        self.quantizer = compression.ScalarQuantization(args) if args.quantization else None
        self.privacy = privacy.RR(args) if args.privacy else None

    def __call__(self, input):
        input = self.quantizer(input) if self.quantizer is not None else input
        input = self.privacy(input) if self.privacy is not None else input
        return input


mean = lambda x: sum(x) / len(x)


class Compression_Privacy:  # Compression Privacy class
    def __init__(self, args):
        if args.compression == 'scalarQ':
            self.compression = compression.ScalarQuantization(args)
        elif args.compression == 'signSGD':
            self.compression = compression.signSGD
        else:
            self.compression = None

        if args.privacy == 'Laplace':
            self.privacy = privacy.Laplace(args)
        elif args.privacy == 'RR':
            self.privacy = privacy.RR(args)
        else:
            self.privacy = None
        self.threshold = args.threshold
        self.device = args.device
        self.malicious = args.malicious
        if self.malicious is not None:
            self.malicious_indexes = torch.randperm(args.num_users)[:int(args.num_users * args.malicious_users_percent)]

    def __call__(self, input, malicious=None):
        input = self.compression(input, malicious) if self.compression is not None else input
        input = self.privacy(input) if self.privacy is not None else input
        return input

    def cpa(self, local_models, global_model):
        state_dict = copy.deepcopy(global_model.state_dict())
        SNR_layers = []
        for key in state_dict.keys():
            histogram = torch.zeros(len(state_dict[key].view(-1)),
                                    self.compression.codebook_len,
                                    len(local_models)).to(self.device)
            local_weights_orig_average = torch.zeros_like(state_dict[key].view(-1))

            for user_idx in range(0, len(local_models)):
                local_weights_orig = local_models[user_idx]['model'].state_dict()[key].view(-1) - state_dict[key].view(
                    -1)
                local_weights_orig_average += local_weights_orig
                if self.malicious is not None and user_idx in self.malicious_indexes:
                    local_weights = self(local_weights_orig, self.malicious)
                else:
                    local_weights = self(local_weights_orig)
                histogram[:, :, user_idx] = local_weights

            histogram = torch.mean(histogram, dim=-1)  # averaging over the users
            if self.threshold is not None:
                histogram[histogram <= self.threshold] = 0  # thresholding
            histogram = histogram * torch.tensor(self.compression.codebook, device=self.device)
            histogram = torch.sum(histogram, dim=-1)  # weighted average

            local_weights_orig_average = local_weights_orig_average / len(local_models)
            SNR_layers.append(torch.var(local_weights_orig_average) / torch.var(local_weights_orig_average - histogram))
            state_dict[key] += histogram.reshape(state_dict[key].shape).to(state_dict[key].dtype)  # global model update

        global_model.load_state_dict(copy.deepcopy(state_dict))
        return mean(SNR_layers)

    def fed_avg(self, local_models, global_model):
        state_dict = copy.deepcopy(global_model.state_dict())
        SNR_layers = []
        for key in state_dict.keys():
            local_weights_average = torch.zeros_like(state_dict[key].view(-1))
            local_weights_orig_average = torch.zeros_like(state_dict[key].view(-1))

            for user_idx in range(0, len(local_models)):
                local_weights_orig = local_models[user_idx]['model'].state_dict()[key].view(-1) - state_dict[key].view(-1)
                local_weights_orig_average += local_weights_orig
                if self.malicious is not None and user_idx in self.malicious_indexes:
                    local_weights = self(local_weights_orig, self.malicious)
                else:
                    local_weights = self(local_weights_orig)
                local_weights_average += local_weights

            local_weights_orig_average = local_weights_orig_average / len(local_models)
            local_weights_average = local_weights_average / len(local_models)
            SNR_layers.append(
                torch.var(local_weights_orig_average) / torch.var(local_weights_orig_average - local_weights_average))
            state_dict[key] += local_weights_average.reshape(state_dict[key].shape).to(state_dict[key].dtype)

        global_model.load_state_dict(copy.deepcopy(state_dict))
        return mean(SNR_layers)
